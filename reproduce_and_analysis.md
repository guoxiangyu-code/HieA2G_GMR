# HieA2M (A6, Commit bd283b2) 复现指引与算法解析文档

本文档详述了如何复现 **HieA2M A6 (Phase 1 HTMA, Epoch 10)** 实验的全部步骤，并对当前代码版本进行了深度的算法架构解析。

---

## 1. 快速复现脚本指引

复现当前结果分为三个步骤：**从头训练 (Phase 1)**、**测试集推理** 以及 **自适应门控与分层诊断评估**。

### 1.1 第一步：A6 阶段训练 (从头训练 10 轮)
在当前代码分支下，运行以下脚本启动 Phase 1 训练。该阶段会冻结 FlashVTG 编码器，专门优化新增的 AMC 与 HTMA 模块（`null_anchor` 系数默认为 `1.0`）：

```bash
# 确保在项目根目录下：/home/guoxiangyu/HieA2G_GMR/generalized-moment-retrieval
python -m training.flash_vtg_gmr.train configs/flash_vtg_gmr/model.py \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/label/Standard/train.jsonl \
  --eval_path data/label/Standard/val.jsonl \
  --eval_split_name val \
  --v_feat_dirs /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/slowfast /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip \
  --t_feat_dir /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip_text \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --max_q_l 40 \
  --max_v_l 75 \
  --clip_length 2 \
  --max_windows 5 \
  --lr 1e-4 \
  --lr_drop 10 \
  --wd 1e-4 \
  --n_epoch 10 \
  --max_es_cnt 15 \
  --bsz 128 \
  --eval_bsz 1 \
  --eval_epoch 1 \
  --num_workers 0 \
  --device 0 \
  --results_root results/amc_ft \
  --exp_id amc_ft_focal_soft_a6 \
  --seed 2024 \
  --hidden_dim 256 \
  --dim_feedforward 1024 \
  --enc_layers 3 \
  --t2v_layers 6 \
  --dummy_layers 2 \
  --nheads 8 \
  --num_dummies 40 \
  --total_prompts 10 \
  --num_prompts 1 \
  --kernel_size 5 \
  --num_conv_layers 1 \
  --num_mlp_layers 5 \
  --use_SRM \
  --input_dropout 0.5 \
  --dropout 0.1 \
  --span_loss_type l1 \
  --lw_reg 1.0 \
  --lw_cls 5.0 \
  --lw_sal 0.0 \
  --lw_saliency 0.0 \
  --lw_wattn 1.0 \
  --lw_ms_align 1.0 \
  --mr_only \
  --eval_full_only \
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1.0 \
  --exist_gate_thd 0.5 \
  --nms_thd 0.7 \
  --train_amc_only True \
  --resume /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/checkpoint/flashVTG_gmr
```
*注：`--train_amc_only True` 会自动调用 `setup_model` 逻辑冻结 Backbone 参数，保持 A6 的训练一致性。*

### 1.2 第二步：测试集 (Test Set) 推理
训练结束后，最佳权重会保存在 `results/amc_ft/hl-video_tef-amc_ft_focal_soft_a6-<TIMESTAMP>/model_best.ckpt`。使用以下命令运行测试集推理：

```bash
MODEL_PATH="results/amc_ft/hl-video_tef-amc_ft_focal_soft_a6-<TIMESTAMP>/model_best.ckpt" \
TEST_PATH="data/label/Standard/test.jsonl" \
SLOWFAST_FEAT_DIR="/home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/slowfast" \
CLIP_FEAT_DIR="/home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip" \
TEXT_FEAT_DIR="/home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip_text" \
RESULTS_DIR="results/amc_ft_test_v6_a6" \
DEVICE="0" \
bash scripts/infer_flash_vtg_gmr.sh
```

### 1.3 第三步：自适应门控评测与细粒度诊断
使用我们编写的两个脚本对生成的推理文件进行综合评测：

```bash
# 1. 运行自适应门控门槛评测 (Relative Peak Gating: max(0.5, 0.8 * top1))
python /home/guoxiangyu/.gemini/antigravity-cli/brain/c87f6867-b3a3-43d5-8888-edc49ccee929/scratch/evaluate_amc_gated.py \
  --submission_path results/amc_ft_test_v6_a6/hl_test_submission_nms_thd_0.7.jsonl \
  --gt_path data/label/Standard/test.jsonl \
  --output_path results/amc_ft_test_v6_a6/gated_submission.jsonl \
  --results_json results/amc_ft_test_v6_a6/gated_metrics.json

# 2. 运行混淆矩阵与分层 G-mIoU 诊断脚本
python /home/guoxiangyu/.gemini/antigravity-cli/brain/c87f6867-b3a3-43d5-8888-edc49ccee929/scratch/diagnose_amc.py \
  --submission_path results/amc_ft_test_v6_a6/hl_test_submission_nms_thd_0.7.jsonl \
  --gt_path data/label/Standard/test.jsonl
```

---

## 2. 算法核心架构解析 (Commit bd283b2)

HieA2M (Hierarchical Temporal Moment Alignment) 针对 GMR 任务中“多时刻目标”与“无目标（空集）拒识”的极高难度设计，分层次落地了以下三级算法架构。

```
                [ HieA2M 层次对齐损失体系 ]
                       /        \
         [时序定位与计数]        [层次特征对齐 HTMA]
            /        \                 /      \
      AMC期望计数    Null Anchor     L1-恢复   L2-Sinkhorn   L3-全局 InfoNCE
```

### 2.1 自适应时刻计数器 (AMC, Adaptive Moment Counter)
AMC 旨在解决模型无法自适应估计视频中相关时刻数量的弊端：
1. **输入特征拼接**：拼接了全局 Max-pooled 跨模态特征、时刻分数直方图以及距离矩阵，确保计数器对多时序重叠和分数分布高度敏感。
2. **多损失联合规范**：
   * **Focal Loss**：使用自适应类别权重（`[0.5, 0.7, 0.9, 0.95, 0.97]`）克服空集和多时序目标的类别严重失衡。
   * **软标签损失**：由于时刻边界的模糊性（如1个15秒的片段和2个相邻5秒的片段可能被标注混淆），将类别标签平滑为当前类别 0.6，相邻类别各 0.2。
   * **正样本限制有序回归 (Positive-only Ordinal Regression)**：计算预测期望与真实计数的期望 MSE 损失。仅限制在正样本上计算，移除了空集在序回归下的强压制。
3. **空集锚定损失 (Null Anchor Loss)**：
   在 `GT = 0` 上施加严格不等式约束，当 Pred $\ge 2$ 的概率之和跨度接近 Pred = 0 的概率时产生回推梯度：
   $$\mathcal{L}_{\text{null-anchor}} = \max\left(0,\; \sum_{c \ge 2} p_c - p_0 + 0.3\right)$$

### 2.2 层级时序时刻对齐 (HTMA, Hierarchical Temporal Moment Alignment)
HTMA 是将空集在特征空间中强制推远、并拉近多时刻时序对齐的特征学习核心：
* **Level-1 词-帧恢复损失 (Word-Frame Recovery)**：
  * 数据管线中以 15% 概率随机掩码 Query 中代表动作实体的词汇（如 `saves`, `pass`, `scores`）。
  * 提取 Query 的掩码位置特征，与视频 Proposal 表示进行 Multihead Cross-Attention 交互，强制利用视频的时序区域重建出被掩码的文本词汇（CE 分类）。
  * **空集推远 Margin 损失**：如果视频是负样本（空集），则利用其特征重建词汇的交叉熵必须大于设定阈值（Margin = 0.5），强力在底座特征层面切割正负样本对的交叠。
* **Level-2 segment-moment 匹配损失 (Optimal Transport)**：
  * 通过正则切分提取 Query 语义中的短语块（分出 Subject 与 Action 段）。
  * 构造时序位置编码 `Sinusoidal Positional Encoding` 注入 Segment 特征。
  * 使用可微分的 **Sinkhorn 最优传输** 算法，计算短语段与视频时序窗口的匹配运输图矩阵 $T \in \mathbb{R}^{M \times N}$，并在匹配引导下回传边界梯度，解决多时刻目标的匹配歧义。
* **Level-3 全局图文对齐 (Global InfoNCE)**：
  * 使用 InfoNCE 对抗性对比损失拉近全局视频 Embedding 和全局文本 Embeddings。

### 2.3 自适应相对峰值门控 (Inference Step-6)
推理阶段不仅进行标准的 NMS 过滤，还加入了抗噪门控。当 AMC 分类器估计时刻数量 $\ge 3$ 时，保留 top-C 窗口。对于索引大于等于 3 的窗口，仅在满足下述相对阈值时才予以保留：
$$Score_i > \max(0.5,\; 0.8 \times Score_{top-1})$$
这既能保留真实的多时刻窗口，又能有效避免在长视频片段中被背景噪音和重合 proposal 刷低 `G-mIoU` 的分母得分。
