# 得分漂移根因分析 (Score Drift Analysis)

经过对训练代码、模型结构、损失函数、数据集处理逻辑以及官方示例提交的得分分布进行全面排查，导致复现模型 `pred_exist_score` 全部 > 0.4 的原因是**多个因素叠加**造成的，核心可以归纳为以下几点：

## 🔴 根因 1：验证集排除了负样本 → 最优 checkpoint 选择对 exist_head 失明

这是**最关键的原因**。在 `training/moment_detr_gmr/train.py` 中：

```python
# train.py line 172
# 验证集构建时 keep_empty_gt = False
# 即：所有 relevant_windows 为空的负样本被过滤掉了
```

这意味着：
- 训练时的 **best checkpoint 选择依据是 `MR-full-mAP`**（纯定位指标）。
- 验证集里**完全没有负样本**，exist_head 在负样本上的表现从未被评估。
- 模型完全可以把 exist_head 的 logit bias 学到偏正值（即倾向于预测"存在"），而不会在验证集上被惩罚。
- 不同的随机种子/训练轮数导致 exist_head 的 bias 偏移幅度不同，这就是**不可精确复现**的直接原因。

## 🔴 根因 2：BCE 损失无类别平衡权重

在 `models/moment_detr_gmr/gmr_adapter.py` 的 `compute_existence_loss()` 中：

```python
F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
# 没有传入 pos_weight 参数
```

训练集正负样本比例约为 `2722:2463 ≈ 1.1:1`，虽然不算极端不平衡，但在 exist_loss 权重本身就较小的情况下，缺少 `pos_weight` 会让模型学到**略偏正的先验**。

## 🟡 根因 3：Exist 损失权重被主任务淹没

```yaml
exist_loss_coef: 1.0    # exist head 的损失权重
span_loss_coef:  10     # 时序定位的损失权重
label_loss_coef: 4      # 前景/背景分类的损失权重
```

exist_head 的梯度信号被时序定位任务**稀释了约 15 倍**。这导致：
- exist_head 的学习速度远慢于主干。
- 容易受到随机初始化和优化噪声的影响，导致不同训练 run 之间得分分布差异巨大。

## 🟡 根因 4：官方得分分布本身就在"悬崖边"

从官方示例提交的分数分布来看：

| 分数区间 | 样本类型 | 说明 |
| :---: | :---: | :--- |
| **0.99 ~ 1.00** | 正样本 (SportsMotion) | 明显的正例，分数很高 |
| **0.30 ~ 0.46** | 正/负混合 (WorldCup) | 大量样本**密集聚集**在阈值 0.4 两侧 |
| **< 0.30** | 极少数负样本 | 非常少 |

官方模型在默认阈值 0.4 时也只有 `Rej-F1 = 64.01`，说明存在性判别本身就很**边缘**——大量正/负样本的分数都挤在 0.30~0.46 这个狭窄区间里。**只要 logit 偏移 0.5~1.0 个单位，sigmoid 后就足以让所有分数从 0.35 跳到 0.50 以上**，整个分布就会"漂"过阈值线。

## 🟢 次要因素

| 因素 | 说明 |
| :--- | :--- |
| **Exist head 仅在最后一层 decoder 监督** | 中间层的 aux loss 不包含 exist loss，监督信号稀薄。 |
| **Exist head 作用于 decoder query 表征** | 这些表征主要为定位优化，未必携带干净的"存在/不存在"信息。 |
| **软门控（soft gating）不够强硬** | 即使 exist_score < 阈值，仍以 score 本身作为乘子（而非直接归零），导致负样本仍可能输出非空预测。 |

## 🎯 总结与建议

> **核心结论**：不是训练过程"出了错"，而是官方的训练流程本身存在设计缺陷——验证集不评估 exist_head、损失权重过小、无类别平衡。这导致 exist_head 的得分校准（calibration）高度依赖训练的随机性（种子、学习率调度、early stopping 时机）。官方结果恰好在一个"刚好能用"的分布上，而复现模型由于训练随机性，logit 整体偏移了约 +0.5~1.0，导致 sigmoid 后全部 > 0.4。

如果要更稳定地复现和改进模型，可以考虑以下优化方向：
1. **完善验证集逻辑**：在验证集中保留负样本，用包含 `Rej-F1` 或 `G-mIoU` 的综合指标来挑选最佳 checkpoint。
2. **类别平衡**：给 BCE loss 加上 `pos_weight` 做类别平衡。
3. **调整损失权重**：提高 `exist_loss_coef`（如从 1.0 提高到 4.0）。
4. **后处理阈值搜索**：最简单的后处理方案是在验证集上进行阈值搜索（例如复现结果中最佳的 0.55 阈值），而不是写死 0.4。









  ### 1. 存在性得分 s(q) 与阈值 τ 的定义（第 5 页，3.3 节）                                               

  论文首先定义了每个模型都会输出一个“存在性得分” s(q)（对于他们提出的方法，就是预测的概率 pᵉˣⁱˢᵗ）。      
  接着在 3.3.1 Null-Set Rejection 这一节明确给出了阈值 τ 的裁决规则：                                     
                                                                                                          
  │ “At operating threshold τ, a query q is classified as null-set if s(q) ≤ τ.”                          
  │ (在工作阈值 τ 下，如果查询 q 的存在得分 s(q) ≤ τ，则将其分类为 null-set，即空集/事件不存在。)         
                                                                                                          
  ### 2. 在整体评估指标中的“门控”作用（第 6 页，3.3.3 节）                                                

  在介绍综合指标  G-mIoU@k  时，论文解释了阈值如何直接干预最终输出：                                      
                                                                                                          
  │ “Using the same operating threshold τ, the model’s top-k predictions 𝒫̂ₖ(q) are gated to emptyset if   
  │ s(q) ≤ τ.”                                                                                            
  │ (使用相同的工作阈值 τ，如果 s(q) ≤ τ，模型给出的前 k 个预测框将被全部阻断并强行输出为空集 emptyset。) 
                                                                                                          
  ### 3. 推理阶段的数学公式（第 7 页，4.1.3 节 Inference）                                                

  论文给出了一个非常清晰的数学分段函数（公式 11）：                                                       
                                                                                                          
        ⎧    emptyset,      pᵉˣⁱˢᵗ < τ                                                                    
        ⎪   (i)  (i)                                                                                      
    𝒯̂ = ⎨ {t̂   ,t̂   }ᴺᵢ₌₁,  pᵉˣⁱˢᵗ ≥ τ                                                                    
        ⎩   s    e                                                                                        

  这明确表示，在推理时，只有当存在性概率 pᵉˣⁱˢᵗ ≥ τ                                                       
  时，模型才会把找出来的时间段（{tₛ,tₑ}）作为最终答案；一旦 <τ，直接输出空集。                            
                                                                                                          
  (注：论文中的描述是硬门控 hard-                                                                         
  gating，即直接丢弃框；而代码实现中为了保持梯度或平滑，默认用的是把框的置信度乘上这个极低分数的软门控    
  soft-gating，但最终达到的过滤效果是一致的。)

  ### 4. 0.4 这个具体数值是怎么来的？（第 7 页，5.1 节）

  在Implementation Details（实验细节）中，作者解释了为何选用 0.4 作为默认值：

  │ “we... select the inference threshold τ = 0.4 based on validation performance.”
  │ (我们……基于验证集上的表现，选择了推理阈值 τ = 0.4。)

  ### 5. 附录的鲁棒性分析（第 13 页，附录 D）

  论文在附录专门讨论了阈值的敏感性（Metric Threshold Sensitivity）。作者测试了 τ ∈ {0.4,0.6,0.8}          
  三个不同的阈值，并在表 8 中汇报了结果。作者的结论是：

  │ “The relative ranking among models remains consistent across all tested thresholds, demonstrating that
  │ the benchmark conclusions in the main paper are robust to the choice of τ.”
  │ (各个模型之间的相对排名在所有测试的阈值下保持一致，证明主论文的结论对阈值 τ 的选择是鲁棒的。)         
  ──────
  总结：在论文中，阈值 τ 被严格定义为过滤空集查询（Null-set queries）的硬性数学开关。0.4
  这个数字并没有绝对的物理意义，只是作者在官方数据集的验证集（Validation
  split）上调参搜索出来的一个最优超参数。这就是为什么你在本地复现时，由于分数分布因为随机性整体平移了，0.4
  这个固定的超参数就不再适用的原因。
