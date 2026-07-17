# HieA2M 完整落地方案

> 基于 HieA2G (AAAI 2025) 的七处理论不对齐分析，针对 GMR 任务的连续、模糊时序特性定制的时序版本。

---

## 修改接口总览

```
FlashVTG 代码库
├─ pipeline/
│  ├─ dataset.py              ← 第一步: 数据预处理
│  ├─ preprocess_masks.py     ← 第一步: 离线文本掩码
│  ├─ preprocess_phrases.py   ← 第一步: 离线短语分割
│  └─ preprocess_sim.py       ← 第一步: 离线查询相似度
├─ models/flash_vtg_gmr/
│  ├─ model.py                ← 第二步+第三步: 架构+前向+损失
│  └─ matcher.py              ← 第三步: Sinkhorn匹配器
├─ inference/
│  └─ eval_gmr.py             ← 第四步: 推理逻辑
└─ configs/
   └─ gmr_config.yaml         ← 第五步: 训练配置
```

---

## 第一步：数据管线重构

### 设计思想

FlashVTG 原始数据管线只输出 `src_txt`、`video_feat`、`relevant_windows`。HieA2M 需要额外三类标签：计数标签、文本掩码、短语分割。**全部在离线预处理阶段完成**，避免训练时重复计算。

### 1.1 AMC 计数标签

```python
# pipeline/dataset.py — __getitem__ 中新增

num_moments = len(sample["relevant_windows"])

# 硬标签: {0, 1, 2, 3, 3+} → 索引 {0, 1, 2, 3, 4}
sample["count_label"] = min(num_moments, 4)

# 软标签: 相邻类别各给0.2, 当前类别0.6
# 应对时序切分歧义 (10秒片段算1个还是2个?)
idx = min(num_moments, 4)
soft = torch.full((5,), 0.0)
soft[idx] = 0.6
if idx > 0: soft[idx - 1] = 0.2
if idx < 4: soft[idx + 1] = 0.2
sample["count_soft"] = soft
```

**数学形式：**

$$\tilde{y}_c = \begin{cases} 0.6 & c = \min(n, 4) \\ 0.2 & c = \min(n, 4) \pm 1 \\ 0 & \text{otherwise} \end{cases}$$

### 1.2 HTMA Level-1 文本掩码

```python
# pipeline/preprocess_masks.py — 离线运行一次

ACTION_WORDS = {"saves", "scores", "passes", "dribbles", "blocks",
                "tackles", "shoots", "crosses", "clears", "fouls",
                "header", "volley", "tackle", "intercept"}

def mask_query(tokens, mask_prob=0.15):
    """优先掩码动作词, 回退到非停用词"""
    mask_positions = [i for i, t in enumerate(tokens)
                      if t.lower() in ACTION_WORDS and random() < mask_prob]

    if not mask_positions:
        candidates = [i for i, t in enumerate(tokens)
                      if t.lower() not in STOPWORDS and len(t) > 3]
        if candidates:
            mask_positions = [random.choice(candidates)]

    masked = tokens.clone()
    original_ids = []
    for pos in mask_positions:
        original_ids.append(tokens[pos].item())
        masked[pos] = MASK_TOKEN_ID

    return masked, mask_positions, original_ids
```

### 1.3 HTMA Level-2 短语分割

```python
# pipeline/preprocess_phrases.py — 离线运行一次

import spacy
nlp = spacy.load("en_core_web_sm")

def extract_phrases(query):
    doc = nlp(query)
    phrases = []

    # 名词短语: "the goalkeeper", "a shot", "the first half"
    for chunk in doc.noun_chunks:
        phrases.append({
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end,
            "type": "noun"
        })

    # 动词短语: "saves a shot", "scores a goal"
    for token in doc:
        if token.pos_ == "VERB":
            subtree = list(token.subtree)
            phrases.append({
                "text": " ".join(t.text for t in subtree),
                "start": subtree[0].i,
                "end": subtree[-1].i + 1,
                "type": "verb"
            })

    return phrases
```

### 1.4 查询语义相似度矩阵（供对比损失用）

```python
# pipeline/preprocess_sim.py — 离线运行一次

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# 对所有查询编码, 计算两两余弦相似度
embeddings = model.encode(all_queries)
sim_matrix = cosine_similarity(embeddings)  # (N, N)

# 保存: 训练时用于对比损失的正样本过滤
save(sim_matrix, "query_sim_matrix.pkl")
```

**设计依据：** 不对齐 #5——同计数 ≠ 语义相似。对比损失的正样本需同时满足计数相同且查询语义相似度 > δ。

---

## 第二步：模型架构改造

### 设计思想

FlashVTG 的 `exist_head`（MLP → sigmoid → 1维）被完全移除。替换为两个模块：**AMC**（计数器）和 **HTMA**（三级对齐头）。关键改造点是 AMC 的输入特征必须包含窗口分布信号（不对齐 #7）。

### 2.1 删除旧模块

```python
# models/flash_vtg_gmr/model.py — __init__ 中

# 删除:
# self.exist_head = nn.Sequential(
#     nn.Linear(hidden_dim, hidden_dim),
#     nn.ReLU(),
#     nn.Linear(hidden_dim, 1)
# )
```

### 2.2 新增 AMC 模块

```python
class AdaptiveMomentCounter(nn.Module):
    def __init__(self, hidden_dim, num_proposals):
        super().__init__()

        D = hidden_dim

        # 窗口分数直方图投影 (不对齐#7: 窗口分数不编码"有几个")
        self.hist_proj = nn.Linear(10, D)

        # 时序距离矩阵投影 (不对齐#7: 重叠窗口→可能重复检测)
        self.dist_proj = nn.Sequential(
            nn.Linear(num_proposals * num_proposals, D * 2),
            nn.LayerNorm(D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D)
        )

        # 计数分类头
        self.count_head = nn.Sequential(
            nn.Linear(D * 3, D),   # text_global + video_global + hist
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D, 5)        # {0, 1, 2, 3, 3+}
        )

        # 对比嵌入头 (降维到128, 训练更稳)
        self.count_proj = nn.Sequential(
            nn.Linear(D * 3, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, 128)
        )

        # 时序精炼: 融合距离矩阵信息
        self.temporal_refine = nn.Sequential(
            nn.Linear(D * 2, D),   # amc_feat + dist_feat
            nn.LayerNorm(D),
            nn.GELU()
        )
```

**数学形式：**

$$M_g = [\text{AP}(T_w);\; \text{AP}(\mathbf{F}_v);\; \phi(\mathbf{s}_{\text{win}})] \in \mathbb{R}^{3D}$$

$$\mathbf{d}_{\text{temp}} = \psi(\text{flatten}(\mathbf{D}_{\text{temp}})) \in \mathbb{R}^{D}$$

$$\mathbf{h}_{\text{count}} = \text{Refine}([M_g;\; \mathbf{d}_{\text{temp}}]) \in \mathbb{R}^{D}$$

$$\hat{y}_c = \text{CountHead}(\mathbf{h}_{\text{count}}) \in \mathbb{R}^5, \quad \mathbf{z} = \text{CountProj}(\mathbf{h}_{\text{count}}) \in \mathbb{R}^{128}$$

**时序距离矩阵定义：**

$$\mathbf{D}_{\text{temp}}[i,j] = 1 - \text{tIoU}(w_i, w_j) = 1 - \frac{\min(e_i, e_j) - \max(s_i, s_j)}{(e_i - s_i) + (e_j - s_j) - [\min(e_i, e_j) - \max(s_i, s_j)]}$$

### 2.3 新增 HTMA 模块

```python
class HTMA(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_clips=10):
        super().__init__()
        D = hidden_dim

        # === Level 1: 词-帧恢复 ===
        # 不对齐#3: FlashVTG没有object embedding, 需要构造moment embedding
        # 不对齐#1: null-set不能跳过, 要推远
        self.cross_attn_w2f = nn.MultiheadAttention(D, 8, batch_first=True)
        self.word_recovery_head = nn.Linear(D, vocab_size)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_clips, D))

        # === Level 2: 短语-片段对齐 ===
        # 不对齐#2: 用最优传输替代二分匹配
        self.phrase_proj = nn.Linear(D, 128)
        self.moment_proj = nn.Linear(D, 128)

        # === Level 3: 全局对齐 ===
        self.text_global_proj = nn.Linear(D, 128)
        self.video_global_proj = nn.Linear(D, 128)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))
```

**Level 1 的 moment embedding 构造（不对齐 #3 的关键）：**

$$\mathbf{m}_j = \frac{1}{e_j - s_j}\int_{s_j}^{e_j} \mathbf{F}_v(t)\, dt + \text{PE}(s_j, e_j)$$

```python
def construct_moment_embedding(self, video_feat, proposals):
    """
    将(start, end)区间内的帧特征池化为moment embedding
    proposals: (B, N, 2) — 每个窗口的(start, end)帧索引
    """
    B, N, _ = proposals.shape
    T = video_feat.size(1)

    moment_feats = []
    for b in range(B):
        feats = []
        for n in range(N):
            s, e = proposals[b, n]
            s, e = int(s.clamp(0, T-1)), int(e.clamp(0, T))
            # 区间内帧特征均值池化
            pooled = video_feat[b, s:e].mean(dim=0)  # (D,)
            # 时序位置编码: 正弦/余弦编码 (start, end)
            pe = sinusoidal_pe(s, e, D=video_feat.size(-1))
            feats.append(pooled + pe)
        moment_feats.append(torch.stack(feats))
    return torch.stack(moment_feats)  # (B, N, D)
```

---

## 第三步：前向传播与损失计算

### 3.1 前向传播

```python
# models/flash_vtg_gmr/model.py — forward()

def forward(self, src_vid, src_txt, proposals, src_txt_masked=None,
            mask_positions=None, phrase_spans=None):
    # ===== FlashVTG 原有流程 (不改动) =====
    vid_feat = self.video_encoder(src_vid)           # (B, T, D)
    txt_feat = self.text_encoder(src_txt)             # (B, L, D)
    fused = self.cross_modal_encoder(vid_feat, txt_feat)
    window_scores = self.window_head(fused, proposals)  # (B, N)

    # ===== AMC 前向 =====
    # 1. 窗口分数直方图
    hist = torch.histc(window_scores, bins=10, min=0, max=1)  # (10,)
    hist = hist.unsqueeze(0).expand(B, -1)                      # (B, 10)
    hist_feat = self.amc.hist_proj(hist)                        # (B, D)

    # 2. 全局特征
    text_global = txt_feat.mean(dim=1)        # (B, D)
    video_global = fused.mean(dim=1)           # (B, D)

    # 3. 时序距离矩阵
    dist = temporal_distance_matrix(proposals)  # (B, N, N), 元素=1-tIoU
    dist_flat = dist.reshape(B, -1)              # (B, N*N)
    dist_feat = self.amc.dist_proj(dist_flat)    # (B, D)

    # 4. 计数预测
    amc_input = torch.cat([text_global, video_global, hist_feat], dim=-1)  # (B, 3D)
    refined = self.amc.temporal_refine(
        torch.cat([amc_input, dist_feat], dim=-1)  # (B, 4D) → Refine → (B, D)
    )
    count_logits = self.amc.count_head(refined)           # (B, 5)
    count_embed = self.amc.count_proj(refined)             # (B, 128)

    # ===== HTMA 前向 =====
    # Level 1: 词-帧恢复
    if src_txt_masked is not None:
        vid_with_pos = vid_feat + self.htma.temporal_pos[:, :vid_feat.size(1), :]
        recovered, _ = self.htma.cross_attn_w2f(
            query=src_txt_masked, key=vid_with_pos, value=vid_with_pos
        )
        word_logits = self.htma.word_recovery_head(
            recovered[mask_positions]  # 只恢复掩码位置
        )
    else:
        word_logits = None

    # Level 2: 短语-片段对齐 (构造moment embedding)
    moment_feats = self.htma.construct_moment_embedding(fused, proposals)
    phrase_feats = self.extract_phrase_features(txt_feat, phrase_spans)
    # 相似度矩阵
    p = F.normalize(self.htma.phrase_proj(phrase_feats), dim=-1)  # (B, M, 128)
    m = F.normalize(self.htma.moment_proj(moment_feats), dim=-1)  # (B, N, 128)
    sim_matrix = torch.einsum('bmd,bnd->bmn', p, m)  # (B, M, N)

    # Level 3: 全局对齐
    t = F.normalize(self.htma.text_global_proj(text_global), dim=-1)  # (B, 128)
    v = F.normalize(self.htma.video_global_proj(video_global), dim=-1)  # (B, 128)

    return {
        "window_scores": window_scores,
        "count_logits": count_logits,
        "count_embed": count_embed,
        "word_logits": word_logits,
        "sim_matrix": sim_matrix,
        "global_t": t,
        "global_v": v,
    }
```

### 3.2 损失计算

```python
# models/flash_vtg_gmr/model.py — SetCriterion

class HieA2MCriterion:
    def forward(self, outputs, targets, query_sim_matrix=None,
                memory_bank=None):
        losses = {}

        # ===== FlashVTG 原有损失 (不改动) =====
        losses["saliency"] = self.loss_saliency(outputs, targets) * 1.0
        losses["offset"] = self.loss_offset(outputs, targets) * 1.0
        losses["guide"] = self.loss_guide(outputs, targets) * 1.0

        # ===== AMC 损失 =====
        losses.update(self.loss_count(outputs, targets, query_sim_matrix,
                                       memory_bank))

        # ===== HTMA 损失 =====
        losses["mask_rec"] = self.loss_masked_word(outputs, targets) * 0.5
        losses["phrase"] = self.loss_phrase_moment(outputs, targets) * 0.3
        losses["global"] = self.loss_global_align(outputs, targets) * 0.2

        return losses
```

### 3.3 AMC 损失详解

```python
def loss_count(self, outputs, targets, query_sim_matrix, memory_bank):
    logits = outputs["count_logits"]    # (B, 5)
    labels = targets["count_label"]     # (B,)
    soft = targets["count_soft"]        # (B, 5)
    embed = outputs["count_embed"]      # (B, 128)

    # --- 损失1: Focal Loss (解决类别失衡) ---
    # Soccer-GMR: 51.3% null, 30.4% single, 18.3% multi
    alpha = torch.tensor([0.5, 0.7, 0.9, 0.95, 0.97]).to(logits.device)
    loss_focal = focal_loss(logits, labels, alpha, gamma=2.0)

    # --- 损失2: 有序回归惩罚 (不对齐#4: 时序计数本质模糊) ---
    # 惩罚跳跃式错误: 预测count=0而真实count=2, 惩罚 > 预测count=1
    probs = F.softmax(logits, dim=-1)  # (B, 5)
    classes = torch.arange(5, device=logits.device).float()  # [0,1,2,3,4]
    expected = (probs * classes).sum(dim=-1)  # (B,) 期望计数
    loss_ord = ((expected - labels.float()) ** 2).mean()  # MSE
    # 额外的有序惩罚项
    penalty = torch.zeros(1, device=logits.device)
    for c_prime in range(5):
        penalty += (abs(c_prime - labels.float()) * probs[:, c_prime] ** 2).mean()
    loss_ord_penalty = penalty

    # --- 损失3: 软标签 (不对齐#4: 标注噪声) ---
    log_probs = F.log_softmax(logits, dim=-1)
    loss_soft = -(soft * log_probs).sum(dim=-1).mean()

    # --- 损失4: 监督对比 (不对齐#5: 同计数+语义相似才为正样本) ---
    loss_contrast = self.supervised_contrast(
        embed, labels, query_sim_matrix, memory_bank
    )

    return {
        "count_focal": loss_focal * 1.0,
        "count_ord": (loss_ord + loss_ord_penalty) * 0.5,
        "count_soft": loss_soft * 0.3,
        "count_contrast": loss_contrast * 0.5,
    }
```

**Focal Loss 数学形式：**

$$\mathcal{L}_{\text{focal}} = -\sum_{c=0}^{4} \alpha_c (1 - p_c)^\gamma y_c \log p_c$$

其中 $\alpha = [0.5, 0.7, 0.9, 0.95, 0.97]$ 按类别频率反比设置。

**有序回归数学形式：**

$$\mathcal{L}_{\text{ord}} = \underbrace{\left(\sum_{c=0}^{4} c \cdot p_c - y_c\right)^2}_{\text{期望值MSE}} + \underbrace{\sum_{c=0}^{4} |c - y_c| \cdot p_c^2}_{\text{跳跃惩罚}}$$

### 3.4 监督对比损失（不对齐 #5 的核心修正）

```python
def supervised_contrast(self, embed, labels, query_sim_matrix, memory_bank):
    """
    正样本条件: 计数相同 AND 查询语义相似度 > δ
    负样本: 其他所有样本 (不管语义是否相似)
    """
    B = embed.size(0)
    tau = 0.07
    delta = 0.5  # 语义相似度阈值

    # 从 memory bank 取额外正负样本
    pos_feats, neg_feats = [], []
    if memory_bank is not None:
        for i in range(B):
            # 正样本: 同计数 + 语义相似
            pos = memory_bank.get_positive(embed[i], labels[i],
                                            query_idx=i,
                                            query_sim=query_sim_matrix,
                                            delta=delta)
            pos_feats.append(pos)
            # 负样本: 不同计数 (不管语义)
            neg = memory_bank.get_negative(labels[i])
            neg_feats.append(neg)

    # In-batch + Memory Bank 合并
    all_feat = torch.cat([embed] + pos_feats + neg_feats, dim=0)
    all_labels = torch.cat([labels, ...])

    # 标准 SupConLoss
    sim = torch.mm(embed, all_feat.t()) / tau  # (B, M)
    # 分子: 只对正样本求和
    pos_mask = ...  # 同计数 AND 语义相似
    num = (sim * pos_mask).sum(dim=-1)
    # 分母: 对所有样本求和
    den = torch.logsumexp(sim, dim=-1)
    return (-num + den).mean()
```

**数学形式：**

$$\mathcal{L}_{\text{con}} = -\frac{1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

$$P(i) = \{p : y_c^p = y_c^i \;\land\; \text{sim}(\mathbf{T}_q^p, \mathbf{T}_q^i) > \delta\}$$

### 3.5 HTMA Level-1 损失（不对齐 #1 的核心修正）

```python
def loss_masked_word(self, outputs, targets):
    """
    不对齐#1: null-set样本不跳过, 而是推远

    正样本: 最小化恢复误差 (帧特征能重建文本→对齐)
    null-set: 最大化恢复误差 (帧特征不能重建文本→不对齐)
    """
    word_logits = outputs["word_logits"]   # (num_masks, V)
    mask_ids = targets["masked_word_ids"]  # (num_masks,)
    is_null = targets["is_null"]           # (B,) bool

    if word_logits is None:
        return torch.tensor(0.0)

    # 正样本: 标准交叉熵 (最小化恢复误差)
    loss_pos = F.cross_entropy(word_logits[~is_null], mask_ids[~is_null])

    # null-set样本: 最大化恢复误差
    # 实现: margin loss, 要求正确词的logit比次高logit低margin
    if is_null.any():
        null_logits = word_logits[is_null]
        null_ids = mask_ids[is_null]
        correct_logit = null_logits[range(len(null_ids)), null_ids]
        max_other = (null_logits - F.one_hot(null_ids, V) * 1e9).max(dim=-1).values
        loss_null = F.relu(correct_logit - max_other + 0.5).mean()  # margin=0.5

    return loss_pos + 0.3 * loss_null
```

**数学形式：**

$$\mathcal{L}_{w2f} = \underbrace{-\sum \log p(\text{correct word})}_{\text{正样本: 拉近}} + 0.3 \cdot \underbrace{\sum \max(0,\; l_{\text{correct}} - l_{\text{other}} + m)}_{\text{null-set: 推远 (margin loss)}}$$

### 3.6 HTMA Level-2 损失（不对齐 #2 的核心修正）

```python
# models/flash_vtg_gmr/matcher.py — 新增

class SinkhornMatcher:
    """最优传输匹配, 替代二分匹配"""

    def __init__(self, epsilon=0.1, n_iters=3):
        self.epsilon = epsilon    # 熵正则化系数
        self.n_iters = n_iters    # Sinkhorn迭代次数

    @torch.no_grad()
    def compute_transport(self, cost_matrix):
        """
        cost_matrix: (B, M, N) — 代价 = 1 - 相似度
        返回: transport plan T, (B, M, N)
        """
        K = torch.exp(-cost_matrix / self.epsilon)  # (B, M, N)

        # 初始化边际分布
        M, N = cost_matrix.shape[1], cost_matrix.shape[2]
        p = torch.full((M,), 1.0/M).to(cost_matrix.device)  # 短语均匀分布
        q = torch.full((N,), 1.0/N).to(cost_matrix.device)  # 时刻均匀分布

        # Sinkhorn 迭代
        u = torch.ones_like(p)
        v = torch.ones_like(q)
        for _ in range(self.n_iters):
            u = p / (K @ v + 1e-8)    # (B, M)
            v = q / (K.transpose(-1,-2) @ u + 1e-8)  # (B, N)

        T = torch.diag_embed(u) @ K @ torch.diag_embed(v)  # (B, M, N)
        return T
```

```python
def loss_phrase_moment(self, outputs, targets):
    """
    不对齐#2: 用最优传输替代二分匹配

    HieA2G: Y是硬匹配矩阵, BCE损失
    HieA2M: T是软传输方案, 传输损失
    """
    sim = outputs["sim_matrix"]  # (B, M, N) 短语-时刻相似度
    cost = 1 - sim               # 代价矩阵

    # Sinkhorn 求解最优传输
    T = self.sinkhorn_matcher.compute_transport(cost)  # (B, M, N)

    # 传输损失: 最大化匹配处的相似度
    # = -sum(T * log(sim))
    loss = -(T * torch.log(sim + 1e-8)).sum(dim=(-2, -1)).mean()

    return loss
```

**数学形式：**

$$T^* = \arg\min_T \langle T, C \rangle + \epsilon H(T), \quad \text{s.t.} \quad T\mathbf{1} = \mathbf{p},\; T^\top\mathbf{1} = \mathbf{q}$$

$$\mathcal{L}_{p2m} = -\sum_{i=1}^{M}\sum_{j=1}^{N} T^*_{i,j} \cdot \log \text{sim}(\hat{T}_p^i, \hat{M}_e^j)$$

### 3.7 HTMA Level-3 损失

```python
def loss_global_align(self, outputs, targets):
    """
    标准 InfoNCE, 拉近匹配的查询-视频对, 推远不匹配的
    """
    t = outputs["global_t"]  # (B, 128)
    v = outputs["global_v"]  # (B, 128)

    sim = torch.mm(t, v.t()) * F.softmax(self.logit_scale)  # (B, B)
    labels = torch.arange(B, device=sim.device)

    loss_t2v = F.cross_entropy(sim, labels)
    loss_v2t = F.cross_entropy(sim.t(), labels)

    return (loss_t2v + loss_v2t) / 2
```

**数学形式：**

$$\mathcal{L}_{t2v} = -\log \frac{\exp(S(V, Q)/\tau)}{\sum_{Q' \in \mathcal{B}} \exp(S(V, Q')/\tau)}$$

---

## 第四步：推理逻辑更新

### 设计思想

用 AMC 的计数预测替代 sigmoid 门控。核心创新是 **3+ 类别的自适应阈值**——不用固定 0.7（不对齐 #6），因为不同查询的窗口分数尺度差异巨大。

### 伪代码

```python
# inference/eval_gmr.py

def adaptive_moment_select(window_scores, count_logits, proposals,
                            nms_threshold=0.5):
    """
    替换 apply_existence_gate(window_scores, exist_scores, 0.5)

    不对齐#6: 不用固定阈值, 用相对峰值阈值
    """
    count = count_logits.argmax(dim=-1).item()

    # Count=0: 直接返回空集
    if count == 0:
        return []

    # NMS 去重 (时序域特有: 同一时刻的不同尺度窗口IoU>0.9)
    selected_idx = temporal_nms(proposals, window_scores, nms_threshold)
    selected_scores = window_scores[selected_idx]
    selected_proposals = proposals[selected_idx]

    if count <= 3:
        # Top-K: 取前count个高分窗口
        topk_idx = torch.argsort(selected_scores, descending=True)[:count]
        return selected_proposals[topk_idx]

    else:
        # Count=3+: 自适应阈值
        # 不用 mean-std (过于宽松), 用相对峰值阈值
        top1 = selected_scores[0]
        thresh = max(0.5, 0.8 * top1)  # 不对齐#6: 相对峰值, 非固定值

        above = selected_scores > thresh
        result = selected_proposals[above]

        # 保底: 至少3个, 至多10个
        if len(result) < 3:
            result = selected_proposals[:3]
        elif len(result) > 10:
            result = selected_proposals[:10]

        return result
```

**3+ 类别的阈值数学形式：**

$$\tau^* = \max(0.5,\; 0.8 \cdot s_{\text{top-1}})$$

$$\text{Output} = \{w_j : s_j > \tau^*\} \cup \text{保底}(\min 3, \max 10)$$

**为什么用 `0.8 * top1` 而非 `mean - std`：** T6 错误分析指出窗口分数存在弥散现象——某些查询会产生大量中低分窗口。`mean - std` 会被这些弥散窗口拉低均值，导致阈值过低。`0.8 * top1` 是**相对于最高分的比例**，不受弥散窗口影响。

### 评测指标更新

```python
def evaluate(predictions, ground_truth):
    metrics = {}

    # === 原有指标 ===
    metrics["mAP"] = compute_map(predictions, ground_truth)
    metrics["mR@1"] = compute_mr_at_k(predictions, ground_truth, k=1)
    metrics["mR+@5"] = compute_mr_plus_at_k(predictions, ground_truth, k=5)
    metrics["G-mIoU@1"] = compute_gmiou(predictions, ground_truth, k=1)

    # === 新增: HieA2G的空集/有目标分离指标 ===
    null_queries = [q for q in ground_truth if len(q["moments"]) == 0]
    pos_queries = [q for q in ground_truth if len(q["moments"]) > 0]

    # N-acc: 空集查询被正确拒绝的比例 (TP / (TP+FN))
    null_correct = sum(1 for q in null_queries
                       if len(predictions[q["id"]]) == 0)
    metrics["N-acc"] = null_correct / max(len(null_queries), 1)

    # T-acc: 有目标查询未被误判为空集的比例 (TN / (TN+FP))
    pos_correct = sum(1 for q in pos_queries
                      if len(predictions[q["id"]]) > 0)
    metrics["T-acc"] = pos_correct / max(len(pos_queries), 1)

    # === 新增: 计数准确率 ===
    count_correct = sum(1 for q in ground_truth
                        if min(len(q["moments"]), 4) ==
                        min(len(predictions[q["id"]]), 4))
    metrics["Count-acc"] = count_correct / len(ground_truth)

    # === 新增: 按计数分层分析 ===
    for c in range(5):
        subset = [q for q in ground_truth
                  if min(len(q["moments"]), 4) == c]
        if len(subset) > 0:
            metrics[f"mAP_c{c}"] = compute_map(
                {q["id"]: predictions[q["id"]] for q in subset}, subset
            )

    return metrics
```

---

## 第五步：训练策略与消融实验

### 5.1 两阶段训练

```
Phase 1 (5 epochs): 冻结 FlashVTG encoder, 只训练 AMC + HTMA
  目的: 新模块先学基础能力, 不破坏预训练特征
  encoder.requires_grad = False
  lr = 1e-4

Phase 2 (10 epochs): 解冻 encoder, 联合微调
  encoder.requires_grad = True
  lr_encoder = 1e-5  (小lr)
  lr_new = 1e-4      (新模块保持)
```

### 5.2 Memory Bank 更新

```python
# 训练循环中, 每个 iteration 后
@torch.no_grad()
def update_memory_bank(model, batch, memory_bank):
    count_embed = model.get_count_embed(batch)
    memory_bank.enqueue(count_embed, batch["count_label"])
    memory_bank.remove_oldest_if_full()
```

### 5.3 完整损失权重配置

```python
LOSS_WEIGHTS = {
    # FlashVTG 原有损失 (保持原配置)
    "saliency": 1.0,
    "offset": 1.0,
    "guide": 1.0,

    # AMC 损失
    "count_focal": 1.0,        # 主损失
    "count_ord": 0.5,          # 有序回归
    "count_soft": 0.3,         # 软标签
    "count_contrast": 0.5,     # 对比损失

    # HTMA 损失
    "mask_rec": 0.5,           # 词级恢复 (Level 1)
    "phrase": 0.3,             # 短语级对齐 (Level 2)
    "global": 0.2,             # 全局对齐 (Level 3)
}
```

**权重设计理由：**
- AMC 总权重 2.3（1.0+0.5+0.3+0.5），与 FlashVTG 三项原有损失（各 1.0）量级相当
- HTMA 三级总权重 1.0（0.5+0.3+0.2），作为辅助对齐信号而非主任务
- 全局对齐权重最低（0.2）：batch 小时 InfoNCE 噪声大，不宜给高权重

### 5.4 消融实验设计

| 实验 | 配置 | 验证目标 |
|------|------|---------|
| **A0** | FlashVTG 原始 (sigmoid 门控) | baseline |
| **A1** | + AMC only (Focal+Soft, 无对比, 无有序) | 验证计数器本身有效 |
| **A2** | A1 + 有序回归 | 验证有序惩罚对模糊计数的帮助 |
| **A3** | A2 + Memory Bank 对比 (语义条件化) | 验证对比损失 (不对齐#5) |
| **A4** | A3 + HTMA L1 (推远式掩码恢复) | 验证 null-set 推远 (不对齐#1) |
| **A5** | A4 + HTMA L2 (Sinkhorn) | 验证最优传输 (不对齐#2) |
| **A6** | A5 + HTMA L3 (全局 InfoNCE) | 完整模型 |
| **A7** | A6, 但 L1 用 α=0 (跳过 null-set) | 验证"推远 > 跳过" |
| **A8** | A6, 但 L2 用二分匹配 | 验证"OT > Bipartite" |
| **A9** | A6, 但对比损失只用同计数 (不加语义) | 验证语义条件化 (不对齐#5) |
| **A10** | A6, 但 3+ 用固定 0.7 阈值 | 验证自适应阈值 (不对齐#6) |
| **A11** | A6, 但 AMC 输入不加直方图+距离矩阵 | 验证窗口分布信号 (不对齐#7) |

### 5.5 关键诊断指标

| 指标 | 含义 | 诊断作用 |
|------|------|---------|
| **N-acc** | 空集正确拒绝率 | AMC 是否解决了误报 |
| **T-acc** | 有目标未被误杀率 | AMC 是否引入了新的漏报 |
| **Count-acc** | 计数预测准确率 | AMC 是否真正学会了计数 |
| **mAP_c0** | 空集子集的 mAP | 空集处理能力 |
| **mAP_c2+** | 多时刻子集的 mAP | 多时刻召回能力 |
| **mR+@5** | 多时刻增量召回 | 核心瓶颈指标 |

---

## 七处不对齐与落地对应关系

| 不对齐 # | 问题 | 对应落地步骤 | 核心修正 |
|---------|------|------------|---------|
| #1 | α=0 跳过 null-set | 第三步·HTMA L1 损失 | null-set 用 margin loss **推远**而非跳过 |
| #2 | 二分匹配→硬匹配 | 第三步·HTMA L2 损失 | **Sinkhorn 最优传输**替代二分匹配 |
| #3 | 无 moment embedding | 第二步·HTMA 模块 | 帧池化 + **时序位置编码**构造 moment embedding |
| #4 | 硬 CE + 类别失衡 | 第三步·AMC 损失 | **Focal + 有序回归 + 软标签** |
| #5 | 同计数≠语义相似 | 第三步·AMC 对比损失 | 正样本条件化为**计数 AND 语义相似度 > δ** |
| #6 | 3+ 回退固定阈值 | 第四步·推理逻辑 | **相对峰值阈值** `max(0.5, 0.8*top1)` |
| #7 | 窗口分数不编码计数 | 第二步·AMC 输入 | 加**分数直方图 + 时序距离矩阵** |

---

## 实施优先级建议

```
第一周: 第一步(数据管线) + 第二步(模型架构)
        → 跑通前向传播, 验证输出 shape 正确

第二周: 第三步(损失计算) — 先只接 AMC, 不接 HTMA
        → 跑 A0 vs A1, 验证计数器本身有效
        → 重点看 Count-acc 和 N-acc

第三周: 第三步(HTMA L1) + A4/A7 消融
        → 验证"推远 > 跳过"
        → 重点看 N-acc 是否提升

第四周: 第三步(HTMA L2+L3) + 第四步(推理)
        → 跑完整 A6, 与 A0 对比
        → 重点看 mR+@5 和 G-mIoU@1

第五周: 第五步(消融实验 A7-A11)
        → 逐个验证七处修正的贡献
        → 准备论文实验表格
```

**建议起点：** 从 A0→A1 开始——先只加 AMC 计数器（Focal+Soft），不加 HTMA，验证计数器本身能否提升 Count-acc 和 N-acc。这是风险最低、信号最清晰的第一步。
