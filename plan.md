总览：FlashVTG → HieA2M 的修改点地图
FlashVTG 原有架构
├─ Encoder: Video(2D+3D) + Text(RoBERTa) + Cross-Modal Encoder
├─ Decoder: 多尺度窗口评分头 (window_score_head)
├─ Exist Gate: exist_head (sigmoid → 阈值0.5)         ← 删除
├─ Loss: saliency + offset + guide + loss_exist       ← loss_exist替换
└─ Inference: apply_existence_gate(score, exist, 0.5)  ← 替换

HieA2M 新增/修改
├─ [删除] exist_head
├─ [新增] count_head + count_proj + temporal_refine   → AMC模块
├─ [新增] masked_word_recovery                        → HTMA Level-1
├─ [新增] phrase_moment_align                          → HTMA Level-2
├─ [新增] global_align                                → HTMA Level-3
├─ [修改] SetCriterion: loss_exist → loss_count + loss_htma
├─ [修改] Dataset: 新增 count_label + 离线掩码预处理
├─ [修改] evaluate: apply_existence_gate → adaptive_moment_select
└─ [新增] memory_bank (训练时维护)
一、AMC（自适应时刻计数器）
设计思想
FlashVTG 的 exist_head 是 先定位再二分类有无——它看的是全局池化特征，不看窗口分数分布。这导致两个问题：

Gate Damage：全局特征不够判别时，sigmoid 分数落在 0.5 附近，阈值一刀切，误杀 152 个有正确定位的正样本
无计数能力：模型只能回答"有没有"，不能回答"有几个"，导致多时刻召回率极低（mR+@5=19.10）
AMC 的核心改变：把"有无"问题变成"有几个"问题，并且输入不仅看全局特征，还看窗口分数的分布形状。

伪代码
// ============ 模型定义 ============

// 替换 exist_head
count_head = MLP(
    input = concat(text_global, video_global, score_histogram)  // 多了直方图
    hidden = hidden_dim * 3 → hidden_dim → 5  // {0,1,2,3,3+}
    norm  = LayerNorm + GELU + Dropout
)

count_proj = MLP(
    input = concat(text_global, video_global, score_histogram)
    hidden = hidden_dim * 3 → 128  // 对比嵌入, 降维
)

temporal_refine = MLP(
    input = concat(count_feat, temporal_dist_matrix.flatten())
    hidden = hidden_dim * 2 → hidden_dim
    // 时序距离矩阵 = (1 - IoU) between all proposal pairs
    // 作用: 重叠窗口多的 → 可能是同一时刻被重复检测 → 计数应偏低
)
// ============ 前向传播 ============

function forward(video_feat, text_feat, proposals):
    // FlashVTG原有流程 (不改动)
    fused = cross_modal_encoder(video_feat, text_feat)
    window_scores = window_score_head(fused, proposals)
    
    // === AMC新增 ===
    // 1. 构造窗口分数直方图 (10个bin, 0.0-1.0)
    hist = histogram(window_scores, bins=10)  // (B, 10)
    
    // 2. 全局特征
    global_feat = concat(mean_pool(text_feat), mean_pool(fused))
    
    // 3. AMC输入 = 全局特征 + 直方图
    amc_input = concat(global_feat, hist_proj(hist))
    
    // 4. 时序距离矩阵
    dist = temporal_distance_matrix(proposals)  // (B, N, N)
    
    // 5. 计数预测
    count_feat = temporal_refine(amc_input, dist)
    count_logits = count_head(count_feat)       // (B, 5)
    count_embed = count_proj(count_feat)        // (B, 128) 供对比损失用
    
    return {window_scores, count_logits, count_embed}
// ============ 损失函数 ============

function loss_count(outputs, targets, memory_bank):
    logits = outputs.count_logits       // (B, 5)
    labels = targets.count_label        // (B,)
    embed = outputs.count_embed         // (B, 128)
    
    // 损失1: Focal Loss (解决类别失衡, 51.3%是count=0)
    // alpha按类别频率反比: {0:0.5, 1:0.7, 2:0.9, 3:0.95, 3+:0.97}
    loss_focal = FocalLoss(logits, labels, alpha, gamma=2.0)
    
    // 损失2: Memory Bank对比 (同类拉近, 异类推远)
    // 从memory_bank取同类的正样本和异类的负样本
    pos, neg = memory_bank.get_pos_neg(embed, labels)
    loss_contrast = SupConLoss(embed, labels, pos, neg, temperature=0.07)
    
    // 损失3: 软标签 (count=2时, 预测count=1比预测count=0的loss小)
    soft = make_soft_label(labels)  // 相邻类别各给0.2分
    loss_soft = CrossEntropy(logits, soft) * 0.3
    
    // 更新memory bank (no_grad)
    memory_bank.enqueue(embed.detach(), labels)
    
    return {loss_focal, loss_contrast, loss_soft}
推理逻辑
// ============ 推理 (替换 apply_existence_gate) ============

function adaptive_moment_select(window_scores, count_logits, proposals):
    count = argmax(count_logits)  // 0,1,2,3,4(=3+)
    
    if count == 0:
        return ∅  // 空集
    
    // 先NMS去重
    selected = NMS(proposals, window_scores, threshold=0.5)
    scores = window_scores[selected]
    
    if count <= 3:
        // 取top-k
        return top_k(selected, k=count)
    else:
        // 3+: 自适应阈值 = top-10分数的均值-标准差
        // 不用固定0.7, 因为不同查询的分数尺度不同
        top10 = scores[:min(10, len(scores))]
        thresh = top10.mean() - top10.std()
        above = selected[scores > thresh]
        // 保底: 至少3个, 至多10个
        return above[3:min(len(above), 10)]
为什么用 mean - std 而非固定阈值： "法国进球"的窗口分数可能集中在 0.8-0.95，"球员跑动"的可能集中在 0.3-0.5。固定 0.7 在后者全杀。mean - std 自适应到每个查询的分数分布。

二、HTMA（层次化时序-文本对齐）
设计思想
HieA2G 的 HMSA 有三级对齐（词/短语/全局），消融证明三级互补，去掉任一级都降。原方案只实现了词级（掩码恢复），短语级和全局级缺失。

三级对齐在时序域的对应关系：

HieA2G (空间域)	HieA2M (时序域)	对齐什么
词-对象	词-帧：掩码"射门"，用帧特征恢复	迫使帧编码器学习"射门长什么样"
短语-对象	子事件-片段：将"上半场+法国+进球"分别对齐到不同片段	迫使模型区分不同子事件对应的时段
文本-图像	查询-视频全局：InfoNCE拉近匹配对	直接服务空集判断
Level 1：词-帧对齐（Masked Word Recovery）
// ============ 离线预处理 (训练前做一次) ============

function preprocess_text_masking(dataset):
    // 足球动作词表 (离线构建)
    ACTION_WORDS = {saves, scores, passes, dribbles, blocks, 
                    tackles, shoots, crosses, clears, fouls, ...}
    
    for each sample in dataset:
        tokens = tokenize(sample.query)
        
        // 优先掩码动作词, mask_prob=0.15
        mask_positions = [i for i, t in enumerate(tokens) 
                         if t in ACTION_WORDS and random() < 0.15]
        
        // 回退: 无动作词时掩码非停用词
        if mask_positions is empty:
            candidates = [i for i, t in enumerate(tokens) 
                         if t not in STOPWORDS and len(t) > 3]
            mask_positions = [random_choice(candidates)]
        
        // 保存: 掩码后的token序列 + 原始token ID
        sample.masked_tokens = replace(tokens, mask_positions, [MASK])
        sample.masked_word_ids = [tokens[i] for i in mask_positions]
    
    save(dataset)  // 序列化到磁盘, 训练时直接加载
// ============ 模型: 词-帧恢复 ============

masked_word_recovery:
    cross_attn = MultiHeadAttention(hidden_dim, 8)
    recovery_head = Linear(hidden_dim, vocab_size)
    temporal_pos = LearnableParameter(1, num_clips, hidden_dim)

function forward_mask_recovery(masked_text_feat, video_feat, mask_positions):
    // 加时序位置编码 (HieA2G不需要, 但时序域必须有)
    video_feat += temporal_pos
    
    // 用被掩码的文本token去query视频帧
    recovered = cross_attn(
        query = masked_text_feat,   // 被掩码的文本
        key = video_feat,            // 帧级特征
        value = video_feat
    )
    
    // 只恢复被掩码位置的词
    masked_tokens = recovered[mask_positions]
    word_logits = recovery_head(masked_tokens)  // (num_masks, vocab_size)
    
    return word_logits

// 损失: 标准交叉熵
loss_mask_rec = CrossEntropy(word_logits, masked_word_ids)
设计要点： 加 temporal_pos 是因为"射门"发生在第 30 分钟和第 80 分钟的帧特征不同，必须有时序位置感知。HieA2G 在图像域不需要这个，因为框没有时序顺序。

Level 2：短语-片段对齐
// ============ 离线预处理: 短语分割 ============

function preprocess_phrase_splitting(dataset):
    // 用spaCy依存分析, 离线做一次
    for each sample in dataset:
        doc = spacy(sample.query)
        
        // 提取名词短语和动词短语
        // e.g. "The goalkeeper saves a shot in the first half"
        //      → ["the goalkeeper", "saves a shot", "in the first half"]
        phrases = [chunk.text for chunk in doc.noun_chunks] + 
                  [chunk.text for chunk in doc.verb_chunks]
        
        sample.phrase_spans = [(p.start, p.end) for p in phrases]
    
    save(dataset)
// ============ 模型: 短语-片段对齐 ============

phrase_moment_align:
    phrase_proj = Linear(hidden_dim, 128)
    moment_proj = Linear(hidden_dim, 128)
    temperature = LearnableParameter(0.07)

function forward_phrase_moment(phrase_feats, moment_feats, match_labels):
    // phrase_feats: 从文本编码器提取的各短语特征 (B, M, D)
    // moment_feats: 各候选时刻的视频特征 (B, K, D)
    // match_labels: 短语-时刻匹配GT (B, M, K)
    
    p = normalize(phrase_proj(phrase_feats))   // (B, M, 128)
    m = normalize(moment_proj(moment_feats))   // (B, K, 128)
    
    sim = einsum('bmd,bkd->bmk', p, m) / temperature  // (B, M, K)
    
    // 双向对比: 短语→时刻 + 时刻→短语
    loss_p2m = -(match_labels * log_softmax(sim, dim=2)).sum(dim=2).mean()
    loss_m2p = -(match_labels.T * log_softmax(sim, dim=1)).sum(dim=1).mean()
    
    return (loss_p2m + loss_m2p) / 2
设计要点： match_labels 的构造方式——对每个子短语，找到与之最匹配的 GT 时刻。例如"上半场"匹配 0-45 分钟的片段，"进球"匹配球进网的时刻片段。这需要基于 GT 时刻的语义类型做匹配，可以在数据预处理阶段完成。

Level 3：全局对齐
// ============ 模型: 全局查询-视频对齐 ============

global_align:
    text_proj = Linear(hidden_dim, 128)
    video_proj = Linear(hidden_dim, 128)
    temperature = LearnableParameter(0.07)

function forward_global_align(text_global, video_global):
    t = normalize(text_proj(text_global))   // (B, 128)
    v = normalize(video_proj(video_global))  // (B, 128)
    
    sim = mm(t, v.T) / temperature  // (B, B)
    
    // InfoNCE: 对角线为正样本
    labels = arange(B)
    loss = (CrossEntropy(sim, labels) + CrossEntropy(sim.T, labels)) / 2
    
    return loss
设计要点： 这一级直接服务空集判断——空集查询与视频的全局语义不匹配，InfoNCE 会推远它们的嵌入。但权重设最低（0.2），因为 batch 小时 InfoNCE 信号弱。

三、完整前向流程
function HieA2M_forward(video, text, proposals):
    // ===== FlashVTG 原有流程 (不改动) =====
    video_feat = video_encoder(video)          // 2D+3D特征
    text_feat = text_encoder(text)              // RoBERTa
    fused = cross_modal_encoder(video_feat, text_feat)
    window_scores = window_score_head(fused, proposals)
    
    // ===== HTMA Level-1: 词-帧恢复 =====
    masked_text = text_feat.clone()
    masked_text[mask_positions] = MASK_EMBED
    word_logits = masked_word_recovery(masked_text, video_feat, mask_positions)
    
    // ===== HTMA Level-2: 短语-片段对齐 =====
    phrase_feats = extract_phrases(text_feat, phrase_spans)
    moment_feats = pool_moments(fused, proposals)  // 各候选时刻的特征
    loss_phrase = phrase_moment_align(phrase_feats, moment_feats, match_labels)
    
    // ===== HTMA Level-3: 全局对齐 =====
    text_global = mean_pool(text_feat)
    video_global = mean_pool(fused)
    loss_global = global_align(text_global, video_global)
    
    // ===== AMC: 计数 =====
    hist = histogram(window_scores, bins=10)
    amc_input = concat(text_global, video_global, hist)
    dist = temporal_distance_matrix(proposals)
    count_feat = temporal_refine(amc_input, dist)
    count_logits = count_head(count_feat)
    count_embed = count_proj(count_feat)
    
    return {
        window_scores,       // FlashVTG原有
        word_logits,         // HTMA L1
        count_logits,        // AMC
        count_embed,         // AMC对比
    }
四、完整损失
function HieA2M_loss(outputs, targets, memory_bank):
    losses = {}
    
    // FlashVTG原有 (不改动)
    losses["saliency"] = loss_saliency(outputs, targets) * 1.0
    losses["offset"]   = loss_offset(outputs, targets) * 1.0
    losses["guide"]    = loss_guide(outputs, targets) * 1.0
    
    // AMC (替代原 loss_exist)
    count = loss_count(outputs, targets, memory_bank)
    losses["count_focal"]    = count.focal * 1.0
    losses["count_contrast"] = count.contrast * 0.5
    losses["count_soft"]     = count.soft * 0.3
    
    // HTMA
    losses["mask_rec"] = loss_mask_rec(outputs, targets) * 0.5   // L1
    losses["phrase"]   = loss_phrase(outputs, targets) * 0.3     // L2
    losses["global"]   = loss_global(outputs, targets) * 0.2     // L3
    
    return losses
权重设计理由：

AMC 总权重 1.8（1.0+0.5+0.3），与 FlashVTG 三项原有损失（各 1.0）量级相当
HTMA 三级总权重 1.0（0.5+0.3+0.2），作为辅助对齐信号而非主任务
全局对齐权重最低：batch 小时 InfoNCE 噪声大，不宜给高权重
五、训练策略
// Phase 1: 冻结encoder, 训练AMC+HTMA (5 epochs)
// 目的: 新模块先学基础能力, 不破坏预训练特征
encoder.requires_grad = False
train(count_head, count_proj, temporal_refine,
      masked_word_recovery, phrase_moment_align, global_align)
lr = 1e-4

// Phase 2: 解冻encoder, 联合微调 (10 epochs)
encoder.requires_grad = True
lr_encoder = 1e-5   // 小lr, 避免破坏预训练
lr_new = 1e-4       // 新模块保持

// Memory Bank: 每个iteration结束后更新
after each forward:
    memory_bank.enqueue(count_embed.detach(), count_labels)
六、数据管线修改
// 离线预处理 (训练前做一次, 保存到磁盘)

function preprocess(dataset):
    for each sample:
        // 1. 计数标签
        num_moments = len(sample.relevant_windows)
        sample.count_label = min(num_moments, 4)   // {0,1,2,3,3+}
        sample.count_exact = num_moments            // 真实数量, 供软标签
        
        // 2. 软标签
        sample.count_soft = make_soft_label(num_moments)
        // count=2 → [0, 0.2, 0.6, 0.2, 0]
        
        // 3. 文本掩码 (用动作词表, 不用NLTK)
        sample.masked_tokens = mask_action_words(sample.query)
        sample.masked_word_ids = get_masked_ids(sample.query)
        
        // 4. 短语分割 (用spaCy依存分析)
        sample.phrase_spans = spacy_split(sample.query)
        
    save(dataset)
七、评测修改
// 替换 apply_existence_gate, 新增诊断指标

function evaluate(predictions, ground_truth):
    metrics = {}
    
    // 原有指标 (保持)
    metrics["mAP"]      = compute_map(...)
    metrics["mR@1"]     = compute_mr_at_k(1)
    metrics["mR+@5"]    = compute_mr_plus_at_k(5)
    metrics["G-mIoU@1"] = compute_gmiou(1)
    
    // 新增: 空集/有目标分离 (来自HieA2G)
    metrics["N-acc"] = null_queries_correctly_rejected / total_null_queries
    metrics["T-acc"] = pos_queries_not_falsely_rejected / total_pos_queries
    
    // 新增: 计数准确率
    metrics["Count-acc"] = count_predicted_correctly / total
    
    // 新增: 按计数分层
    for c in {0,1,2,3,3+}:
        metrics[f"mAP_count{c}"] = compute_map_on_subset(count==c)
    
    return metrics
改进点总结
修改点	原FlashVTG	HieA2M	设计依据
存在性判断	sigmoid二分类+硬阈值	AMC五类计数+自适应选窗口	HieA2G消融：计数法比阈值法好13.9%
AMC输入	仅全局池化特征	全局特征+窗口分数直方图+时序距离矩阵	时序域特有：必须看窗口分布才知道"几个"
对齐方式	单层级（全局）	三级（词/短语/全局）	HieA2G消融：三级互补，缺任一级都降
文本掩码	无	动作词优先掩码+帧特征恢复	时序特有：不同帧对应不同动作，需时序位置编码
推理3+	不适用	自适应阈值mean-std	不回退固定阈值，保持连续性
类别失衡	不处理	Focal Loss+软标签	51.3%是空集，普通CE会全预测0
训练策略	直接训练	两阶段：先冻结后联合	避免新模块破坏预训练特征
评测	AUROC+Rej-F1	+N-acc+T-acc+Count-acc+分层	精准诊断是误报还是漏报
已执行计划4.撰写训练/推理流程的伪代码与设计思想
方案已完整给出。核心就是 三个改动 + 一个新增：

改 exist_head → count_head（二分类→五类计数）
改 推理逻辑（硬阈值→自适应选窗口）
改 数据管线（加计数标签+离线掩码）
新增 HTMA 三级对齐（词-帧/短语-片段/全局）
如果方案方向没问题，下一步建议先做消融验证：只加 AMC（不加 HTMA），跑一个 epoch 看看 Count-acc 和 N-acc 是否比原始 sigmoid 有提升。确认有效后再加 HTMA。