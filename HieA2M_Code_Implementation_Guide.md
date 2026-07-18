# HieA2M Code Implementation Guide

This guide provides step-by-step instructions from an engineering perspective to transform the baseline `generalized-moment-retrieval.git` repository into the current HieA2M version (Commit: `bd283b2`).

## 1. Overall Code Structure Changes

The transformation from the baseline to HieA2M primarily involves modifying the data pipeline to support new labels and features, replacing the naive binary existence head with the Adaptive Moment Counter (AMC) and Hierarchical Text-Moment Alignment (HTMA) modules, and updating the training/inference logic to handle these new structures.

**Key Modified Files:**
*   `training/flash_vtg_gmr/dataset.py`: Added AMC count labels, online tokenization/masking, and phrase boundary extraction.
*   `models/flash_vtg_gmr/model.py`: The core architectural changes. Replaced `exist_head` with `AdaptiveMomentCounter` and `HTMA`. Implemented new loss functions in `SetCriterion`.
*   `training/flash_vtg_gmr/inference.py`: Added parameter freezing for 2-phase training, differential learning rates, and updated the existence gating mechanism.
*   `training/flash_vtg_gmr/config.py`: Registered new hyperparameters and arguments.
*   `scripts/train_amc_only.sh`: New training script for Phase 1.

---

## 2. Data Pipeline Modifications (`dataset.py`)

To support AMC and HTMA, the data loader must provide count labels, masked text inputs, and phrase spans.

### 2.1 AMC Count Labels
Instead of a simple binary `exist_label`, we need fine-grained counting.
*   **Hard Label (`count_label`)**: Capped at 4 (e.g., `{0, 1, 2, 3, 4+}`).
    ```python
    num_moments = len(sample["relevant_windows"])
    sample["count_label"] = min(num_moments, 4)
    ```
*   **Soft Label (`count_soft`)**: To handle ambiguity in temporal boundaries (e.g., is a 10s clip 1 or 2 events?), apply a Gaussian-like smoothing: 0.6 to the target class, 0.2 to adjacent classes.
    ```python
    idx = min(num_moments, 4)
    soft = torch.full((5,), 0.0)
    soft[idx] = 0.6
    if idx > 0: soft[idx - 1] = 0.2
    if idx < 4: soft[idx + 1] = 0.2
    sample["count_soft"] = soft
    ```

### 2.2 HTMA Level-1 Tokenization & Masking
To force the model to recover action semantics, implement online tokenization and masking.
*   Use `AutoTokenizer` from `openai/clip-vit-base-patch32`.
*   Implement probabilistic masking (e.g., 15%) prioritizing action words ("saves", "scores", "passes"). If no action words exist, fallback to random non-stopwords.
*   Add `tokens`, `mask_positions`, and `masked_word_ids` to the batched data.

### 2.3 HTMA Level-2 Phrase Extraction
Instead of treating the query as a single sequence, break it down.
*   Use a heuristic (like splitting at "players from") or spaCy noun/verb chunks to extract `phrase_spans` (start/end token indices for sub-phrases) and pass them to the model.

---

## 3. Model Architecture Modifications (`model.py`)

This is the most critical part of the implementation.

### 3.1 Remove the Old `exist_head`
In `FlashVTG.__init__`, delete the old binary classification head:
```python
# REMOVE THIS:
# self.exist_head = nn.Sequential(
#     nn.Linear(hidden_dim * 2, hidden_dim),
#     nn.ReLU(inplace=True),
#     nn.Linear(hidden_dim, 1),
# )
```

### 3.2 Add the Adaptive Moment Counter (AMC)
Create the `AdaptiveMomentCounter` class. It must ingest global text/video features, a histogram of window scores, and a temporal distance matrix.
*   **Temporal Distance Matrix**: Compute $1 - \text{tIoU}$ for all pairs of proposals $(B, N, N)$.
*   **Feature Projections**:
    *   `hist_proj`: Linear layer mapping a 10-bin histogram to $D$.
    *   `dist_proj`: Resize the $N \times N$ matrix to a fixed $150 \times 150$ via bilinear interpolation, flatten, and pass through an MLP to $D$.
*   **Fusion & Classification**: Concatenate text_global, video_global, hist_feat, and dist_feat $\rightarrow (B, 4D)$. Pass through a refinement MLP and a final classification head yielding $(B, 5)$ logits.

### 3.3 Add the HTMA Module
Create the `HTMA` class with three levels:
*   **Level 1 (Word-Frame Recovery)**: `nn.MultiheadAttention(D, 8)` followed by a linear projection to the vocabulary size (`49408`). Requires `sinusoidal_pe` for temporal positioning.
*   **Level 2 (Phrase-Moment Alignment)**: Linear projections (`phrase_proj`, `moment_proj`) to 128-dim.
*   **Level 3 (Global Alignment)**: Linear projections (`text_global_proj`, `video_global_proj`) to 128-dim.

Create `SinkhornMatcher` implementing Sinkhorn iterations (`compute_sinkhorn`) to replace bipartite matching for Level 2.

### 3.4 Update `FlashVTG.forward`
Integrate the new modules:
1.  **Decode Proposals**: Convert normalized `out_coord` to absolute time bounds.
2.  **AMC Forward**: Pass `src_txt`, `video_emb`, `window_scores`, and decoded proposals to AMC to get `pred_exist_logits`.
3.  **HTMA Level 1**: Substitute masked tokens with a learnable `txt_mask_embed`. Perform cross-attention between masked text and video (with PE) to output `word_logits`.
4.  **HTMA Level 2**: Construct `phrase_feats` (mean-pooling over `phrase_spans`) and `moment_feats` (mean-pooling video features over decoded proposal bounds + PE). Compute cosine similarity matrix `phrase_moment_sim`.
5.  **HTMA Level 3**: Mean-pool text and video to get `global_t` and `global_v`.

---

## 4. Loss Function Implementation (`model.py` -> `SetCriterion`)

Replace the old binary cross-entropy `loss_exist` with four new robust loss functions.

### 4.1 `loss_exist` (AMC Loss)
This replaces the binary BCE and handles the 5-class prediction.
*   **Focal Loss**: To address class imbalance, apply weights $\alpha = [0.5, 0.7, 0.9, 0.95, 0.97]$ to standard cross-entropy.
*   **Soft Label Loss**: Cross-entropy between `count_soft` and predicted log-probabilities.
*   **Ordinal Regression Loss**: Applied *only* to positive samples (label > 0). Calculate expected count and penalize deviation from the true count using MSE, plus an absolute distance penalty.
*   **Null Anchor Loss**: For null-set samples (label == 0), enforce a margin: $P(\text{class} \ge 2) - P(\text{class} == 0) + 0.3 \le 0$.

Total AMC Loss = Focal(1.0) + Soft(0.3) + Ordinal(0.5) + NullAnchor(1.0).

### 4.2 `loss_mask_rec` (HTMA Level 1)
*   **Positive samples**: Standard cross-entropy for predicting the masked word ID.
*   **Null-set samples**: Margin-based loss. Push the logit of the *correct* word below the maximum of other logits by a margin of 0.5. (We want the model to fail recovery if the event doesn't exist).

### 4.3 `loss_phrase_moment` (HTMA Level 2)
*   Compute optimal transport plan $T$ via `compute_sinkhorn(1 - sim)`.
*   Loss = $-\sum (T \cdot \log(\text{sim}))$.

### 4.4 `loss_global_align` (HTMA Level 3)
*   Standard symmetric InfoNCE loss between `global_t` and `global_v` using a learnable `logit_scale`.

### 4.5 Registering Losses
In `build_model1`, update `weight_dict` and `losses` list:
```python
weight_dict["loss_exist"] = 1.0
weight_dict["loss_mask_rec"] = 0.5
weight_dict["loss_phrase"] = 0.3
weight_dict["loss_global"] = 0.2
losses = list(losses) + ["exist", "mask_rec", "phrase", "global"]
```

---

## 5. Training and Inference Logic (`inference.py`)

### 5.1 Parameter Freezing & Differential LRs
To support Phase 1 training (AMC/HTMA only):
*   In `setup_model`, if `--train_amc_only` is set, iterate through `model.named_parameters()`. Set `requires_grad = True` *only* for parameters containing `amc_counter`, `htma`, `txt_mask_embed`, or `logit_scale`.
*   In optimizer setup, apply a lower learning rate (`opt.lr * 0.1`) to the backbone and standard `opt.lr` to the new parameters.

### 5.2 Inference Gating Update
Update the thresholding logic to use the 5-class output:
*   Instead of `sigmoid(pred_exist_logits)`, calculate the probability of existence as the sum of probabilities for classes 1 through 4:
    ```python
    exist_probs = F.softmax(output["pred_exist_logits"], dim=-1)
    exist_score = exist_probs[:, 1:].sum(dim=-1)
    output["_out"]["pred_exist_score"] = exist_score
    ```

### 5.3 Config Arguments (`config.py`)
Add necessary CLI arguments:
*   `--train_amc_only`: Flag for Phase 1.
*   `--exist_gate_thd`: Soft gate threshold.
*   `--pred_topk_for_cls`: For evaluation metric tweaks.

---

## 6. Execution

Create `scripts/train_amc_only.sh` to launch the Phase 1 training. It should load the pretrained `flashVTG_gmr` checkpoint, set `--train_amc_only`, and disable saliency losses (`--lw_saliency 0`, etc.) while AMC and HTMA train. Phase 2 (Consistency) drops the freeze flag and jointly trains the network.
