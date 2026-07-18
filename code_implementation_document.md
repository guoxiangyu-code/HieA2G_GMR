# HieA2M: Detailed Code Implementation and Modification Document

This document traces the code-level differences between the original GMR repository (`dymm9977/generalized-moment-retrieval`) and our optimized **HieA2M (A6 & Phase 2)** codebase. It outlines the exact changes across all critical files to guide future developers and maintainers.

---

## 1. File Modification Overview

The integration of **HieA2M (AMC + HTMA)** required modifications to 5 core files:

```
generalized-moment-retrieval/
├── models/flash_vtg_gmr/model.py          <-- AMC losses, HTMA L1/L2/L3 architectures
├── training/flash_vtg_gmr/dataset.py      <-- Online text masking & label generation
├── training/flash_vtg_gmr/inference.py    <-- Gating logic, parameter freezing & optimizer
├── training/flash_vtg_gmr/config.py       <-- Added configuration arguments
└── eval/metrics.py                        <-- Fixed NumPy 2.0+ compatibility
```

---

## 2. Core File Modifications

### 2.1 models/flash_vtg_gmr/model.py (Losses & Network Architecture)

This is the main algorithmic engine of HieA2M. We implemented:
1. **AMC Loss (`loss_exist` in `SetCriterion`)**:
   - Swapped out the paper's binary classifier for a 5-class counter ($C \in \{0, 1, 2, 3, 4\}$).
   - Added Focal Loss + Soft Label smoothing.
   - Added positive-only Ordinal Regression.
   - Added Null Anchor Loss constraint for empty sets:
     ```python
     loss_null_anchor = torch.max(
         torch.tensor(0.0, device=logits.device),
         probs[:, 2:].sum(dim=-1) - probs[:, 0] + 0.3
     ).mean()
     ```
2. **HTMA Level-1 (Word-Frame Recovery)**:
   - Restores action word embeddings from video proposal features using cross-attention.
   - For negative samples, applies a margin constraint to push representation spaces apart:
     ```python
     # Margin pushing for null-set samples
     loss = torch.max(torch.tensor(0.0, device=logits.device), 0.5 - loss_ce)
     ```
3. **HTMA Level-2 (Phrase-Moment Alignment)**:
   - Uses vectorized **Sinkhorn iterations** to solve for the optimal soft matching matrix between phrase embeddings and temporal moment embeddings.
4. **HTMA Level-3 (Global Figure-Text InfoNCE)**:
   - Implements contrastive projection heads to align global video embeddings and query text representations.
5. **Count-Window Consistency Loss (`loss_count_window_consistency` in `SetCriterion`)**:
   - Aligns AMC expectation outputs with window head scores:
     ```python
     loss = F.mse_loss(expected_count, top3_mean * 4.0)
     ```

---

### 2.2 training/flash_vtg_gmr/dataset.py (Preprocessing & Online Masking)

Added data pipelines for HTMA and AMC labeling in `StartEndDataset`:
1. **Count Label Preprocessing**:
   - Calculates the hard count target (`count_label` = min(num_moments, 4)).
   - Builds soft count distribution (`count_soft`):
     - Target class $c$: `0.6` weight.
     - Neighboring classes $c \pm 1$: `0.2` weight.
2. **HTMA Online Preprocessing**:
   - Dynamically tokenizes and pads textual query strings using `AutoTokenizer` from `openai/clip-vit-base-patch32`.
   - Action words (e.g., *saves, scores, passes, dribbles*) are masked with a 15% probability. If none are selected, a random token is masked.
   - Regex-based noun/verb chunking extracts phrase boundary indices (`phrase_spans`) inside the token space.

---

### 2.3 training/flash_vtg_gmr/inference.py (Optimization & Gating)

Added support for detached training, differential learning rates, and relative peak gating:
1. **GMR Gate Score calculation for 5-Class AMC**:
   - Converts multi-class count logits to existence probability:
     ```python
     pred_exist_scores = F.softmax(logits, dim=-1)[:, 1:].sum(dim=-1)
     ```
2. **Differential Parameter Groups (`setup_model` in `inference.py`)**:
   - If `train_amc_only=True` (Phase 1), freezes all backbone layers and unfreezes only `amc_counter` and `htma`.
   - If `train_amc_only=False` (Phase 2), registers two optimizer parameter groups:
     - New modules (`amc_counter`, `htma`): learning rate `1e-4`.
     - Frozen/pretrained backbone: learning rate scaled down to `1e-5` (differential factor `0.1`).
3. **Robust Pickling (`weights_only=False`)**:
   - Avoids serialization crashes on newer PyTorch environments by disabling strict weights checks during state dict loading.

---

### 2.4 training/flash_vtg_gmr/config.py (Configuration Definitions)

Added command-line parser arguments:
* `--use_exist_head`: Boolean flag to activate AMC existence gating.
* `--exist_loss_coef`: Main scalar weight for AMC losses.
* `--null_anchor_coef`: Dynamic coefficient parser for empty-set anchor loss (defaults to `0.3`).

---

### 2.5 eval/metrics.py (NumPy 2.0 Compatibility)

To prevent runtime environment failures on modern python installations:
* Swapped `np.trapz()` (deprecated in NumPy 2.0+) with `np.trapezoid()` fallback to ensure seamless area-under-the-curve metrics calculation.
