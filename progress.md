# Progress Log: HieA2M & GMR Implementation

## 2026-07-17
* **Initialized** task plan and progress log.
* **Fixed** PyTorch unpickling compatibility issues with `weights_only=False` in `inference.py`.
* **Fixed** NumPy 2.0+ incompatibility with `np.trapezoid` fallback in `eval/metrics.py`.
* **Analyzed** A1 baseline metrics, revealing Possibility A (heavy prediction count=1 bias causing multi-moment flat mIoU).
* **Developed** A2 (AMC Balanced) positive-only ordinal regression to activate multi-moments.
* **Developed** A2-anchor empty-set constraint to recover null-set protection.
* **Implemented** full HTMA Levels 1/2/3 (Word-Frame Recovery, Phrase-Moment Sinkhorn transport, and Global InfoNCE alignment) in `models/flash_vtg_gmr/model.py`.
* **Implemented** online text tokenization, masking, and phrase segmenting boundary extraction in `training/flash_vtg_gmr/dataset.py`.
* **Trained** A6 Phase 1 (Backbone frozen, `null_anchor = 1.0`) model for 10 epochs. Achieved overall gated G-mIoU@3 of **53.45%** (highest overall performance, +19.92% over paper baseline).
* **Diagnosed** Phase 1 results, confirming that backbone freezing limits HTMA's feature-pushing effect.

## 2026-07-18
* **Launched** Phase 2 training with backbone unfrozen and `null_anchor_coef = 0.3` on GPU 1 with batch size 256.
* **Analyzed** Phase 2 results, confirming positive recall improvement but showing empty-set leakage due to lowered constraint.
* **Designed & Implemented** Count-Window Consistency Loss (weight 0.2) in `SetCriterion` to align classification expected counts with top-3 window score peaks.
* **Trained** Phase 2 Consistency model on GPU 1. Achieved gated G-mIoU@3 of **49.74%**, empty-set protection of **84.96%**, and the highest overall un-gated mAP of **25.14%**.
* **Pushed** all optimized model and configuration files to GitHub `main` branch.
