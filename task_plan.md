# Task Plan: Implementing AMC Counter (Focal+Soft) for FlashVTG-GMR

## Goal
Implement the AMC (Adaptive Moment Counter) with Focal Loss and Soft Label Loss, replacing the baseline existence sigmoid gating. Evaluate its impact on Count-acc and N-acc. Do not include HTMA, Contrastive count loss, or Ordinal regression loss for this phase (A1 experiment).

## Phases

| Phase | Description | Status | Details / Notes |
|---|---|---|---|
| Phase 1 | Baseline evaluation (A0) | Complete | Run inference with original checkpoint and log metrics. |
| Phase 2 | Dataset & Preprocessing (AMC labels) | Complete | Update `dataset.py` to produce hard (`count_label`) and soft (`count_soft`) count labels. |
| Phase 3 | Model Architecture & Loss (AMC implementation) | Complete | Replace `exist_head` with AMC counter. Implement Focal + Soft Loss for count classification. |
| Phase 4 | Inference Logic (Adaptive selection) | Complete | Update evaluation inference to use count prediction for gating. |
| Phase 5 | Training & Evaluation (A1) | Pending | Train the new model with AMC and run evaluation to compare with A0 baseline. |

## Errors Encountered
*None so far.*
