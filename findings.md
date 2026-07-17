# Findings: Implementing AMC Counter (Focal+Soft) for FlashVTG-GMR

## Baseline Metrics (A0 - sigmoid existence gate)
Baseline evaluation run on the provided `flashVTG_gmr` checkpoint using `data/label/Standard/test.jsonl` yields:

* **Threshold 0.4**
  - **N-acc**: 59.76% (294 / 492)
  - **T-acc**: 69.30% (377 / 544)
  - **Count-acc**: 29.25% (303 / 1036)

* **Threshold 0.6**
  - **N-acc**: 89.63% (441 / 492)
  - **T-acc**: 49.45% (269 / 544)
  - **Count-acc**: 43.15% (447 / 1036)

* **Standard GMR Metrics**
  - **AUROC**: 74.00%
  - **Rej-F1@0.4**: 61.72%
  - **Rej-F1@0.6**: 73.06%
  - **G-mIoU@1**: 39.60%
  - **mAP**: 25.65%
  - **mR+@5**: 21.30%

## Tasks and Notes
* Environment set up and PyTorch 2.6+ unpickling issue fixed via `weights_only=False` in `setup_model`.
* NumPy 2.0+ incompatibility with `np.trapz` fixed by falling back to `np.trapezoid` in `eval/metrics.py`.
