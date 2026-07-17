# Progress Log: Implementing AMC Counter (Focal+Soft) for FlashVTG-GMR

## 2026-07-17
* **Initialized** task plan and progress log.
* **Discovered** that `torch.load` failed on PyTorch 2.6+ due to `weights_only=True` default. Fixed by adding `weights_only=False` in `training/flash_vtg_gmr/inference.py`.
* **Started** baseline (A0) inference script to run evaluation.
