# HieA2M: Scratch Training Suite

This directory contains the script to train the entire HieA2M network (including the transformer backbone, Adaptive Moment Counter, and Hierarchical Temporal Moment Alignment modules) starting from completely **random parameters (from scratch/from 0)**.

Unlike the main reproduction pipeline, this process does not load any pre-trained model checkpoint.

---

## 📂 Included Scripts

* **`run_scratch_training.sh`**: Trains the model with random parameters, runs inference on the test split, applies AMC gating, and evaluates the final performance.

---

## 🚀 Running the Script

You can start the end-to-end scratch training by running:

```bash
# From the project root directory
DEVICE=1 bash scratch_training/run_scratch_training.sh
```

### ⚙️ Configurable Environment Variables

You can customize the run behavior by specifying environment variables before running:

* **`DEVICE`**: GPU index (default: `1`).
* **`EPOCHS`**: Number of training epochs (default: `400` for thorough convergence from scratch).
* **`BSZ`**: Training batch size (default: `256`).

Example (training on GPU 0 with custom settings):
```bash
DEVICE=0 EPOCHS=400 BSZ=256 bash scratch_training/run_scratch_training.sh
```

> [!WARNING]
> Since the backbone network is fully unfrozen and trained from scratch, a batch size of `256` might consume significant GPU memory. If you experience an Out of Memory (OOM) error on your 24GB GPUs (like the RTX 3090), please reduce the batch size by specifying `BSZ=128` or `BSZ=64` before running.

---

## 📊 Verification Outputs

All checkpoints, logs, and evaluation metrics will be saved in:

* **Logs & Checkpoints**: `results/scratch_training/hl-video_tef-amc_ft_scratch-<TIMESTAMP>/`
* **Gated Metrics**: `results/scratch_training/verification_results/gated_metrics.json`
* **Ungated Metrics**: `results/scratch_training/verification_results/ungated_metrics.json`
