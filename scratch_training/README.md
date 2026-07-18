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
* **`EPOCHS`**: Number of training epochs (default: `30` to give the random parameters sufficient time to learn temporal grounding features).
* **`BSZ`**: Training batch size (default: `128`).

Example:
```bash
DEVICE=0 EPOCHS=40 BSZ=128 bash scratch_training/run_scratch_training.sh
```

---

## 📊 Verification Outputs

All checkpoints, logs, and evaluation metrics will be saved in:

* **Logs & Checkpoints**: `results/scratch_training/hl-video_tef-amc_ft_scratch-<TIMESTAMP>/`
* **Gated Metrics**: `results/scratch_training/verification_results/gated_metrics.json`
* **Ungated Metrics**: `results/scratch_training/verification_results/ungated_metrics.json`
