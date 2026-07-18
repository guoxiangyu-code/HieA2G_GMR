#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-1}"

echo "1. Starting training of A6 model..."
DEVICE="${DEVICE}" ./scripts/train_a6_reproduce.sh

# Find the newly created results directory (it starts with hl-video_tef-amc_ft_focal_soft-)
LATEST_DIR=$(ls -td results/amc_ft/hl-video_tef-amc_ft_focal_soft-* | head -n 1)
BEST_CKPT="${LATEST_DIR}/model_best.ckpt"

echo "2. Training finished. Best checkpoint found at ${BEST_CKPT}"

# Create validation directory
VERIFY_DIR="results/verification_reproduced_a6"
mkdir -p "${VERIFY_DIR}"

echo "3. Running inference on test set using the newly trained checkpoint..."
python -m training.flash_vtg_gmr.inference \
  configs/flash_vtg_gmr/model.py \
  --resume "${BEST_CKPT}" \
  --opt_path configs/flash_vtg_gmr/soccer_gmr.json \
  --eval_split_name test \
  --eval_path data/label/Standard/test.jsonl \
  --eval_results_dir "${VERIFY_DIR}" \
  --v_feat_dirs /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/slowfast /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip \
  --t_feat_dir /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip_text \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --device "${DEVICE}" \
  --nms_thd 0.7

echo "4. Applying AMC count-gating..."
python scripts/apply_amc_gating_cli.py \
  --input_path "${VERIFY_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --output_path "${VERIFY_DIR}/gated_submission.jsonl"

echo "5. Evaluating gated predictions..."
python eval/eval_main.py \
  --submission_path "${VERIFY_DIR}/gated_submission.jsonl" \
  --gt_path data/label/Standard/test.jsonl \
  --save_path "${VERIFY_DIR}/gated_metrics.json" \
  --cls_thresholds 0.5 \
  --gmiou_cls_threshold 0.5

echo "--------------------------------------------------------"
echo "Verification complete. Gated metrics are saved in:"
echo "  ${VERIFY_DIR}/gated_metrics.json"
echo "--------------------------------------------------------"

echo "6. Restoring the main branch..."
git checkout main
echo "Pipeline execution finished successfully!"
