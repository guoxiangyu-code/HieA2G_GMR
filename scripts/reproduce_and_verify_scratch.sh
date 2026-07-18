#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-joint}" # 'pure' or 'joint'
DEVICE="${DEVICE:-1}"

if [ "${MODE}" = "pure" ]; then
  echo "1. Starting PURE scratch training (random initialization)..."
  DEVICE="${DEVICE}" ./scripts/train_from_scratch_pure.sh
  EXP_DIR="results/scratch_pure"
  VERIFY_DIR="results/verification_scratch_pure"
elif [ "${MODE}" = "joint" ]; then
  echo "1. Starting JOINT baseline training (unfrozen from epoch 0)..."
  DEVICE="${DEVICE}" ./scripts/train_fully_joint_baseline.sh
  EXP_DIR="results/scratch_joint_baseline"
  VERIFY_DIR="results/verification_scratch_joint"
else
  echo "Invalid mode. Use 'pure' or 'joint'"
  exit 1
fi

# Find the newly created results directory
LATEST_DIR=$(ls -td ${EXP_DIR}/hl-video_tef-* | head -n 1)
BEST_CKPT="${LATEST_DIR}/model_best.ckpt"

echo "2. Training finished. Best checkpoint found at ${BEST_CKPT}"

# Create verification directory
mkdir -p "${VERIFY_DIR}"

echo "3. Running inference on test set..."
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

echo "6. Evaluating ungated raw temporal localization..."
python eval/eval_main.py \
  --submission_path "${VERIFY_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --gt_path data/label/Standard/test.jsonl \
  --save_path "${VERIFY_DIR}/ungated_metrics.json" \
  --cls_thresholds 0.5 \
  --gmiou_cls_threshold 0.5

echo "--------------------------------------------------------"
echo "Verification complete. Metrics are saved in ${VERIFY_DIR}"
echo "--------------------------------------------------------"
