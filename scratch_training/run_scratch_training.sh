#!/usr/bin/env bash
# ==============================================================================
# HieA2M: Train from Scratch (Random Parameters, No Checkpoints)
# ==============================================================================
set -euo pipefail

# Resolve directories
SCRATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRATCH_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-1}"
EPOCHS="${EPOCHS:-400}"
BSZ="${BSZ:-256}"

# Adjust learning rate decay drop step proportionally to the number of epochs
# For 400 epochs, decay learning rate at epoch 150 and 300.
LR_DROP=$((EPOCHS * 37 / 100))

echo "======================================================================"
# 1. Start Training from Scratch (Random Initialization)
# Note:
#   - We omit --resume to initialize all parameters (backbone + AMC + HTMA) randomly.
#   - We do NOT specify --train_amc_only, which allows the entire network to be trained.
# ==============================================================================
echo "1. Starting training from scratch (device GPU ${DEVICE}, epochs ${EPOCHS}, bsz ${BSZ}, lr_drop ${LR_DROP})..."
python -m training.flash_vtg_gmr.train \
  configs/flash_vtg_gmr/model.py \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/label/Standard/train.jsonl \
  --eval_path data/label/Standard/val.jsonl \
  --eval_split_name val \
  --v_feat_dirs /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/slowfast /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip \
  --t_feat_dir /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/feature/standard/clip_text \
  --v_feat_dim 2816 \
  --t_feat_dim 512 \
  --max_q_l 40 \
  --max_v_l 75 \
  --clip_length 2 \
  --max_windows 5 \
  --lr 1e-4 \
  --lr_drop "${LR_DROP}" \
  --wd 1e-4 \
  --n_epoch "${EPOCHS}" \
  --max_es_cnt 30 \
  --bsz "${BSZ}" \
  --eval_bsz 1 \
  --eval_epoch 2 \
  --device "${DEVICE}" \
  --results_root results/scratch_training \
  --exp_id amc_ft_scratch \
  --seed 2024 \
  --hidden_dim 256 \
  --dim_feedforward 1024 \
  --enc_layers 3 \
  --t2v_layers 6 \
  --dummy_layers 2 \
  --nheads 8 \
  --num_dummies 40 \
  --total_prompts 10 \
  --num_prompts 1 \
  --kernel_size 5 \
  --num_conv_layers 1 \
  --num_mlp_layers 5 \
  --use_SRM \
  --input_dropout 0.5 \
  --dropout 0.1 \
  --span_loss_type l1 \
  --lw_reg 1.0 \
  --lw_cls 5.0 \
  --lw_sal 0.0 \
  --lw_saliency 0.0 \
  --lw_wattn 1.0 \
  --lw_ms_align 1.0 \
  --mr_only \
  --eval_full_only \
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1.0 \
  --exist_gate_thd 0.5 \
  --nms_thd 0.7

# 2. Locate the best checkpoint
LATEST_DIR=$(ls -td results/scratch_training/hl-video_tef-amc_ft_scratch-* | head -n 1)
BEST_CKPT="${LATEST_DIR}/model_best.ckpt"

echo "2. Training finished. Best checkpoint found at ${BEST_CKPT}"

# 3. Create verification directory
VERIFY_DIR="results/scratch_training/verification_results"
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

# 4. Apply post-processing gate
echo "4. Applying AMC count-gating..."
python scripts/apply_amc_gating_cli.py \
  --input_path "${VERIFY_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --output_path "${VERIFY_DIR}/gated_submission.jsonl"

# 5. Run evaluation
echo "5. Evaluating gated predictions..."
python eval/eval_main.py \
  --submission_path "${VERIFY_DIR}/gated_submission.jsonl" \
  --gt_path data/label/Standard/test.jsonl \
  --save_path "${VERIFY_DIR}/gated_metrics.json" \
  --cls_thresholds 0.5 \
  --gmiou_cls_threshold 0.5

# 6. Run raw evaluation (without gating) for standard localization validation
echo "6. Evaluating raw predictions (without gating)..."
python eval/eval_main.py \
  --submission_path "${VERIFY_DIR}/hl_test_submission_nms_thd_0.7.jsonl" \
  --gt_path data/label/Standard/test.jsonl \
  --save_path "${VERIFY_DIR}/ungated_metrics.json" \
  --cls_thresholds 0.5 \
  --gmiou_cls_threshold 0.5

echo "======================================================================"
echo "Scratch training and verification pipeline finished successfully!"
echo "Gated metrics:   ${VERIFY_DIR}/gated_metrics.json"
echo "Ungated metrics: ${VERIFY_DIR}/ungated_metrics.json"
echo "======================================================================"
