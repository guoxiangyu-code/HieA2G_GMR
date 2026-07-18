#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-1}"

echo "Starting fully-joint training (loading GMR baseline, unfrozen backbone from epoch 0)..."
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
  --lr 2e-4 \
  --lr_drop 10 \
  --wd 1e-4 \
  --n_epoch 15 \
  --max_es_cnt 15 \
  --bsz 256 \
  --eval_bsz 1 \
  --eval_epoch 1 \
  --device "${DEVICE}" \
  --results_root results/scratch_joint_baseline \
  --exp_id joint_baseline_run \
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
  --use_exist_head \
  --exist_pool mean \
  --exist_loss_coef 1.0 \
  --exist_gate_thd 0.5 \
  --eval_full_only \
  --mr_only \
  --null_anchor_coef 1.0 \
  --resume /home/guoxiangyu/GMR/generalized-moment-retrieval/Soccer-GMR/checkpoint/flashVTG_gmr

