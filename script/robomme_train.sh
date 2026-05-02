#!/usr/bin/env bash
# NGPU=4 LINGBOT_ENABLE_WANDB=1 bash script/robomme_train.sh
set -euo pipefail

# Run from the lingbot-va repository root.

export LINGBOT_ROBOMME_DATASET="${LINGBOT_ROBOMME_DATASET:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot}"
export LINGBOT_BASE_CKPT="${LINGBOT_BASE_CKPT:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base}"
export LINGBOT_ENABLE_WANDB="${LINGBOT_ENABLE_WANDB:-0}"
export TOKENIZERS_PARALLELISM=false

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/robomme_lingbot_va}"

# LingBot-VA requires flex attention for training.
python tools/robomme/set_transformer_attn_mode.py \
  --ckpt "${LINGBOT_BASE_CKPT}" \
  --mode flex

NGPU="${NGPU}" MASTER_PORT="${MASTER_PORT}" CONFIG_NAME="robomme_train" \
  bash script/run_va_posttrain.sh --save-root "${SAVE_ROOT}"
