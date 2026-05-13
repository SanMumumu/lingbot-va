#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export LINGBOT_CALVIN_DATASET="${LINGBOT_CALVIN_DATASET:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot}"
export LINGBOT_CALVIN_CKPT="${LINGBOT_CALVIN_CKPT:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base}"
export SAVE_ROOT="${SAVE_ROOT:-/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/calvin_lingbot_va}"
export CONFIG_NAME="${CONFIG_NAME:-calvin_train}"
export NGPU="${NGPU:-8}"

python tools/robomme/set_transformer_attn_mode.py \
  --ckpt "${LINGBOT_CALVIN_CKPT}" \
  --mode flex

bash script/run_va_posttrain.sh "$@"
