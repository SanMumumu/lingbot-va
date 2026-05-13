#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CONFIG_NAME="${CONFIG_NAME:-calvin}"
export LINGBOT_CALVIN_CKPT="${LINGBOT_CALVIN_CKPT:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base}"

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29062}"
PORT="${PORT:-29057}"
SAVE_ROOT="${SAVE_ROOT:-visualization/calvin}"

python tools/robomme/set_transformer_attn_mode.py \
  --ckpt "${LINGBOT_CALVIN_CKPT}" \
  --mode torch

python -m torch.distributed.run \
  --nproc_per_node "${NGPU}" \
  --master_port "${MASTER_PORT}" \
  wan_va/wan_va_server.py \
  --config-name "${CONFIG_NAME}" \
  --port "${PORT}" \
  --save_root "${SAVE_ROOT}"
