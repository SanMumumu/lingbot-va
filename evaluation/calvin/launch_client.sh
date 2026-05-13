#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CALVIN_ROOT="${CALVIN_ROOT:-/mnt/hwdata/wangsen/WAM/lingbot-va/simulator/calvin}"
CALVIN_DATASET_PATH="${CALVIN_DATASET_PATH:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/CALVIN/task_ABC_D}"
PORT="${PORT:-29057}"
OUT_DIR="${OUT_DIR:-outputs/calvin}"
CLIENT_CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-0}"
NUM_SEQUENCES="${NUM_SEQUENCES:-0}"
EP_LEN="${EP_LEN:-0}"

export PYTHONPATH="${REPO_ROOT}:${CALVIN_ROOT}:${CALVIN_ROOT}/calvin_models:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES}" python evaluation/calvin/client.py \
  --calvin-root "${CALVIN_ROOT}" \
  --dataset-path "${CALVIN_DATASET_PATH}" \
  --port "${PORT}" \
  --out-dir "${OUT_DIR}" \
  --num-sequences "${NUM_SEQUENCES}" \
  --ep-len "${EP_LEN}" \
  "$@"
