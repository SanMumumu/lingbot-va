#!/usr/bin/env bash
# Launch the LingBot-VA websocket inference server for the RoboMME checkpoint.
#
# Can be run from anywhere. Override REPO_ROOT/INFER_CKPT/DATASET_DIR/SAVE_ROOT
# when moving the experiment to another machine.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
INFER_CKPT="${INFER_CKPT:-${REPO_ROOT}/CKPTS/lingbot-va-robomme-step15000}"
DATASET_DIR="${LINGBOT_ROBOMME_DATASET:-${REPO_ROOT}/DATA/robomme_lingbot}"
PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29061}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/OUTPUTS/robomme_eval/visualization}"

if [[ ! -d "${INFER_CKPT}/transformer" ]]; then
  echo "Inference checkpoint not found at ${INFER_CKPT}." >&2
  echo "Run robomme_lingbot_support/tools/assemble_inference_ckpt.sh first." >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}"

# va_robomme_cfg pulls these from env vars:
#   LINGBOT_BASE_CKPT       -> wan22_pretrained_model_name_or_path (transformer + vae + ...)
#   LINGBOT_ROBOMME_DATASET -> dataset_path (used to load meta/robomme_action_stats.json)
export LINGBOT_BASE_CKPT="${INFER_CKPT}"
export LINGBOT_ROBOMME_DATASET="${DATASET_DIR}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"

python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port "${MASTER_PORT}" \
  wan_va/wan_va_server.py \
  --config-name robomme \
  --port "${PORT}" \
  --save_root "${SAVE_ROOT}"
