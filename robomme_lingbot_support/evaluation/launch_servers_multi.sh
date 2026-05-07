#!/usr/bin/env bash
# Launch one LingBot-VA RoboMME websocket server per GPU on the host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

GPUS="${GPUS:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-29056}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29061}"
INFER_CKPT="${INFER_CKPT:-${REPO_ROOT}/CKPTS/lingbot-va-robomme-step15000}"
DATASET_DIR="${LINGBOT_ROBOMME_DATASET:-${REPO_ROOT}/DATA/robomme_lingbot}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/OUTPUTS/robomme_eval/visualization}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/OUTPUTS/robomme_eval/logs}"

if [[ ! -d "${INFER_CKPT}/transformer" ]]; then
  echo "Inference checkpoint not found at ${INFER_CKPT}." >&2
  echo "Run robomme_lingbot_support/evaluation/prepare_ckpt.sh first." >&2
  exit 1
fi

mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"
export LINGBOT_BASE_CKPT="${INFER_CKPT}"
export LINGBOT_ROBOMME_DATASET="${DATASET_DIR}"
export TOKENIZERS_PARALLELISM=false

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
echo "Launching ${#GPU_ARR[@]} LingBot server(s) on GPUs: ${GPUS}"
echo "Ports: $(for i in "${!GPU_ARR[@]}"; do printf '%s ' "$((BASE_PORT + i))"; done)"

cd "${REPO_ROOT}"

pids=()
for slot in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$slot]}"
  port=$((BASE_PORT + slot))
  master_port=$((BASE_MASTER_PORT + slot))
  log="${LOG_DIR}/server_gpu${gpu}_port${port}.log"
  echo "GPU ${gpu}: port ${port}, master_port ${master_port}, log ${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "${master_port}" \
    wan_va/wan_va_server.py \
    --config-name robomme \
    --port "${port}" \
    --save_root "${SAVE_ROOT}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

echo "Server PIDs: ${pids[*]}"
echo "Tail logs with: tail -f ${LOG_DIR}/server_gpu*.log"
trap 'echo "Stopping servers..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM
wait "${pids[@]}"
