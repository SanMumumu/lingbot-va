#!/usr/bin/env bash
set -euo pipefail

# Prepare CALVIN ABC-D LeRobot data for LingBot-VA without materializing videos.

DATASET_DIR="${DATASET_DIR:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot}"
HF_REPO_ID="${HF_REPO_ID:-fywang/calvin-task-ABC-D-lerobot}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WAN22_CKPT="${WAN22_CKPT:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
TARGET_FPS="${TARGET_FPS:-10}"
LATENT_GPUS="${LATENT_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"
# 默认不下载数据。请先运行 DATA_MY/download_calvin.sh；如果确实想让本脚本下载，设置 DOWNLOAD=1。
DOWNLOAD="${DOWNLOAD:-0}"
SKIP_STATS="${SKIP_STATS:-0}"
SKIP_LATENTS="${SKIP_LATENTS:-0}"
VERIFY_DEEP_LIMIT="${VERIFY_DEEP_LIMIT:-32}"
MAX_EPISODES="${MAX_EPISODES:-0}"
MAX_STAT_EPISODES="${MAX_STAT_EPISODES:-0}"
MAX_SAMPLED_FRAMES_PER_LATENT="${MAX_SAMPLED_FRAMES_PER_LATENT:-0}"
LOG_DIR="${LOG_DIR:-${DATASET_DIR}/logs}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT

DOWNLOAD_ARG=()
if [[ "${DOWNLOAD}" == "1" ]]; then
  DOWNLOAD_ARG+=(--download)
fi

python tools/calvin/prepare_calvin_lingbot_dataset.py \
  --stage metadata \
  --dataset-dir "${DATASET_DIR}" \
  --repo-id "${HF_REPO_ID}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  "${DOWNLOAD_ARG[@]}"

if [[ "${SKIP_STATS}" != "1" ]]; then
  python tools/calvin/prepare_calvin_lingbot_dataset.py \
    --stage stats \
    --dataset-dir "${DATASET_DIR}" \
    --repo-id "${HF_REPO_ID}" \
    --hf-endpoint "${HF_ENDPOINT}" \
    --max-stat-episodes "${MAX_STAT_EPISODES}"
fi

if [[ "${SKIP_LATENTS}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"

  if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
  else
    GPU_ID_ARRAY=()
    for ((gpu=0; gpu<LATENT_GPUS; gpu++)); do
      GPU_ID_ARRAY+=("${gpu}")
    done
  fi

  NUM_LATENT_SHARDS="${#GPU_ID_ARRAY[@]}"
  if [[ "${NUM_LATENT_SHARDS}" -lt 1 ]]; then
    echo "No GPUs selected. Set GPU_IDS=0,1 or LATENT_GPUS=N."
    exit 1
  fi

  latent_pids=()
  for ((shard=0; shard<NUM_LATENT_SHARDS; shard++)); do
    gpu="${GPU_ID_ARRAY[$shard]}"
    log_file="${LOG_DIR}/calvin_latent_gpu${gpu}_shard${shard}.log"
    echo "Starting CALVIN latent shard ${shard}/${NUM_LATENT_SHARDS} on CUDA_VISIBLE_DEVICES=${gpu}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python tools/calvin/prepare_calvin_lingbot_dataset.py \
      --stage latents \
      --dataset-dir "${DATASET_DIR}" \
      --repo-id "${HF_REPO_ID}" \
      --hf-endpoint "${HF_ENDPOINT}" \
      --wan22-path "${WAN22_CKPT}" \
      --height "${IMAGE_SIZE}" \
      --width "${IMAGE_SIZE}" \
      --target-fps "${TARGET_FPS}" \
      --max-sampled-frames "${MAX_SAMPLED_FRAMES_PER_LATENT}" \
      --max-episodes "${MAX_EPISODES}" \
      --device cuda:0 \
      --num-shards "${NUM_LATENT_SHARDS}" \
      --shard-index "${shard}" \
      --resume \
      >"${log_file}" 2>&1 &
    latent_pids+=("$!")
  done

  latent_failed=0
  for pid in "${latent_pids[@]}"; do
    if ! wait "${pid}"; then
      latent_failed=1
    fi
  done

  if [[ "${latent_failed}" != "0" ]]; then
    echo "One or more CALVIN latent extraction shards failed. Last 80 lines from each shard log:"
    for log_file in "${LOG_DIR}"/calvin_latent_gpu*_shard*.log; do
      echo "========== ${log_file} =========="
      tail -n 80 "${log_file}" || true
    done
    exit 1
  fi
fi

python tools/calvin/prepare_calvin_lingbot_dataset.py \
  --stage verify \
  --dataset-dir "${DATASET_DIR}" \
  --repo-id "${HF_REPO_ID}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --max-episodes "${MAX_EPISODES}" \
  --deep-limit "${VERIFY_DEEP_LIMIT}"

echo "Prepared LingBot-VA CALVIN dataset at ${DATASET_DIR}"
