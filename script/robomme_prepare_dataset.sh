#!/usr/bin/env bash
set -euo pipefail

# Run from the lingbot-va repository root after copying this support bundle in.

RAW_DIR="${RAW_DIR:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomee}"
OUT_DIR="${OUT_DIR:-/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot}"
WAN22_CKPT="${WAN22_CKPT:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers}"

FPS="${FPS:-30}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
TARGET_FPS="${TARGET_FPS:-15}"
SEGMENT_MODE="${SEGMENT_MODE:-exec}"
ACTION_KEY="${ACTION_KEY:-eef_action}"
SKIP_BAD_FILES="${SKIP_BAD_FILES:-0}"
CONVERT_WORKERS="${CONVERT_WORKERS:-16}"
LATENT_GPUS="${LATENT_GPUS:-8}"
GPU_IDS="${GPU_IDS:-}"
VIDEO_CODEC="${VIDEO_CODEC:-mpeg4}"
CHUNKS_SIZE="${CHUNKS_SIZE:-100}"
VERIFY_DEEP_LIMIT="${VERIFY_DEEP_LIMIT:-32}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
MAX_SAMPLED_FRAMES_PER_LATENT="${MAX_SAMPLED_FRAMES_PER_LATENT:-81}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONVERT_EXTRA_ARGS=()
if [[ "${SKIP_BAD_FILES}" == "1" ]]; then
  CONVERT_EXTRA_ARGS+=(--skip-bad-files)
fi

if [[ "${SKIP_CONVERT}" == "1" ]]; then
  echo "Skipping H5 -> LeRobot conversion because SKIP_CONVERT=1."
else
  python tools/robomme/convert_robomme_h5_to_lingbot_lerobot.py \
    --raw-dir "${RAW_DIR}" \
    --output-dir "${OUT_DIR}" \
    --fps "${FPS}" \
    --image-size "${IMAGE_SIZE}" \
    --segment-mode "${SEGMENT_MODE}" \
    --action-key "${ACTION_KEY}" \
    --num-workers "${CONVERT_WORKERS}" \
    --video-codec "${VIDEO_CODEC}" \
    --chunks-size "${CHUNKS_SIZE}" \
    --overwrite \
    "${CONVERT_EXTRA_ARGS[@]}"
fi

python tools/robomme/split_robomme_action_config.py \
  --dataset-dir "${OUT_DIR}" \
  --target-fps "${TARGET_FPS}" \
  --max-sampled-frames "${MAX_SAMPLED_FRAMES_PER_LATENT}" \
  --backup

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
  echo "No GPUs selected. Set GPU_IDS=2,3,4,5,6,7 or LATENT_GPUS=N."
  exit 1
fi

done_file="${LOG_DIR}/latent_shards.done"
rm -f "${done_file}"
python tools/robomme/monitor_lingbot_latents.py \
  --dataset-dir "${OUT_DIR}" \
  --done-file "${done_file}" \
  &
monitor_pid="$!"

latent_pids=()
for ((shard=0; shard<NUM_LATENT_SHARDS; shard++)); do
  gpu="${GPU_ID_ARRAY[$shard]}"
  log_file="${LOG_DIR}/latent_gpu${gpu}_shard${shard}.log"
  echo "Starting latent shard ${shard}/${NUM_LATENT_SHARDS} on CUDA_VISIBLE_DEVICES=${gpu}; log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python tools/robomme/extract_lingbot_latents.py \
    --dataset-dir "${OUT_DIR}" \
    --wan22-path "${WAN22_CKPT}" \
    --height "${IMAGE_SIZE}" \
    --width "${IMAGE_SIZE}" \
    --target-fps "${TARGET_FPS}" \
    --max-sampled-frames "${MAX_SAMPLED_FRAMES_PER_LATENT}" \
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
touch "${done_file}"
wait "${monitor_pid}" || true

if [[ "${latent_failed}" != "0" ]]; then
  echo "One or more latent extraction shards failed. Last 80 lines from each shard log:"
  for log_file in "${LOG_DIR}"/latent_gpu*_shard*.log; do
    echo "========== ${log_file} =========="
    tail -n 80 "${log_file}" || true
  done
  exit 1
fi

python tools/robomme/verify_lingbot_robomme_dataset.py \
  --dataset-dir "${OUT_DIR}" \
  --deep-limit "${VERIFY_DEEP_LIMIT}"

echo "Prepared LingBot-VA RoboMME dataset at ${OUT_DIR}"
