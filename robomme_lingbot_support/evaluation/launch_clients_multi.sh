#!/usr/bin/env bash
# Launch one RoboMME evaluation client shard per GPU inside Docker.
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
HOST="${HOST:-172.17.0.1}"
BASE_PORT="${BASE_PORT:-29056}"
NUM_EPISODES="${NUM_EPISODES:-50}"
ACTION_SPACE="${ACTION_SPACE:-ee_pose}"
MAX_STEPS="${MAX_STEPS:-1300}"
OUT_DIR="${OUT_DIR:-/app/runs/lingbot_eval}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
FIRST_FRAME_CONDITIONING="${FIRST_FRAME_CONDITIONING:-0}"
SHARD_MODE="${SHARD_MODE:-balanced}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"

echo "Launching ${NUM_SHARDS} RoboMME client shard(s) on GPUs: ${GPUS}"
echo "Task split mode: ${SHARD_MODE}"

pids=()
for slot in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$slot]}"
  port=$((BASE_PORT + slot))
  log="${LOG_DIR}/client_gpu${gpu}_shard${slot}.log"
  echo "GPU ${gpu}: shard ${slot}/${NUM_SHARDS}, server port ${port}, log ${log}"

  cmd=(python -u client.py
       --host "${HOST}"
       --port "${port}"
       --num-shards "${NUM_SHARDS}"
       --shard-index "${slot}"
       --num-episodes "${NUM_EPISODES}"
       --action-space "${ACTION_SPACE}"
       --max-steps "${MAX_STEPS}"
       --out-dir "${OUT_DIR}"
       --shard-mode "${SHARD_MODE}")

  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    cmd+=(--save-video)
  fi
  if [[ "${FIRST_FRAME_CONDITIONING}" == "1" ]]; then
    cmd+=(--first-frame-conditioning)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log}" 2>&1 &
  pids+=("$!")
done

echo "Client PIDs: ${pids[*]}"
echo "Tail logs with: tail -f ${LOG_DIR}/client_gpu*.log"
trap 'echo "Stopping clients..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

OUT_DIR="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
shards = sorted(out_dir.glob("summary_shard*.json"))
combined = {"per_env": {}, "per_shard": {}, "average_success_rate": 0.0, "num_envs": 0}
for shard in shards:
    data = json.loads(shard.read_text(encoding="utf-8"))
    combined["per_shard"][shard.stem] = data.get("average_success_rate", 0.0)
    combined["per_env"].update(data.get("per_env", {}))
rates = [v["succ_rate"] for v in combined["per_env"].values()]
combined["num_envs"] = len(rates)
combined["average_success_rate"] = float(sum(rates) / max(len(rates), 1))
(out_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Aggregated {combined['num_envs']} task(s); average success rate = {combined['average_success_rate']:.3f}")
print(f"Final summary: {out_dir / 'summary.json'}")
PY

exit "${failed}"
