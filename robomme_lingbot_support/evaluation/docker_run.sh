#!/usr/bin/env bash
# Start a RoboMME Docker client shell with host paths mounted consistently.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

IMAGE="${IMAGE:-robomme:cuda12.8}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/OUTPUTS/robomme_eval}"
DATASET_DIR="${LINGBOT_ROBOMME_DATASET:-${REPO_ROOT}/DATA/robomme_lingbot}"
ROBOMME_ROOT="${ROBOMME_ROOT:-${REPO_ROOT}/simulator/robomme_benchmark}"

mkdir -p "${OUT_DIR}"

docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "${SCRIPT_DIR}/client.py:/app/lingbot_client/client.py:ro" \
  -v "${SCRIPT_DIR}/launch_clients_multi.sh:/app/lingbot_client/launch_clients_multi.sh:ro" \
  -v "${REPO_ROOT}/wan_va/utils/Simple_Remote_Infer:/app/lingbot_client/wan_va/utils/Simple_Remote_Infer:ro" \
  -v "${ROBOMME_ROOT}/src:/app/src:ro" \
  -v "${ROBOMME_ROOT}/scripts:/app/scripts:ro" \
  -v "${DATASET_DIR}:/app/data/robomme_lingbot:ro" \
  -v "${OUT_DIR}:/app/runs/lingbot_eval" \
  "${IMAGE}"
