#!/usr/bin/env bash
# Assemble the trained RoboMME transformer with frozen LingBot-VA base modules.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

TRAINED_TRANSFORMER_DIR="${TRAINED_TRANSFORMER_DIR:-${REPO_ROOT}/OUTPUTS/robomme_lingbot_va/checkpoints/checkpoint_step_15000/transformer}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${REPO_ROOT}/CKPTS/lingbot-va-base}"
INFER_CKPT="${INFER_CKPT:-${REPO_ROOT}/CKPTS/lingbot-va-robomme-step15000}"

cd "${REPO_ROOT}"

bash robomme_lingbot_support/tools/assemble_inference_ckpt.sh \
  "${TRAINED_TRANSFORMER_DIR}" \
  "${BASE_CKPT_DIR}" \
  "${INFER_CKPT}"
