#!/usr/bin/env bash
# Assemble a LingBot-VA inference checkpoint by combining the trained
# transformer with the frozen VAE / tokenizer / text_encoder from the base ckpt.
#
# Usage:
#   bash robomme_lingbot_support/tools/assemble_inference_ckpt.sh \
#     [TRAINED_TRANSFORMER_DIR] [BASE_CKPT_DIR] [OUT_CKPT_DIR]
#
# Defaults match the user's training output at step 15000.
set -euo pipefail

TRAINED_TRANSFORMER_DIR="${1:-/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/robomme_lingbot_va/checkpoints/checkpoint_step_15000/transformer}"
BASE_CKPT_DIR="${2:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base}"
OUT_CKPT_DIR="${3:-/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-robomme-step15000}"

if [[ ! -f "${TRAINED_TRANSFORMER_DIR}/config.json" || ! -f "${TRAINED_TRANSFORMER_DIR}/diffusion_pytorch_model.safetensors" ]]; then
  echo "Trained transformer not found at ${TRAINED_TRANSFORMER_DIR}" >&2
  exit 1
fi
for sub in vae tokenizer text_encoder; do
  if [[ ! -d "${BASE_CKPT_DIR}/${sub}" ]]; then
    echo "Base ckpt missing ${sub}/: ${BASE_CKPT_DIR}/${sub}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_CKPT_DIR}"

# Copy transformer (we must mutate config.json's attn_mode, so don't symlink).
rm -rf "${OUT_CKPT_DIR}/transformer"
mkdir -p "${OUT_CKPT_DIR}/transformer"
cp "${TRAINED_TRANSFORMER_DIR}/config.json" "${OUT_CKPT_DIR}/transformer/config.json"
# Hardlink the safetensors if possible (saves ~10 GB), fall back to copy.
ln "${TRAINED_TRANSFORMER_DIR}/diffusion_pytorch_model.safetensors" \
   "${OUT_CKPT_DIR}/transformer/diffusion_pytorch_model.safetensors" 2>/dev/null \
  || cp "${TRAINED_TRANSFORMER_DIR}/diffusion_pytorch_model.safetensors" \
        "${OUT_CKPT_DIR}/transformer/diffusion_pytorch_model.safetensors"

# Symlink frozen modules from the base checkpoint.
for sub in vae tokenizer text_encoder; do
  rm -rf "${OUT_CKPT_DIR}/${sub}"
  ln -s "${BASE_CKPT_DIR}/${sub}" "${OUT_CKPT_DIR}/${sub}"
done

# Flip attn_mode from flex (training) to torch (inference).
python tools/robomme/set_transformer_attn_mode.py --ckpt "${OUT_CKPT_DIR}" --mode torch

echo "Assembled inference checkpoint at: ${OUT_CKPT_DIR}"
ls -la "${OUT_CKPT_DIR}"
