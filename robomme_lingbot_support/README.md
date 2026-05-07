# RoboMME -> LingBot-VA Support

This bundle converts RoboMME HDF5 data to the LingBot-VA training format and
adds a minimal RoboMME training config. The conversion is episode-parallel and
latent extraction is GPU-sharded.

Default paths match the requested machine layout:

- raw RoboMME H5: `/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomee`
- converted dataset: `/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot`
- LingBot-VA base checkpoint: `/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base`
- Wan2.2 Diffusers checkpoint: `/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers`

## Install into LingBot-VA

Copy this `robomme_lingbot_support` folder to the machine, then run:

```bash
cd robomme_lingbot_support
bash install_into_lingbot_va.sh /mnt/hwdata/wangsen/WAM/lingbot-va
```

The only LingBot-VA code registration is:

```python
from .va_robomme_cfg import va_robomme_cfg
from .va_robomme_train_cfg import va_robomme_train_cfg

VA_CONFIGS["robomme"] = va_robomme_cfg
VA_CONFIGS["robomme_train"] = va_robomme_train_cfg
```

No dataset loader changes are needed.

## Prepare Dataset

```bash
cd /mnt/hwdata/wangsen/WAM/lingbot-va
bash script/robomme_prepare_dataset.sh
```

Useful overrides:

```bash
RAW_DIR=/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomee \
OUT_DIR=/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot \
WAN22_CKPT=/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers \
TARGET_FPS=15 \
SEGMENT_MODE=exec \
ACTION_KEY=eef_action \
CONVERT_WORKERS=16 \
GPU_IDS=2,3,4,5,6,7 \
MAX_SAMPLED_FRAMES_PER_LATENT=81 \
bash script/robomme_prepare_dataset.sh
```

`SEGMENT_MODE=exec` uses frames after RoboMME's `is_video_demo` prefix. Use
`SEGMENT_MODE=subgoal` if you want action_config segments from RoboMME subgoal
boundaries.
`MAX_SAMPLED_FRAMES_PER_LATENT=81` splits long action_config entries into
VAE-friendly windows and keeps frame counts aligned to Wan's 4x temporal
compression.

Conversion failure reports:

- H5 inspection failures stop before conversion and name the bad file.
- Episode conversion failures write `meta/conversion_errors.jsonl`.
- Latent extraction failures write `meta/latent_errors_shardXXX.jsonl` and GPU logs under `logs/`.
- Final verification failures write `meta/verify_errors.jsonl`.

Common recovery commands:

```bash
# Skip a corrupted H5 file only for debugging; fix the file for final training.
SKIP_BAD_FILES=1 bash script/robomme_prepare_dataset.sh

# Load all latent files during verification instead of sampling 32.
VERIFY_DEEP_LIMIT=-1 bash script/robomme_prepare_dataset.sh

# Resume after conversion already succeeded; this only rewrites action_config
# windows if needed and extracts missing latent files.
SKIP_CONVERT=1 GPU_IDS=2 bash script/robomme_prepare_dataset.sh
```

## Train

```bash
cd /mnt/hwdata/wangsen/WAM/lingbot-va
NGPU=8 bash script/robomme_train.sh
```

Useful overrides:

```bash
NGPU=4 \
LINGBOT_BATCH_SIZE=1 \
LINGBOT_GRAD_ACCUM=16 \
LINGBOT_NUM_STEPS=10000 \
SAVE_ROOT=/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/robomme_lingbot_va \
bash script/robomme_train.sh
```
