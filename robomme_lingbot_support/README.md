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
from .va_robomme_i2va import va_robomme_i2va_cfg

VA_CONFIGS["robomme"] = va_robomme_cfg
VA_CONFIGS["robomme_train"] = va_robomme_train_cfg
VA_CONFIGS["robomme_i2av"] = va_robomme_i2va_cfg
```

`install_into_lingbot_va.sh` runs `tools/robomme/register_robomme_config.py`
which patches `wan_va/configs/__init__.py` to add the three lines above.
No dataset loader changes are needed.

The three configs follow the same layout as `va_libero_*.py` /
`va_robotwin_*.py`: `_cfg` is the base inference config, `_train_cfg`
extends it for FSDP training, and `_i2va` extends it for the standalone
image-to-video-action server (`CONFIG_NAME='robomme_i2av'`).

## Prepare Dataset

```bash
cd /mnt/hwdata/wangsen/WAM/lingbot-va
bash script/robomme_prepare_dataset.sh
```

Useful overrides:

```bash
RAW_DIR=/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme \
OUT_DIR=/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot \
WAN22_CKPT=/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers \
TARGET_FPS=15 \
SEGMENT_MODE=exec \
ACTION_KEY=eef_action \
CONVERT_WORKERS=16 \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash script/robomme_prepare_dataset.sh
```

`SEGMENT_MODE=exec` uses frames after RoboMME's `is_video_demo` prefix. Use
`SEGMENT_MODE=subgoal` if you want action_config segments from RoboMME subgoal
boundaries.

### Episode segmentation (`MAX_SAMPLED_FRAMES_PER_LATENT`)

By default `MAX_SAMPLED_FRAMES_PER_LATENT=0`, meaning **one `action_config`
segment per episode** — the same convention `DATA/libero_long` uses.
`tools/robomme/split_robomme_action_config.py --unsplit` runs
automatically after conversion to ensure that, even if a previous run
left split segments behind. The streaming Wan VAE encoder handles long
clips via 4-frame chunks, so a 1411-frame raw episode (the longest in
RoboMME) is encoded in one pass without OOM at 256x256.

Set `MAX_SAMPLED_FRAMES_PER_LATENT=81` (or any 4n+1 value) to fall back
to the older split-by-VAE-window behavior, which slices each episode's
single segment into shorter chunks. This is **only useful** if your
target_fps × longest_episode exceeds available VRAM during latent
extraction; the libero-style 1-segment-per-episode default is what
matches the eval-time conditioning distribution.

Migrate an existing dataset that was previously split:

```bash
# rewrites meta/episodes.jsonl to 1 segment per episode (writes a .bak)
python tools/robomme/split_robomme_action_config.py \
  --dataset-dir /mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot \
  --unsplit --backup

# re-extract latents for the new full-episode segments
GPU_IDS=0,1,2,3 SKIP_CONVERT=1 \
  bash script/robomme_prepare_dataset.sh
```

(The old per-chunk `.pth` files in `latents/` are harmless to leave on
disk; the loader only reads the segments listed in `episodes.jsonl`.)

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
LINGBOT_ROBOMME_DATASET=/path/to/robomme_lingbot \
LINGBOT_BASE_CKPT=/path/to/lingbot-va-base \
SAVE_ROOT=/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/robomme_lingbot_va \
bash script/robomme_train.sh
```

Hyperparameters (`learning_rate`, `batch_size`, `gradient_accumulation_steps`,
`num_steps`, `weight_decay`, `warmup_steps`, `save_interval`, `enable_wandb`,
…) are now literal values inside `wan_va/configs/va_robomme_train_cfg.py`,
matching the libero / robotwin pattern. Edit the file directly to change
them — the previous `LINGBOT_BATCH_SIZE` / `LINGBOT_GRAD_ACCUM` /
`LINGBOT_NUM_STEPS` env-var overrides have been removed for consistency
with the rest of the repo.

Defaults: `batch_size=1`, `gradient_accumulation_steps=10`,
`num_steps=15000`, `learning_rate=1e-5`, `weight_decay=1e-1`,
`warmup_steps=10`, `save_interval=500`, `enable_wandb=True`. If you do
not want wandb logging, set `va_robomme_train_cfg.enable_wandb = False`
in the file (or unset `WANDB_API_KEY` / `WANDB_BASE_URL` so
`wandb.init` no-ops).
