# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""RoboMME config for LingBot-VA.

This is intentionally small: the dataset is converted into the same
LatentLeRobotDataset format used by the existing training code, so no dataset
loader changes are required.
"""

import json
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _pad(values, length: int = 30):
    out = [0.0] * length
    for i, value in enumerate(values[:length]):
        out[i] = float(value)
    return out


def _load_norm_stat(dataset_path: str) -> dict:
    stats_path = _env(
        "LINGBOT_ROBOMME_NORM_STAT",
        os.path.join(dataset_path, "meta", "robomme_action_stats.json"),
    )
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "q01": _pad(data["q01"]),
            "q99": _pad(data["q99"]),
        }
    # Fallback only for import-time sanity before the dataset is prepared.
    return {
        "q01": [-1.0] * 7 + [0.0] * 23,
        "q99": [1.0] * 7 + [0.0] * 23,
    }


def _load_action_source_dim(dataset_path: str) -> int:
    stats_path = _env(
        "LINGBOT_ROBOMME_NORM_STAT",
        os.path.join(dataset_path, "meta", "robomme_action_stats.json"),
    )
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("action_source_dim", 7))
    return 7


def _load_used_action_channel_ids(dataset_path: str) -> list[int]:
    stats_path = _env(
        "LINGBOT_ROBOMME_NORM_STAT",
        os.path.join(dataset_path, "meta", "robomme_action_stats.json"),
    )
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "used_action_channel_ids" in data:
            return [int(v) for v in data["used_action_channel_ids"]]
    # eef_action maps xyz/rpy to left EEF 0:6 and gripper to channel 28.
    return list(range(6)) + [28]


va_robomme_cfg = EasyDict(__name__="Config: VA robomme")
va_robomme_cfg.update(va_shared_cfg)

va_robomme_cfg.dataset_path = _env(
    "LINGBOT_ROBOMME_DATASET",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot",
)

# For training this path should point to lingbot-va-base, whose transformer is
# used as the initialization checkpoint.
va_robomme_cfg.wan22_pretrained_model_name_or_path = _env(
    "LINGBOT_BASE_CKPT",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base",
)

va_robomme_cfg.attn_window = int(_env("LINGBOT_ROBOMME_ATTN_WINDOW", "30"))
va_robomme_cfg.frame_chunk_size = int(_env("LINGBOT_ROBOMME_FRAME_CHUNK_SIZE", "4"))
va_robomme_cfg.env_type = "none"

va_robomme_cfg.height = int(_env("LINGBOT_ROBOMME_HEIGHT", "256"))
va_robomme_cfg.width = int(_env("LINGBOT_ROBOMME_WIDTH", "256"))
va_robomme_cfg.action_dim = 30

# With the provided latent extraction default target_fps=15 and source fps=30,
# LingBot's dataset alignment produces frame_stride * 4 = 8 actions per latent
# frame. If you change target_fps, set this consistently for inference.
va_robomme_cfg.action_per_frame = int(_env("LINGBOT_ROBOMME_ACTION_PER_FRAME", "8"))
va_robomme_cfg.obs_cam_keys = [
    "observation.images.front",
    "observation.images.wrist",
]

va_robomme_cfg.guidance_scale = 5
va_robomme_cfg.action_guidance_scale = 1
va_robomme_cfg.num_inference_steps = 20
va_robomme_cfg.video_exec_step = -1
va_robomme_cfg.action_num_inference_steps = 50
va_robomme_cfg.snr_shift = 5.0
va_robomme_cfg.action_snr_shift = float(_env("LINGBOT_ROBOMME_ACTION_SNR_SHIFT", "0.05"))
va_robomme_cfg.save_visualization = _env("LINGBOT_ROBOMME_SAVE_PRED_VIDEO", "1").lower() not in {
    "0",
    "false",
    "no",
}
va_robomme_cfg.visualization_fps = int(_env("LINGBOT_ROBOMME_VISUALIZATION_FPS", "10"))

# The converted dataset stores actions in LingBot's 30-D standard layout.
va_robomme_cfg.used_action_channel_ids = _load_used_action_channel_ids(va_robomme_cfg.dataset_path)
inverse_used_action_channel_ids = [len(va_robomme_cfg.used_action_channel_ids)] * va_robomme_cfg.action_dim
for i, j in enumerate(va_robomme_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robomme_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robomme_cfg.action_norm_method = "quantiles"
va_robomme_cfg.norm_stat = _load_norm_stat(va_robomme_cfg.dataset_path)
