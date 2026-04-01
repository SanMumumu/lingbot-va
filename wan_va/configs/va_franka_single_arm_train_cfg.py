# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import json
import os

from easydict import EasyDict

from .va_franka_cfg import va_franka_cfg


def _load_json_if_exists(path: str):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _build_inverse_index(action_dim: int, used_action_channel_ids: list[int]) -> list[int]:
    inverse_used_action_channel_ids = [len(used_action_channel_ids)] * action_dim
    for compact_idx, standard_idx in enumerate(used_action_channel_ids):
        inverse_used_action_channel_ids[standard_idx] = compact_idx
    return inverse_used_action_channel_ids


va_franka_single_arm_train_cfg = EasyDict(__name__="Config: VA franka single arm train")
va_franka_single_arm_train_cfg.update(va_franka_cfg)

# Replace this fallback path if you convert a different Franka dataset.
dataset_path = os.getenv("LINGBOT_FRANKA_DATASET_PATH", "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/PickBlueonRed_lerobot")
franka_meta = _load_json_if_exists(os.path.join(dataset_path, "meta", "franka_meta.json")) or {}
latent_meta = _load_json_if_exists(os.path.join(dataset_path, "meta", "latent_config.json")) or {}
norm_stat_path = os.getenv(
    "LINGBOT_FRANKA_NORM_STAT_PATH",
    os.path.join(dataset_path, "meta", "action_norm_stats.json"),
)
norm_stat_json = _load_json_if_exists(norm_stat_path) or {}

va_franka_single_arm_train_cfg.dataset_path = dataset_path
va_franka_single_arm_train_cfg.empty_emb_path = os.path.join(dataset_path, "empty_emb.pt")
va_franka_single_arm_train_cfg.enable_wandb = True
va_franka_single_arm_train_cfg.load_worker = int(os.getenv("LINGBOT_FRANKA_LOAD_WORKER", "16"))
va_franka_single_arm_train_cfg.save_interval = int(os.getenv("LINGBOT_FRANKA_SAVE_INTERVAL", "1000"))
va_franka_single_arm_train_cfg.gc_interval = int(os.getenv("LINGBOT_FRANKA_GC_INTERVAL", "50"))
va_franka_single_arm_train_cfg.cfg_prob = float(os.getenv("LINGBOT_FRANKA_CFG_PROB", "0.1"))

va_franka_single_arm_train_cfg.learning_rate = float(os.getenv("LINGBOT_FRANKA_LR", "1e-5"))
va_franka_single_arm_train_cfg.beta1 = 0.9
va_franka_single_arm_train_cfg.beta2 = 0.95
va_franka_single_arm_train_cfg.weight_decay = float(os.getenv("LINGBOT_FRANKA_WEIGHT_DECAY", "0.1"))
va_franka_single_arm_train_cfg.warmup_steps = int(os.getenv("LINGBOT_FRANKA_WARMUP_STEPS", "10"))
va_franka_single_arm_train_cfg.batch_size = int(os.getenv("LINGBOT_FRANKA_BATCH_SIZE", "1"))
va_franka_single_arm_train_cfg.gradient_accumulation_steps = int(
    os.getenv("LINGBOT_FRANKA_GRAD_ACCUM", "1")
)
va_franka_single_arm_train_cfg.num_steps = int(os.getenv("LINGBOT_FRANKA_NUM_STEPS", "50000"))

# Replace this path if you move to a different pretrained LingBot-VA base checkpoint.
va_franka_single_arm_train_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "LINGBOT_WAN22_MODEL_PATH",
    va_franka_cfg.wan22_pretrained_model_name_or_path,
)

va_franka_single_arm_train_cfg.obs_cam_keys = franka_meta.get(
    "obs_cam_keys",
    [
        "observation.images.left_camera",
        "observation.images.right_camera",
        "observation.images.wrist_camera",
    ],
)

va_franka_single_arm_train_cfg.height = int(
    os.getenv("LINGBOT_FRANKA_LATENT_HEIGHT", str(latent_meta.get("height", 224)))
)
va_franka_single_arm_train_cfg.width = int(
    os.getenv("LINGBOT_FRANKA_LATENT_WIDTH", str(latent_meta.get("width", 320)))
)

va_franka_single_arm_train_cfg.used_action_channel_ids = (
    norm_stat_json.get("used_action_channel_ids")
    or list(range(0, 7)) + list(range(14, 21)) + list(range(28, 29))
)
va_franka_single_arm_train_cfg.inverse_used_action_channel_ids = _build_inverse_index(
    action_dim=va_franka_single_arm_train_cfg.action_dim,
    used_action_channel_ids=va_franka_single_arm_train_cfg.used_action_channel_ids,
)

default_q01 = [0.0] * va_franka_single_arm_train_cfg.action_dim
default_q99 = [0.0] * va_franka_single_arm_train_cfg.action_dim
default_q99[28] = 1.0
va_franka_single_arm_train_cfg.norm_stat = {
    "q01": norm_stat_json.get("q01", default_q01),
    "q99": norm_stat_json.get("q99", default_q99),
}
