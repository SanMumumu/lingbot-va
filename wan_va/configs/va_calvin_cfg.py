# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from pathlib import Path

from easydict import EasyDict

from .shared_config import va_shared_cfg


CALVIN_DATASET_PATH = os.getenv(
    "LINGBOT_CALVIN_DATASET",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot",
)


def _load_calvin_norm_stat(dataset_path):
    stats_path = Path(dataset_path) / "meta" / "calvin_action_stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        if len(stats.get("q01", [])) == 30 and len(stats.get("q99", [])) == 30:
            return {"q01": stats["q01"], "q99": stats["q99"]}
    return {
        "q01": [-1.0] * 7 + [0.0] * 23,
        "q99": [1.0] * 7 + [0.0] * 23,
    }


va_calvin_cfg = EasyDict(__name__='Config: VA calvin')
va_calvin_cfg.update(va_shared_cfg)

va_calvin_cfg.infer_mode = 'server'
va_calvin_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "LINGBOT_CALVIN_CKPT",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base",
)

va_calvin_cfg.attn_window = 30
va_calvin_cfg.frame_chunk_size = 4
va_calvin_cfg.env_type = 'none'

va_calvin_cfg.height = 128
va_calvin_cfg.width = 128
va_calvin_cfg.action_dim = 30
va_calvin_cfg.action_per_frame = 4
va_calvin_cfg.obs_cam_keys = [
    'observation.images.top',
    'observation.images.wrist',
]
va_calvin_cfg.guidance_scale = 5
va_calvin_cfg.action_guidance_scale = 1

va_calvin_cfg.num_inference_steps = 20
va_calvin_cfg.video_exec_step = -1
va_calvin_cfg.action_num_inference_steps = 50

va_calvin_cfg.snr_shift = 5.0
va_calvin_cfg.action_snr_shift = 0.05

va_calvin_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [len(va_calvin_cfg.used_action_channel_ids)] * va_calvin_cfg.action_dim
for i, j in enumerate(va_calvin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_calvin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_calvin_cfg.action_norm_method = 'quantiles'
va_calvin_cfg.norm_stat = _load_calvin_norm_stat(CALVIN_DATASET_PATH)
