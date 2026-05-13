# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


def _load_robomme_action_stats(dataset_path):
    """Read q01/q99 and used_action_channel_ids from the converter's stats JSON.

    Falls back to a 7-D eef_action layout when the file is absent (e.g. when
    importing the config before the dataset has been prepared).
    """
    stats_path = os.environ.get(
        'LINGBOT_ROBOMME_NORM_STAT',
        os.path.join(dataset_path, 'meta', 'robomme_action_stats.json'),
    )
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        q01 = [0.0] * 30
        q99 = [0.0] * 30
        for i, v in enumerate(data['q01'][:30]):
            q01[i] = float(v)
        for i, v in enumerate(data['q99'][:30]):
            q99[i] = float(v)
        used = [int(v) for v in data.get('used_action_channel_ids', list(range(7)))]
        return used, {'q01': q01, 'q99': q99}
    return list(range(7)), {
        'q01': [-1.0] * 7 + [0.] * 23,
        'q99': [1.0] * 7 + [0.] * 23,
    }


va_robomme_cfg = EasyDict(__name__='Config: VA robomme')
va_robomme_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_robomme_cfg.dataset_path = os.environ.get(
    'LINGBOT_ROBOMME_DATASET',
    '/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot',
)
va_robomme_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'LINGBOT_BASE_CKPT',
    '/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base',
)

va_robomme_cfg.attn_window = 30
va_robomme_cfg.frame_chunk_size = 4
va_robomme_cfg.env_type = 'none'

va_robomme_cfg.height = 256
va_robomme_cfg.width = 256
va_robomme_cfg.action_dim = 30
va_robomme_cfg.action_per_frame = 8
va_robomme_cfg.obs_cam_keys = [
    'observation.images.front', 'observation.images.wrist'
]
va_robomme_cfg.guidance_scale = 5
va_robomme_cfg.action_guidance_scale = 1

va_robomme_cfg.num_inference_steps = 20
va_robomme_cfg.video_exec_step = -1
va_robomme_cfg.action_num_inference_steps = 50

va_robomme_cfg.snr_shift = 5.0
va_robomme_cfg.action_snr_shift = 0.05

# RoboMME's eef_action is 7-D ([x,y,z,roll,pitch,yaw,gripper]); the converter
# writes it into channels matching `used_action_channel_ids` from the dataset
# stats (defaults to 0..6 if unavailable).
va_robomme_cfg.used_action_channel_ids, va_robomme_cfg.norm_stat = (
    _load_robomme_action_stats(va_robomme_cfg.dataset_path)
)
inverse_used_action_channel_ids = [len(va_robomme_cfg.used_action_channel_ids)
                                   ] * va_robomme_cfg.action_dim
for i, j in enumerate(va_robomme_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robomme_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robomme_cfg.action_norm_method = 'quantiles'

# RoboMME-specific visualization knobs (consumed by the eval client).
va_robomme_cfg.save_visualization = True
va_robomme_cfg.visualization_fps = 10
