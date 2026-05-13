# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_calvin_cfg import CALVIN_DATASET_PATH, _load_calvin_norm_stat, va_calvin_cfg


va_calvin_train_cfg = EasyDict(__name__='Config: VA calvin train')
va_calvin_train_cfg.update(va_calvin_cfg)

va_calvin_train_cfg.dataset_path = os.getenv("LINGBOT_CALVIN_DATASET", CALVIN_DATASET_PATH)
va_calvin_train_cfg.empty_emb_path = os.path.join(va_calvin_train_cfg.dataset_path, "empty_emb.pt")
va_calvin_train_cfg.norm_stat = _load_calvin_norm_stat(va_calvin_train_cfg.dataset_path)
va_calvin_train_cfg.save_root = os.getenv(
    "SAVE_ROOT",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/calvin_lingbot_va",
)
va_calvin_train_cfg.enable_wandb = os.getenv("ENABLE_WANDB", "0") == "1"
va_calvin_train_cfg.load_worker = int(os.getenv("LOAD_WORKER", "16"))
va_calvin_train_cfg.save_interval = int(os.getenv("SAVE_INTERVAL", "1500")) # 2500
va_calvin_train_cfg.gc_interval = int(os.getenv("GC_INTERVAL", "50"))
va_calvin_train_cfg.cfg_prob = float(os.getenv("CFG_PROB", "0.1"))

# Training parameters aligned with va_libero_train_cfg post-training defaults.
va_calvin_train_cfg.learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
va_calvin_train_cfg.beta1 = float(os.getenv("BETA1", "0.9"))
va_calvin_train_cfg.beta2 = float(os.getenv("BETA2", "0.95"))
va_calvin_train_cfg.weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-1"))
va_calvin_train_cfg.warmup_steps = int(os.getenv("WARMUP_STEPS", "300")) # 500
va_calvin_train_cfg.batch_size = int(os.getenv("BATCH_SIZE", "1"))
va_calvin_train_cfg.gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "10"))
va_calvin_train_cfg.num_steps = int(os.getenv("NUM_STEPS", "30000")) # 50000
