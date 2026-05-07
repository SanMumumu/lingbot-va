# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robomme_cfg import va_robomme_cfg


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


va_robomme_train_cfg = EasyDict(__name__="Config: VA robomme train")
va_robomme_train_cfg.update(va_robomme_cfg)

va_robomme_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_ROBOMME_DATASET",
    "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot",
)
va_robomme_train_cfg.empty_emb_path = os.path.join(va_robomme_train_cfg.dataset_path, "empty_emb.pt")

va_robomme_train_cfg.enable_wandb = os.environ.get("LINGBOT_ENABLE_WANDB", "0") == "1"
va_robomme_train_cfg.load_worker = _env_int("LINGBOT_LOAD_WORKER", 16)
va_robomme_train_cfg.save_interval = _env_int("LINGBOT_SAVE_INTERVAL", 500)
va_robomme_train_cfg.gc_interval = _env_int("LINGBOT_GC_INTERVAL", 50)
va_robomme_train_cfg.cfg_prob = _env_float("LINGBOT_CFG_PROB", 0.1)

va_robomme_train_cfg.learning_rate = _env_float("LINGBOT_LR", 1e-5)
va_robomme_train_cfg.beta1 = 0.9
va_robomme_train_cfg.beta2 = 0.95
va_robomme_train_cfg.weight_decay = _env_float("LINGBOT_WEIGHT_DECAY", 0.1)
va_robomme_train_cfg.warmup_steps = _env_int("LINGBOT_WARMUP_STEPS", 10)
va_robomme_train_cfg.batch_size = _env_int("LINGBOT_BATCH_SIZE", 1)
va_robomme_train_cfg.gradient_accumulation_steps = _env_int("LINGBOT_GRAD_ACCUM", 8)
va_robomme_train_cfg.num_steps = _env_int("LINGBOT_NUM_STEPS", 5000)
