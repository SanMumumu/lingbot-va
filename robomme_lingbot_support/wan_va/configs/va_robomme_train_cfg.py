# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robomme_cfg import va_robomme_cfg
import os

va_robomme_train_cfg = EasyDict(__name__='Config: VA robomme train')
va_robomme_train_cfg.update(va_robomme_cfg)

va_robomme_train_cfg.dataset_path = os.environ.get(
    'LINGBOT_ROBOMME_DATASET',
    '/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot',
)
va_robomme_train_cfg.empty_emb_path = os.path.join(va_robomme_train_cfg.dataset_path, 'empty_emb.pt')
va_robomme_train_cfg.enable_wandb = True
va_robomme_train_cfg.load_worker = 16
va_robomme_train_cfg.save_interval = 500
va_robomme_train_cfg.gc_interval = 50
va_robomme_train_cfg.cfg_prob = 0.1

# Training parameters
va_robomme_train_cfg.learning_rate = 1e-5
va_robomme_train_cfg.beta1 = 0.9
va_robomme_train_cfg.beta2 = 0.95
va_robomme_train_cfg.weight_decay = 1e-1
va_robomme_train_cfg.warmup_steps = 10
va_robomme_train_cfg.batch_size = 1
va_robomme_train_cfg.gradient_accumulation_steps = 10
va_robomme_train_cfg.num_steps = 15000
