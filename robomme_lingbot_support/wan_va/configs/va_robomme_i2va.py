# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robomme_cfg import va_robomme_cfg

va_robomme_i2va_cfg = EasyDict(__name__='Config: VA robomme i2va')
va_robomme_i2va_cfg.update(va_robomme_cfg)

va_robomme_i2va_cfg.input_img_path = 'example/robomme'
va_robomme_i2va_cfg.num_chunks_to_infer = 10
va_robomme_i2va_cfg.prompt = 'pick up the cube and place it on the target, then press the button to stop'
va_robomme_i2va_cfg.infer_mode = 'i2va'
