#!/usr/bin/bash

set -euo pipefail

# Quick copy-paste command collection for LingBot-VA.
# Update dataset/checkpoint paths below if your local paths are different.

# Shared checkpoint path used by the current configs.
export LINGBOT_WAN22_MODEL_PATH="/mnt/hwdata/wangsen/WAM/Ckpts/Robbyant/lingbot-va-base"

# Franka single-arm dataset used by franka_single_arm_train.
export LINGBOT_FRANKA_DATASET_PATH="/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/PickBlueonRed_lerobot"

# Optional: if you use the official Wan2.2 source tree for latent extraction.
# export PYTHONPATH="/mnt/hwdata/wangsen/WAM/Wan2.2:${PYTHONPATH:-}"


##############################
# Franka data preparation
##############################

# Step 1: convert raw Franka recordings to local LeRobot v2.1 format.
python Convert/franka/convert_to_lerobot.py \
  --input-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed \
  --output-root /mnt/hwdata/wangsen/WAM/lingbot-va/DATA/PickBlueonRed_lerobot \
  --task-text "pick blue object on red object" \
  --overwrite

# Step 2: add a single action segment per episode.
python Convert/franka/add_action_config.py \
  --dataset-root /mnt/hwdata/wangsen/WAM/lingbot-va/DATA/PickBlueonRed_lerobot \
  --action-text "pick blue object on red object" \
  --backup

# Step 3: extract Wan2.2 latents from the converted Franka dataset.
# This command expects the official Wan2.2 source tree at /mnt/hwdata/wangsen/WAM/Wan2.2.
python Convert/franka/extract_wan_latents.py \
  --dataset-root /mnt/hwdata/wangsen/WAM/lingbot-va/DATA/PickBlueonRed_lerobot \
  --wan-model-root /mnt/hwdata/wangsen/WAM/Ckpts/Wan2.2-TI2V-5B \
  --wan-backend official \
  --wan-code-root /mnt/hwdata/wangsen/WAM/Wan2.2 \
  --height 224 \
  --width 320 \
  --target-fps 10


##############################
# Online server inference
##############################

NGPU=1 CONFIG_NAME='robotwin' bash script/run_launch_va_server_sync.sh
# Run RobotWin server inference with the standard robotwin config.

NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
# Run RobotWin image-to-action-video inference.

NGPU=1 CONFIG_NAME='franka' bash script/run_launch_va_server_sync.sh
# Run Franka server inference with the standard franka config.

NGPU=1 CONFIG_NAME='franka_i2av' bash script/run_launch_va_server_sync.sh
# Run Franka image-to-action-video inference.

NGPU=1 CONFIG_NAME='demo' bash script/run_launch_va_server_sync.sh
# Run the lightweight demo server config.

NGPU=1 CONFIG_NAME='demo_i2av' bash script/run_launch_va_server_sync.sh
# Run the lightweight demo image-to-action-video config.


##############################
# Post-training
##############################

NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
# Post-train on the RobotWin dataset.
# Update wan_va/configs/va_robotwin_train_cfg.py before using this if your RobotWin dataset path is different.

NGPU=8 CONFIG_NAME='demo_train' bash script/run_va_posttrain.sh
# Post-train on the demo dataset at /mnt/hwdata/wangsen/WAM/lingbot-va/DATA/pick-n-place-sq-lerobot-v21.

NGPU=8 CONFIG_NAME='franka_single_arm_train' bash script/run_va_posttrain.sh
# Post-train on the converted Franka single-arm dataset at $LINGBOT_FRANKA_DATASET_PATH.
