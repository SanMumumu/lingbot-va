# Franka Real-World Data Conversion for LingBot-VA

这个目录补齐了 README 里真机数据准备缺失的 3 个步骤，针对你现在这类单臂 Franka 录制数据：

```text
recordings/
  episode_00000/
    image/
      left_camera/
      right_camera/
      wrist_camera/
    states.npz
```

`states.npz` 里默认读取这些字段：

- `robot/joint_positions`: `[T, 7]`
- `robot/joint_velocities`: `[T, 7]`，如果没有会自动补 0
- `robot/ee_pose`: `[T, 7]`
- `robot/robot_timestamp`: `[T]`，如果没有会按 `--fallback-fps` 生成
- `gripper/width`: `[T]`

## 1. 目录说明

- `Convert/franka/convert_to_lerobot.py`
  - Step 1，把原始录制转成本地 LeRobot v2.1 数据集。
- `Convert/franka/add_action_config.py`
  - Step 2，给 `meta/episodes.jsonl` 增加 `action_config`。
- `Convert/franka/extract_wan_latents.py`
  - Step 3，按 `action_config` 分段抽帧并提取 Wan2.2 latent。
- `wan_va/configs/va_franka_single_arm_train_cfg.py`
  - 新增的单臂 Franka 训练配置，自动读取转换后数据集里的相机键、动作统计和 latent 分辨率。

## 2. 默认字段映射

### 2.1 相机映射

默认把原始三路相机映射成：

- `left_camera -> observation.images.left_camera`
- `right_camera -> observation.images.right_camera`
- `wrist_camera -> observation.images.wrist_camera`

如果你后面想改名字，可以在 Step 1 里用 `--camera-mapping` 覆盖。

### 2.2 observation.state

默认保存成：

```text
[joint_positions(7), ee_pose(7), gripper_width(1)]
```

总共 15 维。

### 2.3 action

默认保存成单臂 compact action：

```text
[ee_pose(t+1), joint_positions(t+1), gripper_width(t+1)]
```

也是 15 维，默认 `--action-offset 1`，也就是用下一时刻状态当监督目标。

训练时这 15 维会自动对齐到 LingBot-VA 的标准 30 维动作布局里，使用的是左臂槽位：

- `0:7` -> left arm eef
- `14:21` -> left arm joints
- `28` -> left gripper

其余维度自动补 0。

## 3. Step 1: Convert to LeRobot

先进入仓库根目录，然后执行：

```bash
python Convert/franka/convert_to_lerobot.py \
  --input-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed \
  --output-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot \
  --task-text "pick blue object on red object" \
  --overwrite
```

输出目录会生成：

```text
2025-10-01_PickBlueonRed_lerobot/
  data/
  videos/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    stats.json
    action_norm_stats.json
    franka_meta.json
```

其中：

- `action_norm_stats.json` 会保存对齐到 30 维后的 `q01/q99`，训练配置会直接读取。
- `franka_meta.json` 会保存相机键和动作槽位映射，训练配置也会直接读取。

如果你想改相机键，可以这样：

```bash
python Convert/franka/convert_to_lerobot.py \
  --input-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed \
  --output-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot \
  --camera-mapping \
    left_camera=observation.images.cam_high \
    right_camera=observation.images.cam_left_wrist \
    wrist_camera=observation.images.cam_right_wrist \
  --overwrite
```

## 4. Step 2: Add action_config

### 4.1 单段动作，整段一句话

如果每条 episode 就是一段完整动作，直接执行：

```bash
python Convert/franka/add_action_config.py \
  --dataset-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot \
  --action-text "pick blue object on red object" \
  --backup
```

这样每个 episode 都会变成：

```json
{
  "episode_index": 0,
  "tasks": ["pick blue object on red object"],
  "length": 224,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 224,
      "action_text": "pick blue object on red object"
    }
  ]
}
```

### 4.2 多段动作

如果一条 episode 里要切成多段，就自己准备一个 JSON，比如 `segments.json`：

```json
{
  "default": [
    {"start_frame": 0, "end_frame": 80, "action_text": "reach to the blue object"},
    {"start_frame": 80, "end_frame": 150, "action_text": "grasp the blue object"},
    {"start_frame": 150, "end_frame": "length", "action_text": "place the blue object on the red object"}
  ]
}
```

然后执行：

```bash
python Convert/franka/add_action_config.py \
  --dataset-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot \
  --segments-json /path/to/segments.json \
  --backup
```

也支持对某一条 episode 单独覆盖：

```json
{
  "default": [
    {"start_frame": 0, "end_frame": "length", "action_text": "pick blue object on red object"}
  ],
  "1": [
    {"start_frame": 0, "end_frame": 120, "action_text": "reach and grasp the blue object"},
    {"start_frame": 120, "end_frame": "length", "action_text": "place it on the red object"}
  ]
}
```

## 5. Step 3: Extract Wan2.2 latents

### 5.1 先说明 Wan backend

`extract_wan_latents.py` 支持两种模式：

- `--wan-backend official`
  - 对应你现在给的这种原始 Wan2.2 checkpoint 布局，默认就是这个。
- `--wan-backend diffusers`
  - 只适用于已经整理成 `vae/`, `text_encoder/`, `tokenizer/` 子目录的 diffusers 格式模型。

### 5.2 用你当前给的 Wan 路径

你给的路径是：

```text
/mnt/hwdata/wangsen/WAM/Ckpts/Wan2.2-TI2V-5B
```

如果这个目录本身就能 `import wan`，可以直接这样跑：

```bash
python Convert/franka/extract_wan_latents.py \
  --dataset-root /mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot \
  --wan-model-root /mnt/hwdata/wangsen/WAM/Ckpts/Wan2.2-TI2V-5B \
  --wan-backend official \
  --wan-code-root /mnt/hwdata/wangsen/WAM/Ckpts/Wan2.2-TI2V-5B \
  --height 224 \
  --width 320 \
  --target-fps 10
```

如果 checkpoint 和 Wan2.2 源码目录不是同一个位置，就把 `--wan-code-root` 改成真正包含 `wan/` Python 包的源码根目录。

脚本会额外生成：

- `latents/chunk-xxx/<camera_key>/episode_xxxxxx_start_end.pth`
- `empty_emb.pt`
- `meta/latent_config.json`

## 6. 训练

新增的训练配置名是：

```text
franka_single_arm_train
```

这个配置会自动从你的数据集里读取：

- `meta/franka_meta.json`
- `meta/action_norm_stats.json`
- `meta/latent_config.json`

所以训练前只需要把路径环境变量设对。

### 6.1 最小训练命令

```bash
export LINGBOT_FRANKA_DATASET_PATH=/mnt/hwdata/wangsen/Real_World/Data/2025-10-01_PickBlueonRed_lerobot
export LINGBOT_WAN22_MODEL_PATH=/mnt/hwdata/wangsen/WAM/Ckpts/Wan2.2-TI2V-5B

NGPU=8 CONFIG_NAME=franka_single_arm_train bash script/run_va_posttrain.sh
```

### 6.2 常用可调环境变量

```bash
export LINGBOT_FRANKA_BATCH_SIZE=1
export LINGBOT_FRANKA_GRAD_ACCUM=8
export LINGBOT_FRANKA_NUM_STEPS=20000
export LINGBOT_FRANKA_LR=1e-5
export LINGBOT_FRANKA_LOAD_WORKER=8
```

## 7. 依赖建议

建议保证这些依赖可用：

```bash
pip install pyarrow imageio[ffmpeg] pillow tqdm lerobot==0.3.3 diffusers==0.36.0 transformers==4.55.2
```

如果你用 `official` backend，还需要本地 Wan2.2 源码能被 Python 导入。

## 8. 现在这套脚本默认假设

- 你是单臂 Franka。
- 训练动作使用 `ee_pose + joint_positions + gripper_width`。
- `action[t]` 默认取 `t+1` 时刻目标。
- `episodes.jsonl` 默认按整段单动作处理，除非你提供 `segments.json`。
- latent 提取默认把视频 resize 到 `224x320`、采样到 `10 fps`。

如果你后面要改成：

- 只用 `ee_pose + gripper`
- action 改成 delta 而不是 next-state
- 改相机布局或分辨率
- 按语言模板批量生成更细的 action_text

可以在这套脚本上继续改，不需要再重写整条链路。
