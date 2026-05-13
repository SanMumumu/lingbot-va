# LingBot-VA 测试 CALVIN ABC-D 简明说明

## 1. 下载 HF 数据

训练数据使用 Hugging Face 上已经转成 LeRobot/parquet 格式的 CALVIN ABC-D 数据：

```text
fywang/calvin-task-ABC-D-lerobot

直接用仓库里的下载脚本：

bash DATA_MY/download_calvin.sh

默认下载到：

/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot
```



## 2. 转成 LingBot-VA 训练格式

下载后的 HF 数据还是普通 LeRobot/parquet 数据，不能直接训练 LingBot-VA。还需要生成：

- `meta/episodes.jsonl` 里的 `action_config`
- `meta/calvin_action_stats.json`
- `empty_emb.pt`
- Wan2.2 VAE 预抽取的 `latents/`

下载完成后运行：

```bash
bash script/calvin_prepare_dataset.sh
```

这个脚本默认不会再下载数据，只会处理：

```text
/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot
```


默认需要 Wan2.2 Diffusers 格式 checkpoint：

```text
/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers
```

如果路径不同：

```bash
WAN22_CKPT=/path/to/Wan2.2-TI2V-5B-Diffusers \
DATASET_DIR=/path/to/calvin_abc_d_lerobot \
bash script/calvin_prepare_dataset.sh
```

冒烟通过后再跑全量：

```bash
LATENT_GPUS=8 bash script/calvin_prepare_dataset.sh
```

## 3. 训练

默认训练配置：

```text
CONFIG_NAME=calvin_train
dataset_path=/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot
save_root=/mnt/hwdata/wangsen/WAM/lingbot-va/OUTPUTS/calvin_lingbot_va
base ckpt=/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/lingbot-va-base
batch_size=1
gradient_accumulation_steps=10
num_steps=5000
save_interval=200
```

这些训练默认值和 `va_libero_train_cfg.py` 的 LingBot-VA 后训练设置对齐。

正式训练：

```bash
ENABLE_WANDB=1 NGPU=8 bash script/calvin_train.sh
```

常用可调环境变量：

```bash
LINGBOT_CALVIN_DATASET=/path/to/calvin_abc_d_lerobot
LINGBOT_CALVIN_CKPT=/path/to/lingbot-va-base
SAVE_ROOT=/path/to/output
NUM_STEPS=5000
NGPU=8
ENABLE_WANDB=1
WANDB_PROJECT=calvin_training
```

如果训练一开始 dataloader 直接 `StopIteration`，通常说明 `latents/` 还没生成或为空，需要先完成第 2 步。

## 4. 测评

评测分成 server 和 client 两个进程。

server 加载 LingBot-VA 模型：

```bash
bash evaluation/calvin/launch_server.sh
```

client 需要官方 CALVIN 代码和官方评测数据环境。HF LeRobot 数据只用于训练，不包含完整 simulator/eval 资源。

需要准备：

```text
CALVIN_ROOT=/path/to/mees/calvin
CALVIN_DATASET_PATH=/path/to/task_ABC_D
```

启动评测：

```bash
CALVIN_ROOT=/path/to/mees/calvin \
CALVIN_DATASET_PATH=/path/to/task_ABC_D \
bash evaluation/calvin/launch_client.sh
```

小规模 debug：

```bash
CALVIN_ROOT=/path/to/mees/calvin \
CALVIN_DATASET_PATH=/path/to/task_ABC_D \
NUM_SEQUENCES=5 EP_LEN=360 \
bash evaluation/calvin/launch_client.sh --debug
```

结果默认写到：

```text
outputs/calvin
```

## 5. 一句话流程

```bash
# 1. 下载 HF LeRobot 数据
bash DATA_MY/download_calvin.sh

# 2. 转成 LingBot-VA 格式并抽 latents
bash script/calvin_prepare_dataset.sh

# 3. 训练并记录 wandb
ENABLE_WANDB=1 NGPU=8 bash script/calvin_train.sh

# 4. 开 server
bash evaluation/calvin/launch_server.sh

# 5. 开 client 做 CALVIN D 环境评测
CALVIN_ROOT=/path/to/mees/calvin \
CALVIN_DATASET_PATH=/path/to/task_ABC_D \
bash evaluation/calvin/launch_client.sh
```
