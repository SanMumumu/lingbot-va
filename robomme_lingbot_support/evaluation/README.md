# RoboMME Evaluation

This folder is the minimal RoboMME evaluation entrypoint for LingBot-VA.

Keep the LingBot-VA websocket server on the host in the `lingbotva` env, and run the RoboMME simulator client in Docker. Both sides write to one shared host output directory, so rollout videos, instructions, and server visualizations stay aligned.

## Files

- `prepare_ckpt.sh`: aggregate the trained transformer with the base VAE/tokenizer/text encoder for inference.
- `launch_server.sh`: start one LingBot-VA websocket server on the host.
- `launch_servers_multi.sh`: start one LingBot-VA websocket server per GPU on the host.
- `docker_run.sh`: start a RoboMME Docker shell with consistent bind mounts.
- `client.py`: run RoboMME episodes inside Docker and save rollout videos.
- `launch_clients_multi.sh`: start one Docker-side client shard per GPU.

For the 16 RoboMME tasks, multi-GPU evaluation shards `BenchmarkEnvBuilder.get_task_list()` by GPU slot. With 4 cards, each card evaluates 4 tasks; with 2 cards, each card evaluates 8 tasks.

## Host Setup

Build the RoboMME image once:

```bash
cd /path/to/lingbot-va/simulator/robomme_benchmark
docker build -t robomme:cuda12.8 .
```

On a different machine, either run from the repo root or set these paths:

```bash
export REPO_ROOT=/mnt/hwdata/wangsen/WAM/lingbot-va
export TRAINED_TRANSFORMER_DIR=$REPO_ROOT/OUTPUTS/robomme_lingbot_va/checkpoints/checkpoint_step_15000/transformer
export BASE_CKPT_DIR=$REPO_ROOT/CKPTS/lingbot-va-base
export INFER_CKPT=$REPO_ROOT/CKPTS/lingbot-va-robomme-step15000
export LINGBOT_ROBOMME_DATASET=$REPO_ROOT/DATA/robomme_lingbot
export OUT_DIR=$REPO_ROOT/OUTPUTS/robomme_eval
```

## 1. Aggregate Checkpoint

This step is required after training. Training saves only `transformer/`; inference also needs `vae/`, `tokenizer/`, and `text_encoder/` from `lingbot-va-base`, and `transformer/config.json` must use `attn_mode=torch`.

```bash
cd /path/to/lingbot-va
bash robomme_lingbot_support/evaluation/prepare_ckpt.sh
```

Override paths when needed:

```bash
TRAINED_TRANSFORMER_DIR=/path/to/checkpoint_step_15000/transformer \
BASE_CKPT_DIR=/path/to/lingbot-va-base \
INFER_CKPT=/path/to/lingbot-va-robomme-step15000 \
bash robomme_lingbot_support/evaluation/prepare_ckpt.sh
```

`launch_server.sh` uses `INFER_CKPT` as the inference checkpoint.

## 2. Start Server

In a host terminal:

```bash
conda activate lingbotva

CUDA_VISIBLE_DEVICES=0 \
PORT=29056 \
bash robomme_lingbot_support/evaluation/launch_server.sh
```

Useful overrides:

```bash
INFER_CKPT=/path/to/ckpt \
LINGBOT_ROBOMME_DATASET=/path/to/robomme_lingbot \
SAVE_ROOT=/path/to/robomme_eval/visualization \
PORT=29056 \
bash robomme_lingbot_support/evaluation/launch_server.sh
```

## 3. Start Docker Client Shell

In another host terminal:

```bash
cd ${REPO_ROOT:-/path/to/lingbot-va}
bash robomme_lingbot_support/evaluation/docker_run.sh
```

This maps:

```text
host:   $OUT_DIR
docker: /app/runs/lingbot_eval
```

It also mounts the current `client.py` and the lightweight LingBot websocket helper, so Docker uses the same evaluation code as the host repo.

If the image name differs:

```bash
IMAGE=my-robomme-image bash robomme_lingbot_support/evaluation/docker_run.sh
```

## 4. Run Evaluation

Inside Docker:

```bash
source /app/.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/lingbot_client:/app/src
cd /app/lingbot_client
```

Install missing pure-Python client deps if the image does not already have them:

```bash
pip install -q websockets msgpack opencv-python-headless imageio tqdm
```

Smoke test one task:

```bash
python -u client.py \
  --host 172.17.0.1 \
  --port 29056 \
  --env-ids PickXtimes \
  --num-episodes 50 \
  --max-steps 200 \
  --out-dir /app/runs/lingbot_eval \
  --save-video 2>&1 | tee /app/runs/lingbot_eval/pickxtimes_client.log
```

Run all RoboMME tasks:

```bash
python -u client.py \
  --host 172.17.0.1 \
  --port 29056 \
  --num-episodes 50 \
  --max-steps 1300 \
  --out-dir /app/runs/lingbot_eval \
  --save-video 2>&1 | tee /app/runs/lingbot_eval/all_tasks_client.log
```

Use `--host host.docker.internal` instead of `172.17.0.1` if your Docker setup exposes the host that way.

## 5. Multi-GPU Evaluation

Use this for the full 16-task RoboMME evaluation. Start the same number of server and client shards; slot `i` talks to `BASE_PORT + i`.

For the current 4-card run on GPUs `0,1,2,3`, start server shards on the host:

```bash
conda activate lingbotva
cd ${REPO_ROOT:-/mnt/hwdata/wangsen/WAM/lingbot-va}

GPUS=0,1,2,3 \
BASE_PORT=29056 \
bash robomme_lingbot_support/evaluation/launch_servers_multi.sh
```

Then start Docker with the normal mounted shell:

```bash
bash robomme_lingbot_support/evaluation/docker_run.sh
```

Inside Docker, start matching client shards:

```bash
source /app/.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/app/lingbot_client:/app/src
cd /app/lingbot_client

GPUS=0,1,2,3 \
HOST=172.17.0.1 \
BASE_PORT=29056 \
NUM_EPISODES=50 \
MAX_STEPS=1300 \
OUT_DIR=/app/runs/lingbot_eval \
bash launch_clients_multi.sh
```

The task split is deterministic:

```text
card slot 0 -> task_list[0::N]
card slot 1 -> task_list[1::N]
...
card slot N-1 -> task_list[N-1::N]
```

Change `GPUS` to control the shard count:

```bash
GPUS=0,1       # 2 shards, 8 tasks per shard
GPUS=0,1,2,3   # 4 shards, 4 tasks per shard
GPUS=0,1,2,3,4,5,6,7   # 8 shards, 2 tasks per shard
```

All shard summaries are merged into:

```text
$OUT_DIR/summary.json
```

## Outputs

Client rollout videos:

```text
$OUT_DIR/pickxtimes/
  pick up the blue cube and place it on the target, repeating this action five times, then press the button to stop_20260506_154933/
    ep0000_False.mp4
    instruction.txt
    metadata.json
```

Server visualizations for the same run:

```text
$OUT_DIR/visualization/pickxtimes/
  pick up the blue cube and place it on the target, repeating this action five times, then press the button to stop_20260506_154933/
    observations.mp4
    pred_video_0.mp4
    pred_video_4.mp4
    metadata.json
    latents_*.pt
    actions_*.pt
    obs_data_*.pt
```

The shared relative key is:

```text
pickxtimes/<instruction>_<timestamp>
```

Use `metadata.json` to confirm the exact instruction, episode id, success flag, and matching server visualization directory.

Task summaries are written at:

```text
$OUT_DIR/PickXtimes.json
$OUT_DIR/summary_all.json
```

## Common Issues

- `ModuleNotFoundError: wan_va`: start Docker with `docker_run.sh`; it mounts the lightweight websocket helper under `/app/lingbot_client/wan_va/...`.
- No files under `$OUT_DIR/visualization`: restart `launch_server.sh`; old server processes do not load code changes.
- Docker cannot reach server: use `--host 172.17.0.1` for bridge networking, or `--host host.docker.internal` where supported.
- To disable server-side predicted videos, start the server with `LINGBOT_ROBOMME_SAVE_PRED_VIDEO=0`.
