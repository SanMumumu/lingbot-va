#!/usr/bin/env python3
"""LingBot-VA evaluation client for the RoboMME benchmark.

Connects to the LingBot-VA websocket server (running in the lingbotva env)
and drives a RoboMME `BenchmarkEnvBuilder` simulator (running in the
robomme env). Mirrors evaluation/libero/client.py: query an action chunk,
apply each action to the env, send key-frame obs back for KV-cache update,
repeat until terminated / truncated / info["status"] == "success".

Multi-GPU sharding: pass --num-shards N --shard-index K to evaluate the
K-th slice of `BenchmarkEnvBuilder.get_task_list()` (round-robin over
shards). Run one client per shard, each pointed at its own server port.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

# RoboMME API.
from robomme.env_record_wrapper import BenchmarkEnvBuilder  # noqa: E402

# Camera keys must match va_robomme_cfg.obs_cam_keys.
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
CAM_KEYS = [FRONT_KEY, WRIST_KEY]


# ---------------------------------------------------------------------------
# RoboMME adapter
# ---------------------------------------------------------------------------

def _to_uint8_rgb(value) -> np.ndarray:
    """Convert a RoboMME image (np or torch, possibly HWC float) to HxWx3 uint8."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def extract_obs(env_obs, frame_idx: int = -1) -> dict[str, np.ndarray]:
    """RoboMME obs (lists per-step) -> LingBot camera-keyed dict.

    frame_idx selects which entry in the cumulative *_rgb_list to return:
      -1  current/most-recent (default; correct after each env.step)
       0  first frame (= start of RoboMME video demo at env.reset);
          use this for the FIRST conditioning frame if training segments
          include the video demo (action_config[0].start_frame == 0).
    """
    front = _to_uint8_rgb(env_obs["front_rgb_list"][frame_idx])
    wrist = _to_uint8_rgb(env_obs["wrist_rgb_list"][frame_idx])
    return {FRONT_KEY: front, WRIST_KEY: wrist}


def get_task_prompt(info, fallback: str) -> str:
    goal = info.get("task_goal") if isinstance(info, dict) else None
    if goal is None:
        return fallback
    if isinstance(goal, (list, tuple)) and goal:
        goal = goal[0]
    if isinstance(goal, bytes):
        goal = goal.decode("utf-8", errors="ignore")
    return str(goal).lower() if goal else fallback


def env_step(env, action: np.ndarray):
    """Single env step. Returns (next_obs_dict, terminated, success)."""
    action = np.asarray(action, dtype=np.float32)
    obs, _, terminated, truncated, info = env.step(action)
    if isinstance(terminated, torch.Tensor):
        terminated = bool(terminated.item())
    if isinstance(truncated, torch.Tensor):
        truncated = bool(truncated.item())
    status = info.get("status", "ongoing")
    success = status == "success"
    done = bool(terminated or truncated) or status in {"success", "fail", "timeout", "error"}
    return extract_obs(obs), done, success


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def save_video(frames: list[dict], path: Path, fps: int = 15) -> None:
    if not frames:
        return
    h, w = frames[0][FRONT_KEY].shape[:2]
    composed = [
        np.hstack([cv2.resize(f[k], (w, h)) for k in CAM_KEYS]).astype(np.uint8)
        for f in frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, composed, fps=fps)


def safe_path_part(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "default"


def run_one_episode(
    model: WebsocketClientPolicy,
    env_id: str,
    episode_idx: int,
    action_space: str,
    max_steps: int,
    out_dir: Path,
    save_video_flag: bool,
    first_frame_conditioning: bool = False,
) -> bool:
    print(f"  [{env_id}] BenchmarkEnvBuilder(...)", flush=True)
    builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="test",
        action_space=action_space,
        gui_render=False,
        max_steps=max_steps,
    )
    print(f"  [{env_id}] make_env_for_episode({episode_idx})", flush=True)
    env = builder.make_env_for_episode(episode_idx)
    print(f"  [{env_id}] env.reset()", flush=True)
    raw_obs, info = env.reset()
    print(f"  [{env_id}] env ready; obs keys={list(raw_obs.keys())[:5]}...", flush=True)
    cur_obs = extract_obs(raw_obs)
    # The conditioning frame for chunk 0 must match training. If training
    # segments started at H5 frame 0 (i.e. action_config[0].start_frame == 0,
    # video demo included), use raw_obs[*_rgb_list][0]. Otherwise the
    # post-demo frame raw_obs[*_rgb_list][-1] is correct.
    cond_obs = extract_obs(raw_obs, frame_idx=0) if first_frame_conditioning else cur_obs
    prompt = get_task_prompt(info, fallback=env_id)

    vis_kwargs = {
        "env_id": env_id,
        "task_name": env_id,
        "save_visualization": save_video_flag,
    }
    reset_meta = model.infer(dict(reset=True, prompt=prompt, **vis_kwargs))
    if not isinstance(reset_meta, dict):
        reset_meta = {}
    task_slug = reset_meta.get("task_slug") or safe_path_part(env_id)
    run_name = reset_meta.get("run_name") or f"{safe_path_part(prompt)}_ep{episode_idx:04d}"
    run_dir = out_dir / task_slug / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")

    full_obs_list: list[dict] = [cur_obs]
    success = False
    done = False
    first = True
    steps_taken = 0

    while steps_taken < max_steps and not done:
        # Chunk 0: send conditioning frame (frame 0 if training included demo).
        # Subsequent chunks: send the latest env state (KV-cache updates use cur_obs).
        infer_obs = cond_obs if first else cur_obs
        ret = model.infer(dict(obs=infer_obs, prompt=prompt, **vis_kwargs))
        action = ret["action"]                 # [D, frame_chunk_size, action_per_frame]
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)

        assert action.shape[2] % 4 == 0, f"unexpected action shape {action.shape}"
        action_per_keyframe = action.shape[2] // 4   # frame_chunk_size hardcoded to 4
        start_idx = 1 if first else 0

        key_frame_list: list[dict] = []
        chunk_done = False
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                cur_obs, done, success = env_step(env, ee_action)
                steps_taken += 1
                full_obs_list.append(cur_obs)
                if done or steps_taken >= max_steps:
                    chunk_done = True
                    break
                if (j + 1) % action_per_keyframe == 0:
                    key_frame_list.append(cur_obs)
            if chunk_done:
                break

        first = False
        if chunk_done:
            break

        # Update KV cache with applied actions + observed key frames.
        model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action, **vis_kwargs))

    if save_video_flag:
        out_file = run_dir / f"ep{episode_idx:04d}_{success}.mp4"
        save_video(full_obs_list, out_file)

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": env_id,
                "task_slug": task_slug,
                "episode_idx": episode_idx,
                "success": success,
                "steps_taken": steps_taken,
                "instruction": prompt,
                "action_space": action_space,
                "max_steps": max_steps,
                "client_video": str(run_dir / f"ep{episode_idx:04d}_{success}.mp4") if save_video_flag else None,
                "server_visualization_dir": reset_meta.get("visualization_dir"),
                "server_run_name": reset_meta.get("run_name"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        env.close()
    except Exception:
        pass
    return success


def run(
    host: str,
    port: int,
    env_ids: list[str],
    episodes: list[int],
    action_space: str,
    max_steps: int,
    out_dir: Path,
    save_video_flag: bool,
    shard_tag: str,
    first_frame_conditioning: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{shard_tag}] opening WebsocketClientPolicy(host={host}, port={port})", flush=True)
    model = WebsocketClientPolicy(host=host, port=port)
    print(f"[{shard_tag}] connected to server, metadata={model.get_server_metadata()}", flush=True)

    summary: dict[str, dict] = {}
    for env_id in env_ids:
        succ = 0
        for ep_idx, episode in enumerate(tqdm(episodes, desc=f"[{shard_tag}] {env_id}")):
            try:
                ok = run_one_episode(
                    model=model,
                    env_id=env_id,
                    episode_idx=episode,
                    action_space=action_space,
                    max_steps=max_steps,
                    out_dir=out_dir,
                    save_video_flag=save_video_flag,
                    first_frame_conditioning=first_frame_conditioning,
                )
            except Exception as exc:
                print(f"[{shard_tag}][{env_id}] episode {episode} crashed: "
                      f"{type(exc).__name__}: {exc}")
                ok = False
            succ += int(ok)
            sr = succ / (ep_idx + 1)
            summary[env_id] = {
                "succ_num": succ,
                "total_num": ep_idx + 1,
                "succ_rate": sr,
                "episodes": episodes[: ep_idx + 1],
            }
            (out_dir / f"{env_id}.json").write_text(
                json.dumps(summary[env_id], indent=2), encoding="utf-8"
            )
            print(f"[{shard_tag}][{env_id}] succ_rate={sr:.3f} ({succ}/{ep_idx + 1})")

    avg_sr = float(np.mean([s["succ_rate"] for s in summary.values()])) if summary else 0.0
    overall = {"per_env": summary, "average_success_rate": avg_sr}
    (out_dir / f"summary_{shard_tag}.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(f"\n[{shard_tag}] Average success rate: {avg_sr:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LINGBOT_SERVER_HOST", "127.0.0.1"),
                        help="Server host. Use 172.17.0.1 (or host.docker.internal) when running inside a bridge-network Docker.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LINGBOT_SERVER_PORT", "29056")))
    parser.add_argument("--env-ids", nargs="*", default=None,
                        help="Subset of RoboMME tasks. Defaults to BenchmarkEnvBuilder.get_task_list().")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Episode indices to run per task. Defaults to range(--num-episodes).")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "50")),
                        help="Number of episodes per task when --episodes is unset.")
    parser.add_argument("--action-space", default=os.environ.get("ACTION_SPACE", "ee_pose"),
                        choices=["ee_pose", "joint_angle", "waypoint"],
                        help="Must match the action_key used during training.")
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "1300")))
    parser.add_argument("--out-dir", type=Path,
                        default=Path(os.environ.get("OUT_DIR", "/app/runs/lingbot_eval")))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--first-frame-conditioning",
        action="store_true",
        help="Use raw_obs[*_rgb_list][0] (start of RoboMME video demo) as the chunk-0 "
             "conditioning frame instead of [-1] (post-demo). Set this if the converter "
             "ran with action_config[0].start_frame == 0 (training segment included demo).",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    all_tasks = args.env_ids or BenchmarkEnvBuilder.get_task_list()
    if args.num_shards > 1:
        if not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}")
        my_tasks = all_tasks[args.shard_index :: args.num_shards]
        shard_tag = f"shard{args.shard_index}of{args.num_shards}"
    else:
        my_tasks = all_tasks
        shard_tag = "all"

    episodes = list(args.episodes) if args.episodes is not None else list(range(args.num_episodes))

    print(f"[{shard_tag}] tasks: {my_tasks}")
    print(f"[{shard_tag}] episodes per task: {episodes}")
    print(f"[{shard_tag}] connecting to ws://{args.host}:{args.port}")

    run(
        host=args.host,
        port=args.port,
        env_ids=my_tasks,
        episodes=episodes,
        action_space=args.action_space,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        save_video_flag=args.save_video,
        shard_tag=shard_tag,
        first_frame_conditioning=args.first_frame_conditioning,
    )


if __name__ == "__main__":
    main()
