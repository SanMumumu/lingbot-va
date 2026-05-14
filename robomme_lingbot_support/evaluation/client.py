#!/usr/bin/env python3
"""LingBot-VA evaluation client for the RoboMME benchmark.

Connects to the LingBot-VA websocket server (running in the lingbotva env)
and drives a RoboMME `BenchmarkEnvBuilder` simulator (running in the
robomme env). Mirrors evaluation/libero/client.py: query an action chunk,
apply each action to the env, send key-frame obs back for KV-cache update,
repeat until terminated / truncated / info["status"] == "success".

Multi-GPU sharding: pass --num-shards N --shard-index K to evaluate the
K-th balanced or round-robin shard. Run one client per shard, each pointed
at its own server port.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
PRED_VIDEO_RE = re.compile(r"^pred_video_(\d+)\.mp4$")

# Measured from a 4-shard RoboMME run with 50 episodes per task. These are
# used only to balance multi-GPU task assignment; success metrics still come
# from the actual evaluation.
DEFAULT_TASK_WEIGHTS = {
    "BinFill": 12998,
    "ButtonUnmask": 6296,
    "ButtonUnmaskSwap": 12899,
    "InsertPeg": 15895,
    "MoveCube": 8097,
    "PatternLock": 3541,
    "PickHighlight": 11778,
    "PickXtimes": 7880,
    "RouteStick": 5322,
    "StopCube": 5401,
    "SwingXtimes": 8324,
    "VideoPlaceButton": 5945,
    "VideoPlaceOrder": 7424,
    "VideoRepick": 4612,
    "VideoUnmask": 4162,
    "VideoUnmaskSwap": 3480,
}


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
    """Single env step. Returns (next_obs_dict_or_none, done, success)."""
    action = np.asarray(action, dtype=np.float32)
    obs, _, terminated, truncated, info = env.step(action)
    if isinstance(terminated, torch.Tensor):
        terminated = bool(terminated.item())
    if isinstance(truncated, torch.Tensor):
        truncated = bool(truncated.item())
    status = info.get("status", "ongoing")
    success = status == "success"
    done = bool(terminated or truncated) or status in {"success", "fail", "timeout", "error"}
    try:
        next_obs = extract_obs(obs)
    except KeyError:
        if done:
            return None, done, success
        raise
    return next_obs, done, success


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


def resolve_shared_path(path_value: str | os.PathLike | None, out_dir: Path) -> Path | None:
    """Resolve a server-side OUTPUTS path inside Docker's mounted out_dir."""
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path

    parts = path.parts
    if "visualization" in parts:
        idx = parts.index("visualization")
        candidate = out_dir.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate
    if "robomme_eval" in parts:
        idx = parts.index("robomme_eval")
        candidate = out_dir.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate
    return path


def copy_prediction_video(
    server_video_path: str | os.PathLike | None,
    out_dir: Path,
    run_dir: Path,
) -> dict | None:
    src = resolve_shared_path(server_video_path, out_dir)
    if src is None:
        return None
    record = {
        "source": str(server_video_path),
        "resolved_source": str(src),
        "local": str(run_dir / src.name),
    }
    if not src.exists():
        record["missing"] = True
        print(f"[warn] prediction video not found: {src}", flush=True)
        return record

    dst = run_dir / src.name
    try:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[warn] failed to copy prediction video {src} -> {dst}: {exc}", flush=True)
    return record


def copy_prediction_videos_from_visualization(
    visualization_dir: str | os.PathLike | None,
    out_dir: Path,
    run_dir: Path,
    seen_names: set[str],
) -> list[dict]:
    vis_dir = resolve_shared_path(visualization_dir, out_dir)
    if vis_dir is None or not vis_dir.exists():
        return []
    records = []
    for src in sorted(vis_dir.glob("pred_video_*.mp4")):
        if src.name in seen_names:
            continue
        record = copy_prediction_video(src, out_dir, run_dir)
        if record is not None:
            seen_names.add(src.name)
            records.append(record)
    return records


def prediction_video_index(path: Path) -> int:
    match = PRED_VIDEO_RE.match(path.name)
    if not match:
        return 10**12
    return int(match.group(1))


def prediction_videos_in(run_dir: Path) -> list[Path]:
    videos = [path for path in run_dir.glob("pred_video_*.mp4") if PRED_VIDEO_RE.match(path.name)]
    return sorted(videos, key=prediction_video_index)


def ffconcat_quote(path: Path) -> str:
    text = str(path.resolve())
    return "'" + text.replace("\\", "\\\\").replace("'", "'\\''") + "'"


def concat_prediction_videos(run_dir: Path, output_name: str = "pred_video_all.mp4") -> str | None:
    videos = prediction_videos_in(run_dir)
    if not videos:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[warn] ffmpeg not found; skip pred_video_all.mp4", flush=True)
        return None

    output_path = run_dir / output_name
    tmp_output = output_path.with_name(output_path.name + ".tmp.mp4")
    list_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False, encoding="utf-8") as handle:
            list_path = Path(handle.name)
            for video in videos:
                handle.write(f"file {ffconcat_quote(video)}\n")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(tmp_output),
        ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_output, output_path)
        return str(output_path)
    except Exception as exc:
        print(f"[warn] failed to concat prediction videos in {run_dir}: {exc}", flush=True)
        return None
    finally:
        if list_path is not None:
            list_path.unlink(missing_ok=True)
        tmp_output.unlink(missing_ok=True)


def balanced_task_shards(tasks: list[str], num_shards: int) -> list[list[str]]:
    """Greedy longest-processing-time assignment using measured task weights."""
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    totals = [0 for _ in range(num_shards)]
    default_weight = max(DEFAULT_TASK_WEIGHTS.values()) if DEFAULT_TASK_WEIGHTS else 1
    for task in sorted(tasks, key=lambda t: DEFAULT_TASK_WEIGHTS.get(t, default_weight), reverse=True):
        shard_idx = min(range(num_shards), key=lambda i: totals[i])
        shards[shard_idx].append(task)
        totals[shard_idx] += DEFAULT_TASK_WEIGHTS.get(task, default_weight)
    return shards


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
    predicted_videos: list[dict] = []
    copied_pred_names: set[str] = set()

    while steps_taken < max_steps and not done:
        # Chunk 0: send conditioning frame (frame 0 if training included demo).
        # Subsequent chunks: send the latest env state (KV-cache updates use cur_obs).
        infer_obs = cond_obs if first else cur_obs
        ret = model.infer(dict(obs=infer_obs, prompt=prompt, **vis_kwargs))
        pred_record = copy_prediction_video(ret.get("video_path"), out_dir, run_dir) if isinstance(ret, dict) else None
        if pred_record is not None:
            if not pred_record.get("missing") and not pred_record.get("error"):
                copied_pred_names.add(Path(pred_record["local"]).name)
            predicted_videos.append(pred_record)
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
                next_obs, done, success = env_step(env, ee_action)
                steps_taken += 1
                if next_obs is not None:
                    cur_obs = next_obs
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

    predicted_videos.extend(
        copy_prediction_videos_from_visualization(
            reset_meta.get("visualization_dir"),
            out_dir,
            run_dir,
            copied_pred_names,
        )
    )
    prediction_video_all = concat_prediction_videos(run_dir)

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
                "prediction_videos": predicted_videos,
                "prediction_video_all": prediction_video_all,
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
        "--shard-mode",
        choices=["balanced", "round_robin"],
        default=os.environ.get("SHARD_MODE", "balanced"),
        help="Task assignment mode when --num-shards > 1. balanced uses measured RoboMME task runtimes.",
    )
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
        if args.shard_mode == "balanced":
            shards = balanced_task_shards(all_tasks, args.num_shards)
            my_tasks = shards[args.shard_index]
            estimated = sum(DEFAULT_TASK_WEIGHTS.get(t, 0) for t in my_tasks)
            print(f"[balanced] estimated shard seconds: {[sum(DEFAULT_TASK_WEIGHTS.get(t, 0) for t in s) for s in shards]}")
            print(f"[balanced] this shard estimated hours: {estimated / 3600:.2f}")
        else:
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
