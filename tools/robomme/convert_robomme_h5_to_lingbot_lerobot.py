#!/usr/bin/env python3
"""Parallel RoboMME HDF5 -> LingBot-VA LeRobot v2.1 converter.

RoboMME H5 format, per official docs:
  episode_<K>/
    setup/task_goal
    timestep_<K>/obs/front_rgb, obs/wrist_rgb
    timestep_<K>/obs/joint_state, obs/eef_state, obs/gripper_state
    timestep_<K>/action/eef_action or joint_action
    timestep_<K>/info/is_video_demo, is_subgoal_boundary, *_subgoal*_online

This converter writes the minimal LeRobot v2.1 structure LingBot-VA uses:
  data/chunk-xxx/episode_xxxxxx.parquet
  videos/chunk-xxx/observation.images.front/episode_xxxxxx.mp4
  videos/chunk-xxx/observation.images.wrist/episode_xxxxxx.mp4
  meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
  meta/robomme_action_stats.json

Each episode is processed independently, so conversion can be parallelized
without sharing mutable LeRobotDataset state across workers.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_RAW_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomee"
DEFAULT_OUT_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot"

FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_CHUNKS_SIZE = 100
STANDARD_ACTION_DIM = 30
EEF_CHANNELS = list(range(0, 7))
LEFT_GRIPPER_CHANNEL = 28
RIGHT_GRIPPER_CHANNEL = 29


@dataclass(frozen=True)
class EpisodeJob:
    source_h5: str
    source_episode_key: str
    episode_index: int
    frame_start_index: int
    task: str
    task_index: int
    length: int
    action_config: list[dict[str, Any]]
    output_dir: str
    fps: int
    image_size: int
    action_key: str
    state_dim: int
    video_codec: str
    chunks_size: int


@dataclass
class EpisodeResult:
    ok: bool
    episode_index: int
    source_h5: str
    source_episode_key: str
    length: int = 0
    selected_actions: list[list[float]] | None = None
    episode_stats: dict[str, Any] | None = None
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def read_text(dataset: h5py.Dataset | None, default: str = "") -> str:
    if dataset is None:
        return default
    try:
        text = decode_text(dataset[()])
        return text if text else default
    except Exception:
        return default


def read_bool(dataset: h5py.Dataset | None, default: bool = False) -> bool:
    if dataset is None:
        return default
    try:
        return bool(np.asarray(dataset[()]).item())
    except Exception:
        return default


def sorted_indexed_keys(group: h5py.Group, prefix: str) -> list[str]:
    def idx(name: str) -> int:
        return int(name.split("_")[-1])

    return sorted([k for k in group.keys() if k.startswith(prefix)], key=idx)


def get_task_goal(episode: h5py.Group) -> str:
    setup = episode.get("setup")
    if setup is None:
        return "robomme manipulation task"
    return read_text(setup.get("task_goal"), "robomme manipulation task").lower()


def first_execution_step(episode: h5py.Group, timestep_keys: list[str]) -> int:
    for local_idx, ts_key in enumerate(timestep_keys):
        info = episode[ts_key].get("info")
        if info is None or not read_bool(info.get("is_video_demo"), False):
            return local_idx
    return len(timestep_keys)


def read_subgoal_text(ts: h5py.Group, fallback: str) -> str:
    info = ts.get("info")
    if info is None:
        return fallback
    for key in (
        "grounded_subgoal_online",
        "simple_subgoal_online",
        "grounded_subgoal",
        "simple_subgoal",
    ):
        text = read_text(info.get(key), "")
        if text and "complete" not in text.lower():
            return text.lower()
    return fallback


def build_action_config(
    episode: h5py.Group,
    timestep_keys: list[str],
    mode: str,
    min_segment_frames: int,
) -> list[dict[str, Any]]:
    length = len(timestep_keys)
    task = get_task_goal(episode)
    exec_start = first_execution_step(episode, timestep_keys)
    start = 0 if mode == "full" else exec_start

    if start >= length or length - start < min_segment_frames:
        return []

    if mode in {"full", "exec"}:
        return [{"start_frame": start, "end_frame": length, "action_text": task}]

    boundaries = [start]
    for local_idx in range(start + 1, length):
        info = episode[timestep_keys[local_idx]].get("info")
        if info is not None and read_bool(info.get("is_subgoal_boundary"), False):
            boundaries.append(local_idx)
    boundaries.append(length)
    boundaries = sorted(set(boundaries))

    segments = []
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        if seg_end - seg_start < min_segment_frames:
            continue
        text = read_subgoal_text(episode[timestep_keys[seg_start]], task)
        segments.append({"start_frame": seg_start, "end_frame": seg_end, "action_text": text})
    return segments or [{"start_frame": start, "end_frame": length, "action_text": task}]


def read_raw_action(ts: h5py.Group, action_key: str) -> np.ndarray:
    action_group = ts.get("action")
    if action_group is None:
        raise StageError("read_action", "missing timestep/action group")

    candidates = [action_key]
    for key in ("eef_action", "joint_action", "waypoint_action"):
        if key not in candidates:
            candidates.append(key)

    for key in candidates:
        if key in action_group:
            action = np.asarray(action_group[key][()], dtype=np.float32).reshape(-1)
            if np.all(np.isfinite(action)):
                return action
            raise StageError("read_action", f"{key} contains NaN/Inf")
    raise StageError(
        "read_action",
        f"no usable action key under action group; found={list(action_group.keys())}, tried={candidates}",
    )


def map_action_to_standard_30(raw_action: np.ndarray, action_key: str) -> np.ndarray:
    """Map RoboMME single-arm action to LingBot's 30-D standard action layout.

    Standard layout:
      0:7   left arm EEF
      7:14  right arm EEF
      14:21 left arm joints
      21:28 right arm joints
      28    left gripper
      29    right gripper

    RoboMME eef_action is [x, y, z, roll, pitch, yaw, gripper], so its
    6-DoF pose is placed at channels 0:6 and gripper at channel 28.
    Missing single-arm / quaternion / right-arm channels stay zero.
    RoboMME joint_action is 7 joints + gripper, so it maps to 14:21 and 28.
    """
    out = np.zeros((STANDARD_ACTION_DIM,), dtype=np.float32)
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action_key in {"eef_action", "waypoint_action"}:
        if raw_action.shape[0] != 7:
            raise StageError("map_action", f"{action_key} expected dim 7, got {raw_action.shape[0]}")
        out[0:6] = raw_action[0:6]
        out[LEFT_GRIPPER_CHANNEL] = raw_action[6]
    elif action_key == "joint_action":
        if raw_action.shape[0] != 8:
            raise StageError("map_action", f"joint_action expected dim 8, got {raw_action.shape[0]}")
        out[14:21] = raw_action[0:7]
        out[LEFT_GRIPPER_CHANNEL] = raw_action[7]
    else:
        raise StageError("map_action", f"unsupported action_key={action_key}")
    return out


def read_state(ts: h5py.Group) -> np.ndarray:
    obs = ts.get("obs")
    if obs is None:
        raise StageError("read_state", "missing timestep/obs group")
    pieces = []
    for key in ("joint_state", "eef_state", "gripper_state"):
        if key in obs:
            pieces.append(np.asarray(obs[key][()], dtype=np.float32).reshape(-1))
    if not pieces:
        raise StageError("read_state", "missing joint_state/eef_state/gripper_state")
    state = np.concatenate(pieces, axis=0).astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise StageError("read_state", "state contains NaN/Inf")
    return state


def read_rgb(ts: h5py.Group, key: str, image_size: int) -> np.ndarray:
    obs = ts.get("obs")
    if obs is None or key not in obs:
        raise StageError("read_rgb", f"missing obs/{key}")
    image = np.asarray(obs[key][()])
    if image.shape != (image_size, image_size, 3):
        raise StageError("read_rgb", f"expected {key} shape {(image_size, image_size, 3)}, got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def fixed_size_list_array(values: np.ndarray, dtype: pa.DataType, width: int) -> pa.FixedSizeListArray:
    flat = values.reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=dtype), width)


def write_parquet(path: Path, episode_index: int, frame_start: int, task_index: int, fps: int, states: np.ndarray, actions: np.ndarray) -> None:
    length = actions.shape[0]
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_arrays(
        [
            fixed_size_list_array(states.astype(np.float32), pa.float32(), states.shape[1]),
            fixed_size_list_array(actions.astype(np.float32), pa.float32(), STANDARD_ACTION_DIM),
            pa.array(np.arange(length, dtype=np.float32) / float(fps), type=pa.float32()),
            pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
            pa.array(np.arange(frame_start, frame_start + length, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
        ],
        names=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )
    pq.write_table(table, tmp)
    tmp.replace(path)


def open_video_writer(path: Path, fps: int, width: int, height: int, codec: str) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream, Path]:
    tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    av.logging.set_level(av.logging.ERROR)
    output = av.open(str(tmp), "w", format="mp4")
    try:
        stream = output.add_stream(codec, rate=fps)
    except Exception:
        output.close()
        if tmp.exists():
            tmp.unlink()
        raise
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return output, stream, tmp


def encode_video_frame(container, stream, rgb: np.ndarray) -> None:
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def close_video_writer(container, stream, tmp: Path, final_path: Path) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    tmp.replace(final_path)


def stats_from_array(array: np.ndarray) -> dict[str, list[Any]]:
    axis = 0
    keepdims = array.ndim == 1
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": [int(len(array))],
    }


def stats_from_video_sums(count_frames: int, pixel_count: int, vmin: np.ndarray, vmax: np.ndarray, vsum: np.ndarray, vsumsq: np.ndarray) -> dict[str, list[Any]]:
    denom = float(pixel_count)
    mean = vsum / denom
    var = np.maximum(vsumsq / denom - mean * mean, 0.0)
    return {
        "min": vmin.reshape(3, 1, 1).tolist(),
        "max": vmax.reshape(3, 1, 1).tolist(),
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": np.sqrt(var).reshape(3, 1, 1).tolist(),
        "count": [int(count_frames)],
    }


def update_video_stats(rgb: np.ndarray, acc: dict[str, np.ndarray]) -> None:
    arr = rgb.astype(np.float64) / 255.0
    flat = arr.reshape(-1, 3)
    acc["min"] = np.minimum(acc["min"], flat.min(axis=0))
    acc["max"] = np.maximum(acc["max"], flat.max(axis=0))
    acc["sum"] += flat.sum(axis=0)
    acc["sumsq"] += (flat * flat).sum(axis=0)
    acc["pixels"] += flat.shape[0]
    acc["frames"] += 1


def empty_video_acc() -> dict[str, Any]:
    return {
        "min": np.full((3,), np.inf, dtype=np.float64),
        "max": np.full((3,), -np.inf, dtype=np.float64),
        "sum": np.zeros((3,), dtype=np.float64),
        "sumsq": np.zeros((3,), dtype=np.float64),
        "pixels": 0,
        "frames": 0,
    }


def chunk_id(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def process_episode(job: EpisodeJob) -> EpisodeResult:
    front_writer = None
    wrist_writer = None
    tmp_paths: list[Path] = []
    try:
        out_dir = Path(job.output_dir)
        chunk = chunk_id(job.episode_index, job.chunks_size)
        front_video = out_dir / "videos" / f"chunk-{chunk:03d}" / FRONT_KEY / f"episode_{job.episode_index:06d}.mp4"
        wrist_video = out_dir / "videos" / f"chunk-{chunk:03d}" / WRIST_KEY / f"episode_{job.episode_index:06d}.mp4"
        parquet_path = out_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{job.episode_index:06d}.parquet"

        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        front_acc = empty_video_acc()
        wrist_acc = empty_video_acc()

        front_container, front_stream, front_tmp = open_video_writer(front_video, job.fps, job.image_size, job.image_size, job.video_codec)
        wrist_container, wrist_stream, wrist_tmp = open_video_writer(wrist_video, job.fps, job.image_size, job.image_size, job.video_codec)
        front_writer = (front_container, front_stream)
        wrist_writer = (wrist_container, wrist_stream)
        tmp_paths.extend([front_tmp, wrist_tmp])

        with h5py.File(job.source_h5, "r") as data:
            episode = data[job.source_episode_key]
            timestep_keys = sorted_indexed_keys(episode, "timestep_")
            if len(timestep_keys) != job.length:
                raise StageError("scan_consistency", f"expected {job.length} timesteps, found {len(timestep_keys)}")

            for ts_key in timestep_keys:
                ts = episode[ts_key]
                raw_action = read_raw_action(ts, job.action_key)
                action = map_action_to_standard_30(raw_action, job.action_key)
                state = read_state(ts)
                if action.shape[0] != STANDARD_ACTION_DIM:
                    raise StageError("map_action", f"mapped action dim is {action.shape[0]}, expected {STANDARD_ACTION_DIM}")
                if state.shape[0] != job.state_dim:
                    raise StageError("read_state", f"state dim changed from {job.state_dim} to {state.shape[0]}")

                front = read_rgb(ts, "front_rgb", job.image_size)
                wrist = read_rgb(ts, "wrist_rgb", job.image_size)

                encode_video_frame(front_container, front_stream, front)
                encode_video_frame(wrist_container, wrist_stream, wrist)
                update_video_stats(front, front_acc)
                update_video_stats(wrist, wrist_acc)
                states.append(state)
                actions.append(action)

        close_video_writer(front_container, front_stream, front_tmp, front_video)
        close_video_writer(wrist_container, wrist_stream, wrist_tmp, wrist_video)
        front_writer = None
        wrist_writer = None

        states_np = np.stack(states, axis=0).astype(np.float32)
        actions_np = np.stack(actions, axis=0).astype(np.float32)
        write_parquet(parquet_path, job.episode_index, job.frame_start_index, job.task_index, job.fps, states_np, actions_np)

        train_mask = np.zeros((job.length,), dtype=bool)
        for segment in job.action_config:
            train_mask[int(segment["start_frame"]) : int(segment["end_frame"])] = True
        selected_actions = actions_np[train_mask].tolist()

        episode_stats = {
            FRONT_KEY: stats_from_video_sums(front_acc["frames"], front_acc["pixels"], front_acc["min"], front_acc["max"], front_acc["sum"], front_acc["sumsq"]),
            WRIST_KEY: stats_from_video_sums(wrist_acc["frames"], wrist_acc["pixels"], wrist_acc["min"], wrist_acc["max"], wrist_acc["sum"], wrist_acc["sumsq"]),
            STATE_KEY: stats_from_array(states_np),
            ACTION_KEY: stats_from_array(actions_np),
            "timestamp": stats_from_array(np.arange(job.length, dtype=np.float32) / float(job.fps)),
            "frame_index": stats_from_array(np.arange(job.length, dtype=np.int64)),
            "episode_index": stats_from_array(np.full(job.length, job.episode_index, dtype=np.int64)),
            "index": stats_from_array(np.arange(job.frame_start_index, job.frame_start_index + job.length, dtype=np.int64)),
            "task_index": stats_from_array(np.full(job.length, job.task_index, dtype=np.int64)),
        }

        return EpisodeResult(
            ok=True,
            episode_index=job.episode_index,
            source_h5=job.source_h5,
            source_episode_key=job.source_episode_key,
            length=job.length,
            selected_actions=selected_actions,
            episode_stats=episode_stats,
        )

    except Exception as exc:
        if front_writer is not None:
            try:
                front_writer[0].close()
            except Exception:
                pass
        if wrist_writer is not None:
            try:
                wrist_writer[0].close()
            except Exception:
                pass
        for tmp in tmp_paths:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        stage = exc.stage if isinstance(exc, StageError) else "process_episode"
        return EpisodeResult(
            ok=False,
            episode_index=job.episode_index,
            source_h5=job.source_h5,
            source_episode_key=job.source_episode_key,
            error_stage=stage,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )


def inspect_h5_files(raw_dir: Path, action_key: str, segment_mode: str, min_segment_frames: int, image_size: int, skip_bad_files: bool) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    h5_files = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found under {raw_dir}")

    episode_specs = []
    task_to_index: dict[str, int] = {}
    action_dim: int | None = None
    state_dim: int | None = None
    bad_files = []

    h5_iter = tqdm(h5_files, desc="inspect h5 files", unit="file")
    for h5_path in h5_iter:
        h5_iter.set_postfix_str(h5_path.name)
        try:
            with h5py.File(h5_path, "r") as data:
                ep_keys = sorted_indexed_keys(data, "episode_")
                if not ep_keys:
                    raise StageError("inspect_h5", "no episode_* groups")

                ep_iter = tqdm(
                    ep_keys,
                    desc=f"inspect {h5_path.name}",
                    unit="episode",
                    leave=False,
                )
                for ep_key in ep_iter:
                    ep_iter.set_postfix_str(ep_key)
                    episode = data[ep_key]
                    ts_keys = sorted_indexed_keys(episode, "timestep_")
                    if not ts_keys:
                        raise StageError("inspect_h5", f"{ep_key} has no timestep_* groups")
                    task = get_task_goal(episode)
                    if task not in task_to_index:
                        task_to_index[task] = len(task_to_index)

                    first_ts = episode[ts_keys[0]]
                    cur_action_dim = map_action_to_standard_30(read_raw_action(first_ts, action_key), action_key).shape[0]
                    cur_state_dim = read_state(first_ts).shape[0]
                    _ = read_rgb(first_ts, "front_rgb", image_size)
                    _ = read_rgb(first_ts, "wrist_rgb", image_size)

                    if action_dim is None:
                        action_dim = cur_action_dim
                    elif action_dim != cur_action_dim:
                        raise StageError("inspect_h5", f"action dim changed: {action_dim} vs {cur_action_dim} in {h5_path}:{ep_key}")

                    if state_dim is None:
                        state_dim = cur_state_dim
                    elif state_dim != cur_state_dim:
                        raise StageError("inspect_h5", f"state dim changed: {state_dim} vs {cur_state_dim} in {h5_path}:{ep_key}")

                    action_config = build_action_config(episode, ts_keys, segment_mode, min_segment_frames)
                    if not action_config:
                        continue

                    episode_specs.append(
                        {
                            "source_h5": str(h5_path),
                            "source_episode_key": ep_key,
                            "task": task,
                            "length": len(ts_keys),
                            "action_config": action_config,
                        }
                    )
        except Exception as exc:
            bad_files.append((h5_path, type(exc).__name__, str(exc)))

    if bad_files:
        msg = "\n".join(f"  - {path}: {etype}: {reason}" for path, etype, reason in bad_files)
        if not skip_bad_files:
            raise RuntimeError(
                "Found unreadable or structurally invalid HDF5 file(s) before conversion:\n"
                f"{msg}\n"
                "Most common causes: truncated download/extract, missing required RoboMME fields, "
                "or action/image shape mismatch. Re-download/re-extract those files, or pass "
                "--skip-bad-files to convert the remaining files."
            )
        print("Skipping bad HDF5 file(s):", flush=True)
        print(msg, flush=True)

    if not episode_specs:
        raise RuntimeError("No valid RoboMME episodes found after inspection")
    assert action_dim is not None and state_dim is not None
    if action_dim != STANDARD_ACTION_DIM:
        raise RuntimeError(f"Mapped LingBot-VA action_dim must be 30, got {action_dim}")
    return episode_specs, task_to_index, action_dim, state_dim


def build_jobs(
    episode_specs: list[dict[str, Any]],
    task_to_index: dict[str, int],
    output_dir: Path,
    fps: int,
    image_size: int,
    action_key: str,
    state_dim: int,
    video_codec: str,
    chunks_size: int,
) -> list[EpisodeJob]:
    jobs = []
    frame_start = 0
    for episode_index, spec in enumerate(episode_specs):
        jobs.append(
            EpisodeJob(
                source_h5=spec["source_h5"],
                source_episode_key=spec["source_episode_key"],
                episode_index=episode_index,
                frame_start_index=frame_start,
                task=spec["task"],
                task_index=task_to_index[spec["task"]],
                length=spec["length"],
                action_config=spec["action_config"],
                output_dir=str(output_dir),
                fps=fps,
                image_size=image_size,
                action_key=action_key,
                state_dim=state_dim,
                video_codec=video_codec,
                chunks_size=chunks_size,
            )
        )
        frame_start += int(spec["length"])
    return jobs


def build_features(action_dim: int, state_dim: int, image_size: int, fps: int, video_codec: str) -> dict[str, dict[str, Any]]:
    video_info = {
        "video.height": image_size,
        "video.width": image_size,
        "video.codec": video_codec,
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    return {
        FRONT_KEY: {
            "dtype": "video",
            "shape": [3, image_size, image_size],
            "names": ["channels", "height", "width"],
            "info": video_info,
        },
        WRIST_KEY: {
            "dtype": "video",
            "shape": [3, image_size, image_size],
            "names": ["channels", "height", "width"],
            "info": video_info,
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": [STANDARD_ACTION_DIM],
            "names": [
                *[f"left_eef_{i}" for i in range(7)],
                *[f"right_eef_{i}" for i in range(7)],
                *[f"left_joint_{i}" for i in range(7)],
                *[f"right_joint_{i}" for i in range(7)],
                "left_gripper",
                "right_gripper",
            ],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata(
    output_dir: Path,
    jobs: list[EpisodeJob],
    results: list[EpisodeResult],
    task_to_index: dict[str, int],
    action_dim: int,
    state_dim: int,
    fps: int,
    image_size: int,
    video_codec: str,
    chunks_size: int,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    results_by_episode = {r.episode_index: r for r in results}
    total_frames = sum(job.length for job in jobs)
    total_episodes = len(jobs)
    total_chunks = (total_episodes + chunks_size - 1) // chunks_size

    info = {
        "codebase_version": "v2.1",
        "robot_type": "robomme",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_episodes * 2,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features(action_dim, state_dim, image_size, fps, video_codec),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    tasks_rows = [{"task_index": idx, "task": task} for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1])]
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)

    episodes_rows = []
    source_rows = []
    stats_rows = []
    selected_actions = []
    for job in jobs:
        result = results_by_episode[job.episode_index]
        episodes_rows.append(
            {
                "episode_index": job.episode_index,
                "tasks": [job.task],
                "length": job.length,
                "action_config": job.action_config,
            }
        )
        source_rows.append(
            {
                "episode_index": job.episode_index,
                "source_h5": job.source_h5,
                "source_episode_key": job.source_episode_key,
                "frame_start_index": job.frame_start_index,
            }
        )
        stats_rows.append({"episode_index": job.episode_index, "stats": result.episode_stats})
        if result.selected_actions:
            selected_actions.extend(result.selected_actions)

    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "robomme_sources.jsonl", source_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows)

    action_values = np.asarray(selected_actions, dtype=np.float32)
    q01 = np.zeros((STANDARD_ACTION_DIM,), dtype=np.float32)
    q99 = np.zeros((STANDARD_ACTION_DIM,), dtype=np.float32)
    used_action_channel_ids = sorted(set(np.where(np.any(np.abs(action_values) > 1e-12, axis=0))[0].astype(int).tolist()))
    if not used_action_channel_ids:
        raise RuntimeError("No non-zero action channels found after standard 30-D mapping")
    q01_values = np.quantile(action_values[:, used_action_channel_ids], 0.01, axis=0)
    q99_values = np.quantile(action_values[:, used_action_channel_ids], 0.99, axis=0)
    for idx, q01_value, q99_value in zip(used_action_channel_ids, q01_values, q99_values):
        q01[idx] = float(q01_value)
        q99[idx] = float(q99_value)
    action_stats = {
        "action_key": jobs[0].action_key,
        "action_source_dim": STANDARD_ACTION_DIM,
        "used_action_channel_ids": used_action_channel_ids,
        "standard_action_layout": {
            "left_eef": [0, 1, 2, 3, 4, 5, 6],
            "right_eef": [7, 8, 9, 10, 11, 12, 13],
            "left_joints": [14, 15, 16, 17, 18, 19, 20],
            "right_joints": [21, 22, 23, 24, 25, 26, 27],
            "left_gripper": [28],
            "right_gripper": [29],
        },
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "num_action_rows": int(action_values.shape[0]),
    }
    (meta_dir / "robomme_action_stats.json").write_text(json.dumps(action_stats, indent=2), encoding="utf-8")


def write_error_log(output_dir: Path, failures: list[EpisodeResult]) -> Path:
    error_path = output_dir / "meta" / "conversion_errors.jsonl"
    rows = [
        {
            "episode_index": f.episode_index,
            "source_h5": f.source_h5,
            "source_episode_key": f.source_episode_key,
            "stage": f.error_stage,
            "error_type": f.error_type,
            "error_message": f.error_message,
            "traceback": f.traceback,
        }
        for f in failures
    ]
    write_jsonl(error_path, rows)
    return error_path


def convert(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    episode_specs, task_to_index, action_dim, state_dim = inspect_h5_files(
        raw_dir=raw_dir,
        action_key=args.action_key,
        segment_mode=args.segment_mode,
        min_segment_frames=args.min_segment_frames,
        image_size=args.image_size,
        skip_bad_files=args.skip_bad_files,
    )
    jobs = build_jobs(
        episode_specs=episode_specs,
        task_to_index=task_to_index,
        output_dir=output_dir,
        fps=args.fps,
        image_size=args.image_size,
        action_key=args.action_key,
        state_dim=state_dim,
        video_codec=args.video_codec,
        chunks_size=args.chunks_size,
    )

    print(
        f"Discovered {len(jobs)} episodes, {sum(j.length for j in jobs)} frames, "
        f"{len(task_to_index)} tasks, action_dim={action_dim}, state_dim={state_dim}.",
        flush=True,
    )
    print(f"Writing dataset to {output_dir} with {args.num_workers} worker(s).", flush=True)

    results: list[EpisodeResult] = []
    failures: list[EpisodeResult] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_episode, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="episodes"):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = EpisodeResult(
                    ok=False,
                    episode_index=job.episode_index,
                    source_h5=job.source_h5,
                    source_episode_key=job.source_episode_key,
                    error_stage="worker_crash",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            if result.ok:
                results.append(result)
            else:
                failures.append(result)

    if failures:
        error_path = write_error_log(output_dir, failures)
        preview = "\n".join(
            f"  - episode={f.episode_index} h5={Path(f.source_h5).name} "
            f"source={f.source_episode_key} stage={f.error_stage} "
            f"{f.error_type}: {f.error_message}"
            for f in failures[:10]
        )
        raise RuntimeError(
            f"{len(failures)} episode(s) failed. Error log: {error_path}\n{preview}\n"
            "The dataset metadata was not written because partial episode failures would "
            "create invalid episode indices. Fix the reported H5/field/codec issue and rerun."
        )

    results.sort(key=lambda r: r.episode_index)
    write_metadata(
        output_dir=output_dir,
        jobs=jobs,
        results=results,
        task_to_index=task_to_index,
        action_dim=action_dim,
        state_dim=state_dim,
        fps=args.fps,
        image_size=args.image_size,
        video_codec=args.video_codec,
        chunks_size=args.chunks_size,
    )
    print(f"Finished conversion: {len(results)} episodes written to {output_dir}", flush=True)
    print(f"Action stats: {output_dir / 'meta' / 'robomme_action_stats.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-key", default="eef_action", choices=["eef_action", "joint_action", "waypoint_action"])
    parser.add_argument("--segment-mode", choices=["exec", "full", "subgoal"], default="exec")
    parser.add_argument("--min-segment-frames", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--video-codec", default="mpeg4", help="PyAV codec. Use mpeg4 for speed/compatibility, libx264 if available.")
    parser.add_argument("--chunks-size", type=int, default=DEFAULT_CHUNKS_SIZE)
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
