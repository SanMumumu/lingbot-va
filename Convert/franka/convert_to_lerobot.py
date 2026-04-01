from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from common import (
    COMPACT_ACTION_FEATURE_NAMES,
    DEFAULT_CAMERA_MAPPING,
    OBSERVATION_STATE_FEATURE_NAMES,
    SINGLE_ARM_USED_ACTION_CHANNEL_IDS,
    compact_stats_to_standard,
    get_episode_chunk,
    infer_task_text,
    natural_sorted,
    parse_camera_mapping,
    to_posix_relative,
    write_json,
    write_jsonl,
)


REQUIRED_STATE_KEYS = (
    "robot/joint_positions",
    "robot/ee_pose",
    "gripper/width",
)


@dataclass
class RunningVectorStats:
    dims: int
    min_value: np.ndarray | None = None
    max_value: np.ndarray | None = None
    sum_value: np.ndarray | None = None
    sum_sq_value: np.ndarray | None = None
    count: int = 0

    def update(self, array: np.ndarray) -> None:
        values = np.asarray(array, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] != self.dims:
            raise ValueError(f"Expected {self.dims} dims, got {values.shape[1]}.")
        local_min = values.min(axis=0)
        local_max = values.max(axis=0)
        local_sum = values.sum(axis=0)
        local_sum_sq = np.square(values).sum(axis=0)
        if self.min_value is None:
            self.min_value = local_min
            self.max_value = local_max
            self.sum_value = local_sum
            self.sum_sq_value = local_sum_sq
        else:
            self.min_value = np.minimum(self.min_value, local_min)
            self.max_value = np.maximum(self.max_value, local_max)
            self.sum_value += local_sum
            self.sum_sq_value += local_sum_sq
        self.count += values.shape[0]

    def to_dict(self) -> dict[str, list[float]]:
        if self.count == 0 or self.min_value is None:
            raise ValueError("Stats are empty.")
        mean = self.sum_value / self.count
        var = np.maximum(self.sum_sq_value / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        count = [int(self.count)]
        return {
            "min": self.min_value.astype(np.float32).tolist(),
            "max": self.max_value.astype(np.float32).tolist(),
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "count": count,
        }


@dataclass
class RunningImageStats:
    min_value: np.ndarray | None = None
    max_value: np.ndarray | None = None
    sum_value: np.ndarray | None = None
    sum_sq_value: np.ndarray | None = None
    count: int = 0

    def update(self, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {image.shape}.")
        values = image.astype(np.float64).reshape(-1, 3)
        local_min = values.min(axis=0)
        local_max = values.max(axis=0)
        local_sum = values.sum(axis=0)
        local_sum_sq = np.square(values).sum(axis=0)
        if self.min_value is None:
            self.min_value = local_min
            self.max_value = local_max
            self.sum_value = local_sum
            self.sum_sq_value = local_sum_sq
        else:
            self.min_value = np.minimum(self.min_value, local_min)
            self.max_value = np.maximum(self.max_value, local_max)
            self.sum_value += local_sum
            self.sum_sq_value += local_sum_sq
        self.count += values.shape[0]

    def to_dict(self) -> dict[str, list[list[list[float]]]]:
        if self.count == 0 or self.min_value is None:
            raise ValueError("Image stats are empty.")
        mean = self.sum_value / self.count
        var = np.maximum(self.sum_sq_value / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)

        def as_chw(vector: np.ndarray) -> list[list[list[float]]]:
            return vector.astype(np.float32).reshape(3, 1, 1).tolist()

        return {
            "min": as_chw(self.min_value),
            "max": as_chw(self.max_value),
            "mean": as_chw(mean),
            "std": as_chw(std),
            "count": [int(self.count)],
        }


def fixed_size_list_array(array: np.ndarray) -> pa.Array:
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.shape}.")
    flat = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, array.shape[1])


def build_features(camera_keys: list[str]) -> dict[str, Any]:
    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "shape": [len(COMPACT_ACTION_FEATURE_NAMES)],
            "names": list(COMPACT_ACTION_FEATURE_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(OBSERVATION_STATE_FEATURE_NAMES)],
            "names": list(OBSERVATION_STATE_FEATURE_NAMES),
        },
        "observation.joint_positions": {
            "dtype": "float32",
            "shape": [7],
            "names": list(COMPACT_ACTION_FEATURE_NAMES[7:14]),
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": [7],
            "names": list(COMPACT_ACTION_FEATURE_NAMES[:7]),
        },
        "observation.gripper_width": {
            "dtype": "float32",
            "shape": [1],
            "names": ["gripper_width"],
        },
        "observation.joint_velocities": {
            "dtype": "float32",
            "shape": [7],
            "names": [f"joint_velocity_{idx + 1}" for idx in range(7)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for key in camera_keys:
        features[key] = {
            "dtype": "video",
            "shape": [3, None, None],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": 1.0,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "has_audio": False,
            },
        }
    return features


def compute_episode_fps(timestamp: np.ndarray, fallback_fps: float) -> float:
    if timestamp.size < 2:
        return float(fallback_fps)
    diffs = np.diff(timestamp)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float(fallback_fps)
    return float(round(1.0 / np.median(diffs)))


def build_action_targets(
    ee_pose: np.ndarray,
    joint_positions: np.ndarray,
    gripper_width: np.ndarray,
    action_offset: int,
) -> np.ndarray:
    compact_action = np.concatenate(
        [ee_pose, joint_positions, gripper_width[:, None]],
        axis=1,
    ).astype(np.float32)
    if action_offset <= 0:
        return compact_action

    if action_offset >= compact_action.shape[0]:
        return np.repeat(compact_action[-1:], compact_action.shape[0], axis=0)

    tail = np.repeat(compact_action[-1:], action_offset, axis=0)
    shifted = np.concatenate([compact_action[action_offset:], tail], axis=0)
    return shifted[: compact_action.shape[0]]


def load_episode(
    episode_dir: Path,
    camera_mapping: dict[str, str],
    fallback_fps: float,
    action_offset: int,
) -> dict[str, Any]:
    states_path = episode_dir / "states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing states file: {states_path}")

    state_dict = np.load(states_path, allow_pickle=True)
    for key in REQUIRED_STATE_KEYS:
        if key not in state_dict:
            raise KeyError(f"Missing required key {key!r} in {states_path}")

    joint_positions = np.asarray(state_dict["robot/joint_positions"], dtype=np.float32)
    ee_pose = np.asarray(state_dict["robot/ee_pose"], dtype=np.float32)
    gripper_width = np.asarray(state_dict["gripper/width"], dtype=np.float32)
    joint_velocities = np.asarray(
        state_dict["robot/joint_velocities"],
        dtype=np.float32,
    ) if "robot/joint_velocities" in state_dict else np.zeros_like(joint_positions)

    frame_lists: dict[str, list[Path]] = {}
    for raw_camera_name in camera_mapping:
        camera_dir = episode_dir / "image" / raw_camera_name
        if not camera_dir.exists():
            raise FileNotFoundError(f"Missing camera directory: {camera_dir}")
        frames = natural_sorted(list(camera_dir.glob("*.png")))
        if not frames:
            raise FileNotFoundError(f"No PNG frames found under {camera_dir}")
        frame_lists[raw_camera_name] = frames

    common_length = min(
        joint_positions.shape[0],
        ee_pose.shape[0],
        gripper_width.shape[0],
        joint_velocities.shape[0],
        *(len(frames) for frames in frame_lists.values()),
    )
    if common_length < 2:
        raise ValueError(
            f"{episode_dir} only has {common_length} aligned frames. Need at least 2."
        )

    robot_timestamp = np.asarray(
        state_dict["robot/robot_timestamp"],
        dtype=np.float64,
    ) if "robot/robot_timestamp" in state_dict else None
    if robot_timestamp is not None and robot_timestamp.shape[0] >= common_length:
        timestamp = (robot_timestamp[:common_length] - robot_timestamp[0]).astype(np.float32)
        fps = compute_episode_fps(robot_timestamp[:common_length], fallback_fps)
    else:
        fps = float(fallback_fps)
        timestamp = (np.arange(common_length, dtype=np.float32) / fps).astype(np.float32)

    joint_positions = joint_positions[:common_length]
    ee_pose = ee_pose[:common_length]
    gripper_width = gripper_width[:common_length]
    joint_velocities = joint_velocities[:common_length]
    compact_action = build_action_targets(
        ee_pose=ee_pose,
        joint_positions=joint_positions,
        gripper_width=gripper_width,
        action_offset=action_offset,
    )
    observation_state = np.concatenate(
        [joint_positions, ee_pose, gripper_width[:, None]],
        axis=1,
    ).astype(np.float32)

    return {
        "fps": fps,
        "timestamp": timestamp,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "ee_pose": ee_pose,
        "gripper_width": gripper_width[:, None].astype(np.float32),
        "observation_state": observation_state,
        "action": compact_action,
        "frames": {
            raw_name: frames[:common_length]
            for raw_name, frames in frame_lists.items()
        },
        "length": common_length,
    }


def write_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    image_stats: RunningImageStats,
    sample_count: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = set(
        np.linspace(
            0,
            len(frame_paths) - 1,
            num=min(sample_count, len(frame_paths)),
            dtype=int,
        ).tolist()
    )

    width = -1
    height = -1
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        ffmpeg_log_level="error",
        macro_block_size=1,
    ) as writer:
        for frame_idx, frame_path in enumerate(frame_paths):
            image = np.asarray(Image.open(frame_path).convert("RGB"))
            if width == -1:
                height, width = image.shape[0], image.shape[1]
            writer.append_data(image)
            if frame_idx in sample_indices:
                image_stats.update(image)
    return height, width


def build_table(
    episode_index: int,
    global_index_start: int,
    task_index: int,
    episode: dict[str, Any],
    camera_mapping: dict[str, str],
    video_rel_paths: dict[str, str],
) -> pa.Table:
    length = episode["length"]
    columns: dict[str, pa.Array] = {
        "action": fixed_size_list_array(episode["action"]),
        "observation.state": fixed_size_list_array(episode["observation_state"]),
        "observation.joint_positions": fixed_size_list_array(episode["joint_positions"]),
        "observation.ee_pose": fixed_size_list_array(episode["ee_pose"]),
        "observation.gripper_width": fixed_size_list_array(episode["gripper_width"]),
        "observation.joint_velocities": fixed_size_list_array(episode["joint_velocities"]),
        "timestamp": pa.array(episode["timestamp"], type=pa.float32()),
        "frame_index": pa.array(np.arange(length, dtype=np.int64)),
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
        "index": pa.array(
            np.arange(global_index_start, global_index_start + length, dtype=np.int64)
        ),
        "task_index": pa.array(np.full(length, task_index, dtype=np.int64)),
    }

    for raw_camera_name, lerobot_key in camera_mapping.items():
        columns[lerobot_key] = pa.array(
            [video_rel_paths[raw_camera_name]] * length,
            type=pa.string(),
        )

    return pa.table(columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-arm Franka recordings to a local LeRobot v2.1 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing recordings/episode_xxxxx.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where the converted LeRobot dataset will be written.",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default=None,
        help="Task text stored in tasks.jsonl and episodes.jsonl. Defaults to the input folder name.",
    )
    parser.add_argument(
        "--camera-mapping",
        nargs="*",
        default=None,
        help=(
            "Override the raw-camera to LeRobot-key mapping. "
            "Format: RAW_NAME=LEROBOT_KEY"
        ),
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=1000,
        help="LeRobot chunk size for data/, videos/, and latents/ directory layout.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=15.0,
        help="FPS used when robot timestamps are missing or invalid.",
    )
    parser.add_argument(
        "--action-offset",
        type=int,
        default=1,
        help=(
            "Action[t] is built from state[t + action_offset]. "
            "Use 1 for next-state targets, 0 for same-step targets."
        ),
    )
    parser.add_argument(
        "--image-stat-samples",
        type=int,
        default=32,
        help="How many frames per camera video are sampled to estimate image stats.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing the converted dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    camera_mapping = parse_camera_mapping(args.camera_mapping)

    recordings_root = input_root / "recordings"
    if not recordings_root.exists():
        raise FileNotFoundError(f"Missing recordings directory: {recordings_root}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    task_text = args.task_text or infer_task_text(input_root)
    task_index = 0

    episode_dirs = natural_sorted(
        [path for path in recordings_root.iterdir() if path.is_dir() and path.name.startswith("episode_")]
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {recordings_root}")

    image_stats = {
        lerobot_key: RunningImageStats()
        for lerobot_key in camera_mapping.values()
    }
    action_stats = RunningVectorStats(dims=len(COMPACT_ACTION_FEATURE_NAMES))
    observation_state_stats = RunningVectorStats(dims=len(OBSERVATION_STATE_FEATURE_NAMES))
    action_value_rows: list[np.ndarray] = []

    parquet_write_paths: list[Path] = []
    episodes_jsonl: list[dict[str, Any]] = []
    tasks_jsonl = [{"task_index": task_index, "task": task_text}]
    fps_values: list[float] = []
    total_frames = 0
    global_index = 0

    for episode_index, episode_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        episode = load_episode(
            episode_dir=episode_dir,
            camera_mapping=camera_mapping,
            fallback_fps=args.fallback_fps,
            action_offset=args.action_offset,
        )
        fps_values.append(episode["fps"])

        episode_chunk = get_episode_chunk(episode_index, args.chunks_size)
        data_path = (
            output_root
            / "data"
            / f"chunk-{episode_chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)

        video_rel_paths: dict[str, str] = {}
        for raw_camera_name, lerobot_key in camera_mapping.items():
            video_path = (
                output_root
                / "videos"
                / f"chunk-{episode_chunk:03d}"
                / lerobot_key
                / f"episode_{episode_index:06d}.mp4"
            )
            height, width = write_video(
                frame_paths=episode["frames"][raw_camera_name],
                output_path=video_path,
                fps=episode["fps"],
                image_stats=image_stats[lerobot_key],
                sample_count=args.image_stat_samples,
            )
            video_rel_paths[raw_camera_name] = to_posix_relative(video_path, output_root)

        table = build_table(
            episode_index=episode_index,
            global_index_start=global_index,
            task_index=task_index,
            episode=episode,
            camera_mapping=camera_mapping,
            video_rel_paths=video_rel_paths,
        )
        pq.write_table(table, data_path, compression="zstd")
        parquet_write_paths.append(data_path)

        action_stats.update(episode["action"])
        observation_state_stats.update(episode["observation_state"])
        action_value_rows.append(episode["action"])

        episodes_jsonl.append(
            {
                "episode_index": episode_index,
                "tasks": [task_text],
                "length": int(episode["length"]),
            }
        )

        global_index += episode["length"]
        total_frames += episode["length"]

    fps_reference = float(round(float(np.median(np.asarray(fps_values))), 3))
    if max(fps_values) - min(fps_values) > 1e-3:
        print(
            "[WARN] Episode FPS values are not identical. "
            f"Using median fps={fps_reference} in meta/info.json."
        )

    stats_json = {
        "action": action_stats.to_dict(),
        "observation.state": observation_state_stats.to_dict(),
    }
    for lerobot_key, stats in image_stats.items():
        stats_json[lerobot_key] = stats.to_dict()

    action_values = np.concatenate(action_value_rows, axis=0)
    q01_compact = np.quantile(action_values, 0.01, axis=0).astype(np.float32)
    q99_compact = np.quantile(action_values, 0.99, axis=0).astype(np.float32)
    action_norm_stats = compact_stats_to_standard(
        q01_compact=q01_compact,
        q99_compact=q99_compact,
        used_action_channel_ids=SINGLE_ARM_USED_ACTION_CHANNEL_IDS,
    )

    meta_root = output_root / "meta"
    write_jsonl(meta_root / "tasks.jsonl", tasks_jsonl)
    write_jsonl(meta_root / "episodes.jsonl", episodes_jsonl)
    write_json(meta_root / "stats.json", stats_json)
    write_json(meta_root / "action_norm_stats.json", action_norm_stats)
    write_json(
        meta_root / "franka_meta.json",
        {
            "task_text": task_text,
            "obs_cam_keys": list(camera_mapping.values()),
            "camera_mapping": camera_mapping,
            "compact_action_feature_names": list(COMPACT_ACTION_FEATURE_NAMES),
            "observation_state_feature_names": list(OBSERVATION_STATE_FEATURE_NAMES),
            "used_action_channel_ids": list(SINGLE_ARM_USED_ACTION_CHANNEL_IDS),
            "action_offset": int(args.action_offset),
            "fps": fps_reference,
        },
    )

    info_json = {
        "codebase_version": "v2.1",
        "robot_type": "franka_single_arm",
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episode_dirs) * len(camera_mapping),
        "total_chunks": math.ceil(len(episode_dirs) / args.chunks_size),
        "chunks_size": args.chunks_size,
        "fps": fps_reference,
        "splits": {"train": f"0:{len(episode_dirs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": build_features(list(camera_mapping.values())),
    }
    for lerobot_key in camera_mapping.values():
        info_json["features"][lerobot_key]["info"]["video.fps"] = fps_reference
    write_json(meta_root / "info.json", info_json)

    print(f"Converted {len(episode_dirs)} episodes into {output_root}")
    print(f"Task text: {task_text}")
    print(f"Camera keys: {', '.join(camera_mapping.values())}")
    print("Next step: run add_action_config.py, then extract_wan_latents.py")


if __name__ == "__main__":
    main()
