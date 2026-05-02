#!/usr/bin/env python3
"""Verify the converted RoboMME LingBot-VA dataset before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from tqdm import tqdm


DEFAULT_DATASET_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot"
CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]
REQUIRED_LATENT_KEYS = {
    "latent",
    "latent_num_frames",
    "latent_height",
    "latent_width",
    "video_num_frames",
    "video_height",
    "video_width",
    "text_emb",
    "text",
    "frame_ids",
    "start_frame",
    "end_frame",
    "fps",
    "ori_fps",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rel_data_path(info: dict[str, Any], episode_index: int) -> str:
    chunk = episode_index // int(info["chunks_size"])
    return info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def rel_video_path(info: dict[str, Any], episode_index: int, camera_key: str) -> str:
    chunk = episode_index // int(info["chunks_size"])
    return info["video_path"].format(episode_chunk=chunk, video_key=camera_key, episode_index=episode_index)


def latent_path(dataset_dir: Path, info: dict[str, Any], episode_index: int, camera_key: str, start: int, end: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return (
        dataset_dir
        / "latents"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start}_{end}.pth"
    )


def verify(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    errors = []

    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    action_stats_path = dataset_dir / "meta" / "robomme_action_stats.json"
    empty_emb_path = dataset_dir / "empty_emb.pt"

    for path in [info_path, episodes_path, tasks_path, stats_path, action_stats_path, empty_emb_path]:
        if not path.exists():
            errors.append({"stage": "metadata_exists", "path": str(path), "error": "missing required file"})

    if errors:
        error_path = dataset_dir / "meta" / "verify_errors.jsonl"
        write_jsonl(error_path, errors)
        raise RuntimeError(f"Dataset verification failed before loading metadata. Error log: {error_path}")

    info = load_json(info_path)
    episodes = load_jsonl(episodes_path)
    stats_rows = load_jsonl(stats_path)
    action_stats = load_json(action_stats_path)

    if len(episodes) != int(info["total_episodes"]):
        errors.append({"stage": "metadata_count", "error": f"episodes rows {len(episodes)} != info total_episodes {info['total_episodes']}"})
    if len(stats_rows) != len(episodes):
        errors.append({"stage": "metadata_count", "error": f"episodes_stats rows {len(stats_rows)} != episodes rows {len(episodes)}"})
    if len(action_stats.get("q01", [])) != 30 or len(action_stats.get("q99", [])) != 30:
        errors.append({"stage": "action_stats", "error": "q01/q99 must each have length 30"})

    loaded_latents = 0
    for episode in tqdm(episodes, desc="verify episodes"):
        episode_index = int(episode["episode_index"])
        expected_len = int(episode["length"])

        parquet_path = dataset_dir / rel_data_path(info, episode_index)
        if not parquet_path.exists():
            errors.append({"stage": "parquet_exists", "episode_index": episode_index, "path": str(parquet_path), "error": "missing parquet"})
        else:
            try:
                table = pq.read_table(parquet_path, columns=["action", "episode_index", "frame_index"])
                if table.num_rows != expected_len:
                    errors.append({"stage": "parquet_rows", "episode_index": episode_index, "error": f"{table.num_rows} != {expected_len}"})
            except Exception as exc:
                errors.append({"stage": "parquet_read", "episode_index": episode_index, "path": str(parquet_path), "error": f"{type(exc).__name__}: {exc}"})

        for camera_key in CAMERA_KEYS:
            video_path = dataset_dir / rel_video_path(info, episode_index, camera_key)
            if not video_path.exists():
                errors.append({"stage": "video_exists", "episode_index": episode_index, "camera_key": camera_key, "path": str(video_path), "error": "missing video"})

        for segment in episode.get("action_config", []):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if not (0 <= start < end <= expected_len):
                errors.append({"stage": "action_config_range", "episode_index": episode_index, "segment": segment, "error": "invalid start/end"})
            for camera_key in CAMERA_KEYS:
                path = latent_path(dataset_dir, info, episode_index, camera_key, start, end)
                if not path.exists():
                    errors.append({"stage": "latent_exists", "episode_index": episode_index, "camera_key": camera_key, "path": str(path), "error": "missing latent"})
                    continue
                if args.deep_limit < 0 or loaded_latents < args.deep_limit:
                    try:
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                        missing = sorted(REQUIRED_LATENT_KEYS - set(payload))
                        if missing:
                            errors.append({"stage": "latent_keys", "path": str(path), "error": f"missing keys {missing}"})
                        loaded_latents += 1
                    except Exception as exc:
                        errors.append({"stage": "latent_load", "path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if errors:
        error_path = dataset_dir / "meta" / "verify_errors.jsonl"
        write_jsonl(error_path, errors)
        preview = "\n".join(f"  - {e}" for e in errors[:10])
        raise RuntimeError(f"Dataset verification failed with {len(errors)} error(s). Error log: {error_path}\n{preview}")

    print(
        f"Verification passed: {len(episodes)} episodes, "
        f"{sum(len(e.get('action_config', [])) for e in episodes)} segment(s), "
        f"deep-loaded {loaded_latents} latent file(s)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--deep-limit",
        type=int,
        default=32,
        help="Number of latent .pth files to load. Use -1 to load all; 0 only checks file existence.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
