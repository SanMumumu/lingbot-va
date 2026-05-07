#!/usr/bin/env python3
"""Monitor LingBot latent extraction progress by counting written .pth files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latent_path(dataset_dir: Path, info: dict[str, Any], episode_index: int, camera_key: str, start: int, end: int) -> Path:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    return (
        dataset_dir
        / "latents"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start}_{end}.pth"
    )


def expected_paths(dataset_dir: Path, camera_keys: list[str]) -> list[Path]:
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    paths = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        for segment in episode.get("action_config", []):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            for camera_key in camera_keys:
                paths.append(latent_path(dataset_dir, info, episode_index, camera_key, start, end))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAMERA_KEYS)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--done-file",
        required=True,
        help="A file created by the parent script when all shard processes have exited.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    done_file = Path(args.done_file)
    paths = expected_paths(dataset_dir, args.camera_keys)

    with tqdm(total=len(paths), desc="latent files", unit="file") as pbar:
        last_done = 0
        while True:
            done = sum(1 for path in paths if path.exists())
            if done > last_done:
                pbar.update(done - last_done)
                last_done = done
            if done >= len(paths):
                break
            if done_file.exists():
                # Shards exited but files are incomplete; parent will print the shard logs.
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
