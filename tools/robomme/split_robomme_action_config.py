#!/usr/bin/env python3
"""Split long RoboMME action_config segments into VAE-friendly windows.

The converted LeRobot videos and parquet files stay untouched. Only
meta/episodes.jsonl is rewritten, so latent extraction and training consume
shorter clips while preserving the same episode data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_DATASET_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def normalize_max_sampled_frames(value: int) -> int:
    if value < 5:
        raise ValueError("--max-sampled-frames must be >= 5")
    # Wan VAE temporal compression is 4x. Keep sampled frame counts as 4n+1.
    return ((int(value) - 1) // 4) * 4 + 1


def split_segment(
    segment: dict[str, Any],
    stride: int,
    max_sampled_frames: int,
) -> list[dict[str, Any]]:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    if end <= start:
        return []

    frame_ids = list(range(start, end, stride))
    if len(frame_ids) <= max_sampled_frames:
        return [segment]

    source_start = int(segment.get("source_start_frame", start))
    source_end = int(segment.get("source_end_frame", end))
    parts: list[dict[str, Any]] = []
    chunk_id = 0
    for offset in range(0, len(frame_ids), max_sampled_frames):
        ids = frame_ids[offset : offset + max_sampled_frames]
        if len(ids) < 5:
            break

        part = dict(segment)
        part["start_frame"] = int(ids[0])
        part["end_frame"] = int(min(end, ids[-1] + stride))
        part["source_start_frame"] = source_start
        part["source_end_frame"] = source_end
        part["segment_part"] = chunk_id
        part["segment_parts_total"] = None
        parts.append(part)
        chunk_id += 1

    if parts and parts[-1]["end_frame"] < end:
        parts[-1]["end_frame"] = end

    total = len(parts)
    for part in parts:
        part["segment_parts_total"] = total
    return parts


def collapse_to_single_segment(episode: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the episode's action_config so it's one segment covering [0, length).

    Mirrors libero/robotwin where each episode is one training sample. Reuses
    the first segment's action_text (the natural-language goal); falls back to
    the episode's `tasks[0]` if no action_text is present.
    """
    action_config = episode.get("action_config", [])
    length = int(episode["length"])
    if action_config:
        action_text = str(action_config[0].get("action_text") or "")
    else:
        action_text = ""
    if not action_text:
        tasks = episode.get("tasks") or [""]
        action_text = str(tasks[0])
    episode["action_config"] = [
        {
            "start_frame": 0,
            "end_frame": length,
            "action_text": action_text,
            "source_start_frame": 0,
            "source_end_frame": length,
            "segment_part": 0,
            "segment_parts_total": 1,
        }
    ]
    return episode


def split_dataset(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)

    if args.unsplit:
        old_segments = sum(len(ep.get("action_config", [])) for ep in episodes)
        for episode in tqdm(episodes, desc="unsplit action_config"):
            collapse_to_single_segment(episode)
        new_segments = len(episodes)
        if old_segments == new_segments:
            print(
                f"No action_config unsplit needed: {old_segments} segment(s) "
                f"already match 1-per-episode."
            )
            return
        if args.backup:
            backup_path = episodes_path.with_suffix(".jsonl.bak")
            if not backup_path.exists():
                shutil.copy2(episodes_path, backup_path)
        write_jsonl(episodes_path, episodes)
        print(
            f"Rewrote {episodes_path}: {old_segments} -> {new_segments} "
            f"action_config segment(s) (libero-style; one per episode)."
        )
        return

    ori_fps = int(info["fps"])
    stride = max(1, int(round(float(ori_fps) / float(args.target_fps))))
    max_sampled_frames = normalize_max_sampled_frames(args.max_sampled_frames)

    changed = False
    old_segments = 0
    new_segments = 0
    for episode in tqdm(episodes, desc="split action_config"):
        action_config = episode.get("action_config", [])
        old_segments += len(action_config)
        new_action_config: list[dict[str, Any]] = []
        for segment in action_config:
            parts = split_segment(segment, stride, max_sampled_frames)
            new_action_config.extend(parts)
            if len(parts) != 1 or (parts and parts[0] != segment):
                changed = True
        episode["action_config"] = new_action_config
        new_segments += len(new_action_config)

    if not changed:
        print(
            f"No action_config split needed: {old_segments} segment(s), "
            f"max_sampled_frames={max_sampled_frames}, stride={stride}."
        )
        return

    if args.backup:
        backup_path = episodes_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            shutil.copy2(episodes_path, backup_path)

    write_jsonl(episodes_path, episodes)
    print(
        f"Rewrote {episodes_path}: {old_segments} -> {new_segments} action_config segment(s), "
        f"max_sampled_frames={max_sampled_frames}, stride={stride}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=81,
        help="Maximum sampled RGB frames per latent file; rounded down to 4n+1.",
    )
    parser.add_argument(
        "--unsplit",
        action="store_true",
        help="Reverse a previous split: collapse each episode's action_config back to "
             "one segment covering [0, length). Matches the libero/robotwin convention "
             "of one segment per episode. Ignores --max-sampled-frames / --target-fps.",
    )
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    split_dataset(parse_args())
