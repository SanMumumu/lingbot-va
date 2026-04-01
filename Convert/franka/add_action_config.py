from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common import (
    build_default_action_config,
    infer_task_text,
    read_json,
    read_jsonl,
    resolve_end_frame,
    write_jsonl,
)


def validate_action_config(
    action_config: list[dict[str, Any]],
    episode_length: int,
    episode_index: int,
) -> None:
    previous_end = 0
    for segment_id, segment in enumerate(action_config):
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        action_text = str(segment["action_text"]).strip()
        if start_frame < 0:
            raise ValueError(
                f"Episode {episode_index}: segment {segment_id} has negative start_frame={start_frame}."
            )
        if end_frame <= start_frame:
            raise ValueError(
                f"Episode {episode_index}: segment {segment_id} has invalid range "
                f"[{start_frame}, {end_frame})."
            )
        if end_frame > episode_length:
            raise ValueError(
                f"Episode {episode_index}: segment {segment_id} exceeds length={episode_length}."
            )
        if start_frame < previous_end:
            raise ValueError(
                f"Episode {episode_index}: segment {segment_id} overlaps a previous segment."
            )
        if not action_text:
            raise ValueError(
                f"Episode {episode_index}: segment {segment_id} has empty action_text."
            )
        previous_end = end_frame


def build_segments_for_episode(
    episode: dict[str, Any],
    default_action_text: str,
    segment_overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    episode_index = int(episode["episode_index"])
    episode_length = int(episode["length"])

    if not segment_overrides:
        action_text = default_action_text or episode["tasks"][0]
        return build_default_action_config(length=episode_length, action_text=action_text)

    per_episode_key = str(episode_index)
    segments_payload = None
    if per_episode_key in segment_overrides:
        segments_payload = segment_overrides[per_episode_key]
    elif "default" in segment_overrides:
        segments_payload = segment_overrides["default"]

    if segments_payload is None:
        action_text = default_action_text or episode["tasks"][0]
        return build_default_action_config(length=episode_length, action_text=action_text)

    segments: list[dict[str, Any]] = []
    for segment in segments_payload:
        segment_action_text = (
            segment.get("action_text")
            or default_action_text
            or episode["tasks"][0]
        )
        segments.append(
            {
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(resolve_end_frame(segment["end_frame"], episode_length)),
                "action_text": str(segment_action_text),
            }
        )
    validate_action_config(segments, episode_length=episode_length, episode_index=episode_index)
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add action_config to meta/episodes.jsonl for a converted Franka LeRobot dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Converted LeRobot dataset root produced by convert_to_lerobot.py.",
    )
    parser.add_argument(
        "--action-text",
        type=str,
        default=None,
        help="Default action_text used for every segment when no segment-specific text is provided.",
    )
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file describing multi-segment action_config. "
            "Supported format: {'default': [...]} or {'0': [...], '1': [...]}."
        ),
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create episodes.jsonl.bak before overwriting meta/episodes.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")

    episodes = read_jsonl(episodes_path)
    if not episodes:
        raise ValueError(f"No episode rows found in {episodes_path}")

    franka_meta = read_json(dataset_root / "meta" / "franka_meta.json", default={}) or {}
    default_action_text = (
        args.action_text
        or franka_meta.get("task_text")
        or infer_task_text(dataset_root)
    )
    segments_json = read_json(args.segments_json, default=None) if args.segments_json else None

    updated_rows: list[dict[str, Any]] = []
    for episode in episodes:
        row = dict(episode)
        row["action_config"] = build_segments_for_episode(
            episode=episode,
            default_action_text=default_action_text,
            segment_overrides=segments_json,
        )
        updated_rows.append(row)

    if args.backup:
        shutil.copy2(episodes_path, episodes_path.with_suffix(".jsonl.bak"))
    write_jsonl(episodes_path, updated_rows)

    print(f"Updated {episodes_path} with action_config for {len(updated_rows)} episodes.")
    if segments_json:
        print(f"Segments source: {args.segments_json}")
    else:
        print("Segments source: single segment per episode [0, length).")


if __name__ == "__main__":
    main()
