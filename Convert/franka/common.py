from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CAMERA_MAPPING = {
    "left_camera": "observation.images.left_camera",
    "right_camera": "observation.images.right_camera",
    "wrist_camera": "observation.images.wrist_camera",
}

EEF_FEATURE_NAMES = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "eef_quat_w",
]
JOINT_FEATURE_NAMES = [f"joint_{idx + 1}" for idx in range(7)]
GRIPPER_FEATURE_NAMES = ["gripper_width"]

COMPACT_ACTION_FEATURE_NAMES = (
    EEF_FEATURE_NAMES + JOINT_FEATURE_NAMES + GRIPPER_FEATURE_NAMES
)
OBSERVATION_STATE_FEATURE_NAMES = (
    JOINT_FEATURE_NAMES + EEF_FEATURE_NAMES + GRIPPER_FEATURE_NAMES
)

# LingBot-VA standard 30D action layout:
# 0:7   -> left arm eef
# 7:14  -> right arm eef
# 14:21 -> left arm joints
# 21:28 -> right arm joints
# 28    -> left gripper
# 29    -> right gripper
SINGLE_ARM_USED_ACTION_CHANNEL_IDS = list(range(0, 7)) + list(range(14, 21)) + [28]
STANDARD_ACTION_DIM = 30


def natural_key(value: str | Path) -> list[Any]:
    text = str(value)
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def natural_sorted(values: list[Path]) -> list[Path]:
    return sorted(values, key=natural_key)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def infer_task_text(path_like: str | Path) -> str:
    path = Path(path_like)
    stem = path.name
    if "_" in stem:
        stem = stem.split("_", maxsplit=1)[-1]
    stem = stem.replace("-", " ").replace("_", " ")
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "franka manipulation"


def parse_camera_mapping(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERA_MAPPING)

    mapping: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --camera-mapping value {item!r}. Expected RAW_NAME=LEROBOT_KEY."
            )
        raw_name, lerobot_key = item.split("=", maxsplit=1)
        raw_name = raw_name.strip()
        lerobot_key = lerobot_key.strip()
        if not raw_name or not lerobot_key:
            raise ValueError(
                f"Invalid --camera-mapping value {item!r}. Empty names are not allowed."
            )
        mapping[raw_name] = lerobot_key
    return mapping


def to_posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def get_episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def build_default_action_config(length: int, action_text: str) -> list[dict[str, Any]]:
    return [
        {
            "start_frame": 0,
            "end_frame": length,
            "action_text": action_text,
        }
    ]


def resolve_end_frame(value: int | str, length: int) -> int:
    if isinstance(value, str):
        if value.lower() == "length":
            return length
        value = int(value)
    return int(value)


def compact_stats_to_standard(
    q01_compact: np.ndarray,
    q99_compact: np.ndarray,
    used_action_channel_ids: list[int] | None = None,
) -> dict[str, list[float]]:
    used_action_channel_ids = (
        used_action_channel_ids
        if used_action_channel_ids is not None
        else SINGLE_ARM_USED_ACTION_CHANNEL_IDS
    )
    q01_standard = np.zeros(STANDARD_ACTION_DIM, dtype=np.float32)
    q99_standard = np.zeros(STANDARD_ACTION_DIM, dtype=np.float32)
    q01_standard[used_action_channel_ids] = q01_compact.astype(np.float32)
    q99_standard[used_action_channel_ids] = q99_compact.astype(np.float32)
    return {
        "q01": q01_standard.tolist(),
        "q99": q99_standard.tolist(),
        "used_action_channel_ids": list(used_action_channel_ids),
        "compact_feature_names": list(COMPACT_ACTION_FEATURE_NAMES),
        "standard_action_dim": STANDARD_ACTION_DIM,
    }

