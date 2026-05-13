#!/usr/bin/env python3
"""Prepare a CALVIN LeRobot dataset for LingBot-VA training.

The Hugging Face CALVIN LeRobot repos store images inline in parquet files
instead of under ``videos/``.  LingBot-VA only needs the LeRobot action rows,
``episodes.jsonl`` action segments, Wan2.2 VAE latents, and ``empty_emb.pt``;
this script extracts latents directly from parquet image bytes to avoid a
large intermediate video copy.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/calvin_abc_d_lerobot"
DEFAULT_REPO_ID = "fywang/calvin-task-ABC-D-lerobot"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_WAN22 = "/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_CAMERA_KEYS = ["observation.images.top", "observation.images.wrist"]
REQUIRED_WAN22_FILES = [
    "vae/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
]


def add_lingbot_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "wan_va"))
    sys.path.insert(0, str(repo_root))


add_lingbot_to_path()

from modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae  # noqa: E402

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: E402
except Exception:  # pragma: no cover

    def prompt_clean(text: str) -> str:
        return text


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def maybe_download(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    if (dataset_dir / "meta" / "info.json").exists():
        return
    if not args.download:
        raise FileNotFoundError(
            f"Missing {dataset_dir / 'meta' / 'info.json'}. "
            "Run DATA_MY/download_calvin.sh first, or pass --download to fetch the "
            "Hugging Face LeRobot dataset from this script."
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli",
        "download",
        "--repo-type",
        "dataset",
        "--resume-download",
        args.repo_id,
        "--local-dir",
        str(dataset_dir),
    ]
    env = os.environ.copy()
    env["HF_ENDPOINT"] = args.hf_endpoint
    subprocess.run(cmd, check=True, env=env)


def data_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset_dir / rel


def latent_path(dataset_dir: Path, info: dict[str, Any], episode_index: int, camera_key: str, start: int, end: int) -> Path:
    chunk = episode_index // int(info["chunks_size"])
    return (
        dataset_dir
        / "latents"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start}_{end}.pth"
    )


def ensure_action_config(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    changed = False
    for episode in episodes:
        if episode.get("action_config"):
            continue
        tasks = episode.get("tasks") or ["calvin task"]
        episode["action_config"] = [
            {
                "start_frame": 0,
                "end_frame": int(episode["length"]),
                "action_text": str(tasks[0]),
            }
        ]
        changed = True
    if changed:
        write_jsonl(episodes_path, episodes)
    print(f"Checked action_config for {len(episodes)} episode(s); changed={changed}.")


def action_stat_path(dataset_dir: Path) -> Path:
    return dataset_dir / "meta" / "calvin_action_stats.json"


def compute_action_stats(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if args.max_stat_episodes > 0:
        episodes = episodes[: args.max_stat_episodes]

    chunks = []
    for episode in tqdm(episodes, desc="action stats"):
        table = pq.read_table(data_path(dataset_dir, info, int(episode["episode_index"])), columns=["action"])
        chunks.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
    if not chunks:
        raise RuntimeError("No episodes available for action stats.")

    actions = np.concatenate(chunks, axis=0)
    q01_7 = np.quantile(actions, 0.01, axis=0)
    q99_7 = np.quantile(actions, 0.99, axis=0)
    payload = {
        "source": str(dataset_dir),
        "episodes": len(episodes),
        "frames": int(actions.shape[0]),
        "q01": q01_7.tolist() + [0.0] * 23,
        "q99": q99_7.tolist() + [0.0] * 23,
    }
    write_json(action_stat_path(dataset_dir), payload)
    print(f"Wrote {action_stat_path(dataset_dir)} from {actions.shape[0]} frame(s).")


def validate_wan22_path(wan22_path: Path) -> None:
    missing = [rel for rel in REQUIRED_WAN22_FILES if not (wan22_path / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"--wan22-path must point to a Diffusers-format Wan2.2 directory; "
            f"missing {missing} under {wan22_path}."
        )


def normalize_max_sampled_frames(value: int) -> int:
    if value <= 0:
        return 0
    if value < 5:
        raise ValueError("--max-sampled-frames must be 0 or >= 5")
    return ((int(value) - 1) // 4) * 4 + 1


def frame_ids_for_segment(start: int, end: int, ori_fps: int, target_fps: int) -> tuple[list[int], int]:
    if end <= start:
        return [], ori_fps
    stride = max(1, int(round(float(ori_fps) / float(target_fps))))
    frame_ids = list(range(start, end, stride))
    usable = ((len(frame_ids) - 1) // 4) * 4 + 1
    frame_ids = frame_ids[:usable] if usable >= 1 else []
    actual_fps = max(1, int(round(float(ori_fps) / stride)))
    return frame_ids, actual_fps


def decode_image_cell(cell: dict[str, Any], height: int, width: int) -> np.ndarray:
    if cell.get("bytes") is None:
        raise RuntimeError(f"Image cell has no inline bytes; path-only image cells are unsupported: {cell.get('path')}")
    image = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def read_parquet_frames(parquet_path: Path, camera_key: str, frame_ids: list[int], height: int, width: int) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=[camera_key])
    rows = table.column(camera_key).to_pylist()
    frames = [decode_image_cell(rows[idx], height, width) for idx in frame_ids]
    if not frames:
        raise RuntimeError(f"No frames selected from {parquet_path}")
    return np.stack(frames, axis=0)


@torch.no_grad()
def encode_text(text: str, tokenizer, text_encoder, device: torch.device, dtype: torch.dtype, max_sequence_length: int) -> torch.Tensor:
    text = prompt_clean(text or "")
    inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    enc_device = next(text_encoder.parameters()).device
    hidden = text_encoder(inputs.input_ids.to(enc_device), mask.to(enc_device)).last_hidden_state
    hidden = hidden.to(dtype=dtype, device=device)
    hidden = [u[:v] for u, v in zip(hidden, seq_lens)]
    hidden = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in hidden],
        dim=0,
    )
    return hidden[0].detach().cpu()


def normalize_latents(vae, mu: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)
    return ((mu.float() - latents_mean) * (1.0 / latents_std)).to(dtype)


@torch.no_grad()
def encode_video_segment(frames: np.ndarray, streaming_vae: WanVAEStreamingWrapper, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
    video = video / 255.0 * 2.0 - 1.0
    vae_device = next(streaming_vae.vae.parameters()).device
    streaming_vae.clear_cache()
    video = video.to(vae_device, dtype=dtype)
    enc_chunks = [streaming_vae.encode_chunk(video[:, :, :1])]
    for start in range(1, video.shape[2], 4):
        enc_chunks.append(streaming_vae.encode_chunk(video[:, :, start : start + 4]))
    enc = torch.cat(enc_chunks, dim=2)
    mu, _logvar = torch.chunk(enc, 2, dim=1)
    return normalize_latents(streaming_vae.vae, mu, dtype).to(device)


def write_latent_file(
    output_path: Path,
    latent_5d: torch.Tensor,
    text_emb: torch.Tensor,
    text: str,
    frame_ids: list[int],
    start: int,
    end: int,
    fps: int,
    ori_fps: int,
    video_height: int,
    video_width: int,
) -> None:
    latent = latent_5d[0]
    c, f, h, w = latent.shape
    flat = latent.permute(1, 2, 3, 0).contiguous().view(f * h * w, c).detach().cpu()
    payload = {
        "latent": flat.to(torch.bfloat16),
        "latent_num_frames": int(f),
        "latent_height": int(h),
        "latent_width": int(w),
        "video_num_frames": int(len(frame_ids)),
        "video_height": int(video_height),
        "video_width": int(video_width),
        "text_emb": text_emb.to(torch.bfloat16),
        "text": text,
        "frame_ids": [int(v) for v in frame_ids],
        "start_frame": int(start),
        "end_frame": int(end),
        "fps": int(fps),
        "ori_fps": int(ori_fps),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(output_path)


def build_units(episodes: list[dict[str, Any]], camera_keys: list[str], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    units = []
    unit_id = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        tasks = episode.get("tasks") or ["calvin task"]
        for segment in episode.get("action_config", []):
            for camera_key in camera_keys:
                if unit_id % num_shards == shard_index:
                    units.append(
                        {
                            "unit_id": unit_id,
                            "episode_index": episode_index,
                            "camera_key": camera_key,
                            "start": int(segment["start_frame"]),
                            "end": int(segment["end_frame"]),
                            "text": str(segment.get("action_text") or tasks[0]),
                        }
                    )
                unit_id += 1
    return units


def extract_latents(args: argparse.Namespace) -> None:
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    dataset_dir = Path(args.dataset_dir)
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    units = build_units(episodes, args.camera_keys, args.num_shards, args.shard_index)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    model_device = device if not args.offload else torch.device("cpu")
    wan22_path = Path(args.wan22_path)
    validate_wan22_path(wan22_path)

    vae = load_vae(wan22_path / "vae", torch_dtype=dtype, torch_device=model_device)
    streaming_vae = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(wan22_path / "tokenizer")
    text_encoder = load_text_encoder(wan22_path / "text_encoder", torch_dtype=dtype, torch_device=model_device)

    if args.shard_index == 0:
        empty_emb = encode_text("", tokenizer, text_encoder, device, dtype, args.max_sequence_length)
        tmp = dataset_dir / f"empty_emb.pt.tmp.{os.getpid()}"
        torch.save(empty_emb, tmp)
        tmp.replace(dataset_dir / "empty_emb.pt")

    ori_fps = int(info["fps"])
    max_sampled_frames = normalize_max_sampled_frames(args.max_sampled_frames)
    failures = []
    for unit in tqdm(units, desc=f"calvin latents shard {args.shard_index}/{args.num_shards}"):
        output_path = latent_path(dataset_dir, info, unit["episode_index"], unit["camera_key"], unit["start"], unit["end"])
        try:
            if output_path.exists() and args.resume:
                continue
            frame_ids, actual_fps = frame_ids_for_segment(unit["start"], unit["end"], ori_fps, args.target_fps)
            if len(frame_ids) < args.min_sampled_frames:
                raise RuntimeError(f"segment too short after fps sampling: {unit}, sampled={len(frame_ids)}")
            if max_sampled_frames and len(frame_ids) > max_sampled_frames:
                raise RuntimeError(f"segment too long for one VAE encode: {unit}, sampled={len(frame_ids)}")

            parquet_path = data_path(dataset_dir, info, unit["episode_index"])
            frames = read_parquet_frames(parquet_path, unit["camera_key"], frame_ids, args.height, args.width)
            text_emb = encode_text(unit["text"], tokenizer, text_encoder, device, dtype, args.max_sequence_length)
            latent = encode_video_segment(frames, streaming_vae, device, dtype)

            expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
            if latent.shape[2] != expected_latent_frames:
                raise RuntimeError(f"latent frame mismatch: got={latent.shape[2]} expected={expected_latent_frames}")

            write_latent_file(
                output_path,
                latent,
                text_emb,
                unit["text"],
                frame_ids,
                unit["start"],
                unit["end"],
                actual_fps,
                ori_fps,
                args.height,
                args.width,
            )
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            failures.append(
                {
                    "unit": unit,
                    "output_path": str(output_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures:
        error_path = dataset_dir / "meta" / f"calvin_latent_errors_shard{args.shard_index:03d}.jsonl"
        write_jsonl(error_path, failures)
        preview = "\n".join(
            f"  - episode={f['unit']['episode_index']} camera={f['unit']['camera_key']} {f['error_type']}: {f['error_message']}"
            for f in failures[:10]
        )
        raise RuntimeError(f"{len(failures)} latent unit(s) failed. Error log: {error_path}\n{preview}")

    print(f"Shard {args.shard_index}/{args.num_shards} wrote or skipped {len(units)} latent unit(s).")


def verify(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    errors = []
    required = [dataset_dir / "empty_emb.pt", action_stat_path(dataset_dir)]
    for path in required:
        if not path.exists():
            errors.append({"stage": "required_file", "path": str(path), "error": "missing"})

    loaded = 0
    for episode in tqdm(episodes, desc="verify calvin"):
        episode_index = int(episode["episode_index"])
        parquet_path = data_path(dataset_dir, info, episode_index)
        if not parquet_path.exists():
            errors.append({"stage": "parquet_exists", "episode_index": episode_index, "path": str(parquet_path)})
            continue
        for segment in episode.get("action_config", []):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if not (0 <= start < end <= int(episode["length"])):
                errors.append({"stage": "action_config", "episode_index": episode_index, "segment": segment})
            for camera_key in args.camera_keys:
                path = latent_path(dataset_dir, info, episode_index, camera_key, start, end)
                if not path.exists():
                    errors.append({"stage": "latent_exists", "path": str(path)})
                    continue
                if args.deep_limit < 0 or loaded < args.deep_limit:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    for key in ["latent", "text_emb", "frame_ids", "start_frame", "end_frame", "fps"]:
                        if key not in payload:
                            errors.append({"stage": "latent_key", "path": str(path), "key": key})
                    loaded += 1

    if errors:
        error_path = dataset_dir / "meta" / "calvin_verify_errors.jsonl"
        write_jsonl(error_path, errors)
        raise RuntimeError(f"CALVIN verification failed with {len(errors)} error(s). Error log: {error_path}")
    print(f"Verification passed: {len(episodes)} episodes, deep-loaded {loaded} latent file(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--hf-endpoint", default=os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--wan22-path", default=DEFAULT_WAN22)
    parser.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAMERA_KEYS)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--min-sampled-frames", type=int, default=5)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-stat-episodes", type=int, default=0)
    parser.add_argument("--deep-limit", type=int, default=32)
    parser.add_argument(
        "--stage",
        choices=["all", "metadata", "stats", "latents", "verify"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    maybe_download(args)
    if args.stage in ("all", "metadata"):
        ensure_action_config(args)
    if args.stage in ("all", "stats"):
        compute_action_stats(args)
    if args.stage in ("all", "latents"):
        extract_latents(args)
    if args.stage in ("all", "verify"):
        verify(args)


if __name__ == "__main__":
    main()
