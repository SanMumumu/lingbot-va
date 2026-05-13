#!/usr/bin/env python3
"""Shardable LingBot-VA latent extraction for converted RoboMME datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_DATASET_DIR = "/mnt/hwdata/wangsen/WAM/lingbot-va/DATA/robomme_lingbot"
DEFAULT_WAN22 = "/mnt/hwdata/wangsen/WAM/lingbot-va/CKPTS/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]
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


def validate_wan22_path(wan22_path: Path) -> None:
    missing = [rel for rel in REQUIRED_WAN22_FILES if not (wan22_path / rel).exists()]
    if missing:
        found = []
        if wan22_path.exists():
            found = sorted(str(p.relative_to(wan22_path)) for p in wan22_path.glob("*"))
        raise FileNotFoundError(
            f"--wan22-path must point to a Diffusers-format Wan2.2 directory. "
            f"Missing required file(s): {missing}. "
            f"Given path: {wan22_path}. "
            f"Top-level entries found: {found[:30]}. "
            "Use Wan-AI/Wan2.2-TI2V-5B-Diffusers, not the original non-Diffusers checkpoint layout."
        )


def get_episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    return episode_index // int(info.get("chunks_size", 1000))


def resolve_video_path(dataset_dir: Path, info: dict[str, Any], episode_index: int, camera_key: str) -> Path:
    chunk = get_episode_chunk(info, episode_index)
    rel = info["video_path"].format(
        episode_chunk=chunk,
        video_key=camera_key,
        episode_index=episode_index,
    )
    return dataset_dir / rel


def resolve_latent_path(dataset_dir: Path, info: dict[str, Any], episode_index: int, camera_key: str, start: int, end: int) -> Path:
    chunk = get_episode_chunk(info, episode_index)
    return (
        dataset_dir
        / "latents"
        / f"chunk-{chunk:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start}_{end}.pth"
    )


def frame_ids_for_segment(start: int, end: int, ori_fps: int, target_fps: int) -> tuple[list[int], int]:
    if end <= start:
        return [], ori_fps
    stride = max(1, int(round(float(ori_fps) / float(target_fps))))
    frame_ids = list(range(start, end, stride))
    usable = ((len(frame_ids) - 1) // 4) * 4 + 1
    if usable >= 1:
        frame_ids = frame_ids[:usable]
    actual_fps = max(1, int(round(float(ori_fps) / stride)))
    return frame_ids, actual_fps


def normalize_max_sampled_frames(value: int) -> int:
    if value <= 0:
        return 0
    if value < 5:
        raise ValueError("--max-sampled-frames must be 0 or >= 5")
    return ((int(value) - 1) // 4) * 4 + 1


def read_video_frames(video_path: Path, frame_ids: list[int], height: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    target_ids = set(int(v) for v in frame_ids)
    max_id = max(target_ids)
    found: dict[int, np.ndarray] = {}
    frames = []
    try:
        for frame_idx in range(max_id + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(
                    f"OpenCV reached EOF before frame {max_id} in {video_path}; "
                    f"last_read={frame_idx - 1}"
                )
            if frame_idx not in target_ids:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != height or rgb.shape[1] != width:
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            found[frame_idx] = rgb.astype(np.uint8, copy=False)
    finally:
        cap.release()

    missing = [int(v) for v in frame_ids if int(v) not in found]
    if missing:
        raise RuntimeError(f"Missing requested frames {missing[:10]} from {video_path}")
    frames = [found[int(v)] for v in frame_ids]
    if not frames:
        raise RuntimeError(f"No frames selected for {video_path}")
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

    # Wan's temporal VAE encoder is causal. It must be called as one first
    # frame followed by 4-frame chunks while preserving the conv cache. Calling
    # the encoder on a long sampled clip directly makes temporal downsample
    # branches produce different lengths (for example 81 vs 41 frames).
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
    latent = latent_5d[0]  # [C, F, H, W]
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
        tasks = episode.get("tasks") or ["robomme task"]
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


def extract(args: argparse.Namespace) -> None:
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    dataset_dir = Path(args.dataset_dir)
    info = load_json(dataset_dir / "meta" / "info.json")
    episodes = load_jsonl(dataset_dir / "meta" / "episodes.jsonl")
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
    progress = tqdm(units, desc=f"latents shard {args.shard_index}/{args.num_shards}")
    for unit in progress:
        output_path = resolve_latent_path(
            dataset_dir,
            info,
            unit["episode_index"],
            unit["camera_key"],
            unit["start"],
            unit["end"],
        )
        try:
            if output_path.exists() and args.resume:
                continue

            frame_ids, actual_fps = frame_ids_for_segment(unit["start"], unit["end"], ori_fps, args.target_fps)
            if len(frame_ids) < args.min_sampled_frames:
                raise RuntimeError(
                    f"segment too short after fps sampling: start={unit['start']} end={unit['end']} "
                    f"ori_fps={ori_fps} target_fps={args.target_fps} sampled={len(frame_ids)}"
                )
            if max_sampled_frames and len(frame_ids) > max_sampled_frames:
                raise RuntimeError(
                    f"segment too long for one Wan VAE encode: start={unit['start']} end={unit['end']} "
                    f"ori_fps={ori_fps} target_fps={args.target_fps} sampled={len(frame_ids)} "
                    f"max_sampled_frames={max_sampled_frames}. "
                    "Run tools/robomme/split_robomme_action_config.py first or lower "
                    "MAX_SAMPLED_FRAMES_PER_LATENT in script/robomme_prepare_dataset.sh."
                )

            text_emb = encode_text(unit["text"], tokenizer, text_encoder, device, dtype, args.max_sequence_length)
            video_path = resolve_video_path(dataset_dir, info, unit["episode_index"], unit["camera_key"])
            frames = read_video_frames(video_path, frame_ids, args.height, args.width)
            latent = encode_video_segment(frames, streaming_vae, device, dtype)

            expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
            if latent.shape[2] != expected_latent_frames:
                raise RuntimeError(
                    f"VAE latent frame mismatch: got={latent.shape[2]} expected={expected_latent_frames} "
                    f"sampled_frames={len(frame_ids)} video={video_path}"
                )

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
        error_path = dataset_dir / "meta" / f"latent_errors_shard{args.shard_index:03d}.jsonl"
        write_jsonl(error_path, failures)
        preview = "\n".join(
            f"  - episode={f['unit']['episode_index']} camera={f['unit']['camera_key']} "
            f"start={f['unit']['start']} end={f['unit']['end']} "
            f"{f['error_type']}: {f['error_message']}"
            for f in failures[:10]
        )
        raise RuntimeError(f"{len(failures)} latent unit(s) failed. Error log: {error_path}\n{preview}")

    print(f"Shard {args.shard_index}/{args.num_shards} wrote {len(units)} latent unit(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--wan22-path", default=DEFAULT_WAN22)
    parser.add_argument("--camera-keys", nargs="+", default=DEFAULT_CAMERA_KEYS)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--min-sampled-frames", type=int, default=5)
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Maximum sampled RGB frames per VAE encode (0 = no guard, default). "
             "Set to a positive 4n+1 value if you ran "
             "tools/robomme/split_robomme_action_config.py with the same value.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    extract(parse_args())
