from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import get_episode_chunk, read_json, read_jsonl, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image)


def build_frame_ids(
    start_frame: int,
    end_frame: int,
    ori_fps: float,
    target_fps: float,
) -> tuple[list[int], int]:
    if end_frame - start_frame < 2:
        raise ValueError(
            f"Segment [{start_frame}, {end_frame}) is too short. Need at least 2 source frames."
        )
    frame_stride = max(int(round(ori_fps / target_fps)), 1)
    frame_ids = list(range(start_frame, end_frame, frame_stride))
    if len(frame_ids) < 2:
        frame_ids = [start_frame, end_frame - 1]
    sampled_fps = max(int(round(ori_fps / frame_stride)), 1)
    return frame_ids, sampled_fps


def load_sampled_frames(
    video_path: Path,
    frame_ids: list[int],
    width: int,
    height: int,
) -> torch.Tensor:
    reader = imageio.get_reader(video_path, format="ffmpeg")
    frames: list[np.ndarray] = []
    try:
        for frame_id in frame_ids:
            frame = reader.get_data(frame_id)
            frames.append(resize_frame(frame, width=width, height=height))
    finally:
        reader.close()

    frame_array = np.stack(frames, axis=0)
    return torch.from_numpy(frame_array)


class DiffusersWanEncoder:
    def __init__(
        self,
        model_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> None:
        from diffusers import AutoencoderKLWan
        from transformers import T5TokenizerFast, UMT5EncoderModel

        from wan_va.modules.utils import WanVAEStreamingWrapper

        self.device = device
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length

        vae_path = model_root / "vae"
        text_encoder_path = model_root / "text_encoder"
        tokenizer_path = model_root / "tokenizer"
        if not vae_path.exists() or not text_encoder_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                "Diffusers backend expects model_root/{vae,text_encoder,tokenizer}. "
                f"Got model_root={model_root}"
            )

        self.vae = AutoencoderKLWan.from_pretrained(str(vae_path), torch_dtype=dtype).to(device)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            str(text_encoder_path),
            torch_dtype=dtype,
        ).to(device)
        self.tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))

    def encode_text(self, text: str) -> torch.Tensor:
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids,
                attention_mask,
            ).last_hidden_state[0]
        seq_len = int(attention_mask[0].sum().item())
        output = hidden_states.new_zeros(
            (self.max_sequence_length, hidden_states.shape[-1]),
            dtype=self.dtype,
        )
        output[:seq_len] = hidden_states[:seq_len].to(self.dtype)
        return output.detach().cpu()

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(self.device, dtype=torch.float32)
        video = video / 255.0 * 2.0 - 1.0
        with torch.no_grad():
            enc_out = self.streaming_vae.encode_chunk(video.to(self.dtype))
            mu, _ = torch.chunk(enc_out, 2, dim=1)
            latents_mean = torch.tensor(
                self.vae.config.latents_mean,
                device=mu.device,
                dtype=torch.float32,
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                self.vae.config.latents_std,
                device=mu.device,
                dtype=torch.float32,
            ).view(1, -1, 1, 1, 1)
            mu_norm = ((mu.float() - latents_mean) * (1.0 / latents_std)).to(self.dtype)
        return mu_norm[0].detach().cpu()


class OfficialWanEncoder:
    def __init__(
        self,
        model_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
        wan_code_root: Path | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.model_root = model_root

        code_root = (wan_code_root or model_root).resolve()
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))

        self.t5_cls = self._import_symbol("wan.modules.t5", "T5EncoderModel")
        self.vae = self._build_vae(model_root=model_root)
        self.text_encoder = self._build_text_encoder(model_root=model_root)

    def _import_symbol(self, module_name: str, symbol_name: str) -> Any:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)

    def _find_tokenizer_path(self, model_root: Path) -> Path:
        candidates = [
            model_root / "google" / "umt5-xxl",
            model_root / "google" / "umt5-xxl" / "tokenizer",
            model_root / "tokenizer",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not locate tokenizer files under the official Wan model root. "
            f"Tried: {candidates}"
        )

    def _build_text_encoder(self, model_root: Path) -> Any:
        text_ckpt = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
        if not text_ckpt.exists():
            raise FileNotFoundError(f"Missing official Wan text encoder checkpoint: {text_ckpt}")
        tokenizer_path = self._find_tokenizer_path(model_root)
        return self.t5_cls(
            text_len=self.max_sequence_length,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=str(text_ckpt),
            tokenizer_path=str(tokenizer_path),
        )

    def _build_vae(self, model_root: Path) -> Any:
        vae_ckpt = model_root / "Wan2.2_VAE.pth"
        if not vae_ckpt.exists():
            raise FileNotFoundError(f"Missing official Wan VAE checkpoint: {vae_ckpt}")

        constructors = [
            (
                "wan.modules.vae",
                "WanVAE",
                {
                    "vae_path": str(vae_ckpt),
                    "device": self.device,
                    "dtype": self.dtype,
                    "cache_device": "cpu",
                },
            ),
            (
                "wan.modules.vae2_2",
                "Wan2_2_VAE",
                {
                    "vae_pth": str(vae_ckpt),
                    "device": self.device,
                },
            ),
        ]
        errors: list[str] = []
        for module_name, class_name, kwargs in constructors:
            try:
                cls = self._import_symbol(module_name, class_name)
                return cls(**kwargs)
            except Exception as exc:
                errors.append(f"{module_name}.{class_name}: {exc}")
        raise RuntimeError(
            "Failed to initialize the official Wan VAE wrapper. "
            f"Errors: {errors}"
        )

    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.text_encoder([text], self.device)
        if isinstance(encoded, (list, tuple)):
            encoded = encoded[0]
        if encoded.ndim == 3:
            encoded = encoded[0]
        if encoded.ndim != 2:
            raise ValueError(f"Unexpected official Wan text embedding shape: {tuple(encoded.shape)}")
        seq_len = min(encoded.shape[0], self.max_sequence_length)
        output = encoded.new_zeros((self.max_sequence_length, encoded.shape[-1]), dtype=self.dtype)
        output[:seq_len] = encoded[:seq_len].to(self.dtype)
        return output.detach().cpu()

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        video = frames.permute(3, 0, 1, 2).to(torch.float32)
        video = video / 255.0 * 2.0 - 1.0
        with torch.no_grad():
            if hasattr(self.vae, "encode"):
                latents = self.vae.encode([video.to(self.device)])[0]
                if latents.ndim == 5:
                    latents = latents[0]
                return latents.to(self.dtype).detach().cpu()

            if hasattr(self.vae, "model") and hasattr(self.vae.model, "encode"):
                enc_out = self.vae.model.encode(video.unsqueeze(0).to(self.device))
                if isinstance(enc_out, (list, tuple)):
                    enc_out = enc_out[0]
                mu, _ = torch.chunk(enc_out, 2, dim=1)
                latents = mu[0]
                mean = torch.as_tensor(
                    getattr(self.vae, "mean", 0.0),
                    device=latents.device,
                    dtype=torch.float32,
                ).view(-1, 1, 1, 1)
                std = torch.as_tensor(
                    getattr(self.vae, "std", 1.0),
                    device=latents.device,
                    dtype=torch.float32,
                ).view(-1, 1, 1, 1)
                scale = torch.as_tensor(
                    getattr(self.vae, "scale", 1.0),
                    device=latents.device,
                    dtype=torch.float32,
                ).view(-1, 1, 1, 1)
                latents = scale * (latents.float() - mean) / std
                return latents.to(self.dtype).detach().cpu()

        raise RuntimeError("Official Wan VAE wrapper does not expose a usable encode path.")


def build_encoder(
    backend: str,
    model_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    wan_code_root: Path | None,
) -> Any:
    if backend == "diffusers":
        return DiffusersWanEncoder(
            model_root=model_root,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
    if backend == "official":
        return OfficialWanEncoder(
            model_root=model_root,
            wan_code_root=wan_code_root,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def flatten_latents(latents: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if latents.ndim != 4:
        raise ValueError(f"Expected latent tensor [C, F, H, W], got {tuple(latents.shape)}")
    channels, num_frames, height, width = latents.shape
    flat = latents.permute(1, 2, 3, 0).reshape(num_frames * height * width, channels)
    return flat.contiguous(), int(num_frames), int(height), int(width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LingBot-VA training latents from a Franka LeRobot dataset using Wan2.2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Converted LeRobot dataset root containing meta/, videos/, and data/.",
    )
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        required=True,
        help="Wan2.2 checkpoint root.",
    )
    parser.add_argument(
        "--wan-backend",
        choices=("diffusers", "official"),
        default="official",
        help=(
            "Use 'diffusers' when wan-model-root already contains vae/, text_encoder/, tokenizer/. "
            "Use 'official' for the original Wan2.2 checkpoint layout."
        ),
    )
    parser.add_argument(
        "--wan-code-root",
        type=Path,
        default=None,
        help=(
            "Only used for --wan-backend official. "
            "Point this to the Wan2.2 source tree if it is separate from wan-model-root."
        ),
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target video sampling fps before VAE encoding.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Resized frame height before VAE encoding.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Resized frame width before VAE encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for VAE/text encoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="Torch dtype used during encoding.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Text embedding length used for action_text and empty prompt embedding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent files and empty_emb.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_root = args.wan_model_root.resolve()
    wan_code_root = args.wan_code_root.resolve() if args.wan_code_root else None

    info = read_json(dataset_root / "meta" / "info.json")
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    franka_meta = read_json(dataset_root / "meta" / "franka_meta.json", default={}) or {}
    if not info or not episodes:
        raise FileNotFoundError(
            f"Dataset root {dataset_root} is missing meta/info.json or meta/episodes.jsonl"
        )

    camera_keys = franka_meta.get("obs_cam_keys") or [
        key for key, spec in info["features"].items() if spec.get("dtype") == "video"
    ]
    if not camera_keys:
        raise ValueError("No video camera keys found in dataset metadata.")

    if any("action_config" not in episode for episode in episodes):
        raise ValueError(
            "episodes.jsonl is missing action_config. Run add_action_config.py first."
        )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    encoder = build_encoder(
        backend=args.wan_backend,
        model_root=model_root,
        wan_code_root=wan_code_root,
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_sequence_length,
    )

    text_cache: dict[str, torch.Tensor] = {}

    empty_emb_path = dataset_root / "empty_emb.pt"
    if args.overwrite or not empty_emb_path.exists():
        empty_embedding = encoder.encode_text("")
        torch.save(empty_embedding.to(torch.bfloat16).cpu(), empty_emb_path)

    ori_fps = float(info["fps"])
    for episode in tqdm(episodes, desc="Encoding action segments"):
        episode_index = int(episode["episode_index"])
        episode_chunk = get_episode_chunk(episode_index, int(info["chunks_size"]))
        for segment in episode["action_config"]:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            action_text = str(segment["action_text"])
            frame_ids, sampled_fps = build_frame_ids(
                start_frame=start_frame,
                end_frame=end_frame,
                ori_fps=ori_fps,
                target_fps=args.target_fps,
            )

            if action_text not in text_cache:
                text_cache[action_text] = encoder.encode_text(action_text).to(torch.bfloat16).cpu()
            text_emb = text_cache[action_text]

            for camera_key in camera_keys:
                video_path = (
                    dataset_root
                    / "videos"
                    / f"chunk-{episode_chunk:03d}"
                    / camera_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing source video: {video_path}")

                latent_path = (
                    dataset_root
                    / "latents"
                    / f"chunk-{episode_chunk:03d}"
                    / camera_key
                    / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )
                if latent_path.exists() and not args.overwrite:
                    continue

                frames = load_sampled_frames(
                    video_path=video_path,
                    frame_ids=frame_ids,
                    width=args.width,
                    height=args.height,
                )
                latents = encoder.encode_video(frames).to(torch.bfloat16).cpu()
                latent_flat, latent_num_frames, latent_height, latent_width = flatten_latents(latents)

                payload = {
                    "latent": latent_flat,
                    "latent_num_frames": latent_num_frames,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                    "video_num_frames": int(len(frame_ids)),
                    "video_height": int(args.height),
                    "video_width": int(args.width),
                    "text_emb": text_emb,
                    "text": action_text,
                    "frame_ids": [int(frame_id) for frame_id in frame_ids],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": int(sampled_fps),
                    "ori_fps": float(ori_fps),
                }
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, latent_path)

    write_json(
        dataset_root / "meta" / "latent_config.json",
        {
            "wan_backend": args.wan_backend,
            "wan_model_root": str(model_root),
            "wan_code_root": str(wan_code_root) if wan_code_root else None,
            "obs_cam_keys": camera_keys,
            "height": int(args.height),
            "width": int(args.width),
            "target_fps": float(args.target_fps),
            "dtype": args.dtype,
            "max_sequence_length": int(args.max_sequence_length),
        },
    )

    print(f"Latent extraction finished for dataset: {dataset_root}")
    print(f"Saved empty prompt embedding to: {empty_emb_path}")
    print(
        "Next step: export LINGBOT_FRANKA_DATASET_PATH and start training with "
        "config-name franka_single_arm_train"
    )


if __name__ == "__main__":
    main()
