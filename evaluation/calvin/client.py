#!/usr/bin/env python3
"""Evaluate LingBot-VA with the official CALVIN long-horizon protocol."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy


def add_calvin_to_path(calvin_root: str | None) -> None:
    if not calvin_root:
        return
    root = Path(calvin_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "calvin_models"))


def to_uint8_hwc(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.size and arr.max() <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return np.ascontiguousarray(arr[:, :, :3])


def extract_lingbot_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    rgb_obs = obs.get("rgb_obs", obs)

    def first_present(*values):
        for value in values:
            if value is not None:
                return value
        return None

    top = first_present(
        rgb_obs.get("rgb_static"),
        rgb_obs.get("static"),
        obs.get("observation.images.top"),
        obs.get("rgb_static"),
    )
    wrist = first_present(
        rgb_obs.get("rgb_gripper"),
        rgb_obs.get("gripper"),
        obs.get("observation.images.wrist"),
        obs.get("rgb_gripper"),
    )
    if top is None or wrist is None:
        raise KeyError(f"Could not find CALVIN rgb_static/rgb_gripper images in obs keys: {sorted(obs.keys())}")
    return {
        "observation.images.top": to_uint8_hwc(top),
        "observation.images.wrist": to_uint8_hwc(wrist),
    }


class LingBotVACalvinModel:
    """Small adapter matching CALVIN's CustomModel reset/step interface."""

    def __init__(self, port: int, action_per_frame: int = 4, save_visualization: bool = False):
        self.policy = WebsocketClientPolicy(port=port)
        self.action_per_frame = int(action_per_frame)
        self.save_visualization = save_visualization
        self.current_prompt = ""
        self.needs_server_reset = True
        self.first_prediction = True
        self.action_queue: list[np.ndarray] = []
        self.pending_key_frames: list[dict[str, np.ndarray]] = []
        self.last_action_tensor = None
        self.steps_since_generation = 0

    def reset(self) -> None:
        self.needs_server_reset = True
        self.first_prediction = True
        self.action_queue = []
        self.pending_key_frames = []
        self.last_action_tensor = None
        self.steps_since_generation = 0

    @staticmethod
    def _as_action_list(action: Any, first_prediction: bool) -> list[np.ndarray]:
        arr = np.asarray(action)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        actions = []
        if arr.ndim == 3 and arr.shape[0] <= 30:
            start_i = 1 if first_prediction and arr.shape[1] > 1 else 0
            for i in range(start_i, arr.shape[1]):
                for j in range(arr.shape[2]):
                    actions.append(arr[:7, i, j])
        elif arr.ndim == 3 and arr.shape[-1] <= 30:
            start_i = 1 if first_prediction and arr.shape[0] > 1 else 0
            for i in range(start_i, arr.shape[0]):
                for j in range(arr.shape[1]):
                    actions.append(arr[i, j, :7])
        elif arr.ndim == 2:
            actions = [row[:7] for row in arr]
        elif arr.ndim == 1:
            actions = [arr[:7]]
        else:
            raise ValueError(f"Unsupported LingBot-VA action shape: {arr.shape}")

        if not actions:
            raise RuntimeError(f"LingBot-VA returned no executable actions from shape {arr.shape}")
        return [np.asarray(a, dtype=np.float32) for a in actions]

    def _reset_server_if_needed(self, prompt: str) -> None:
        if not self.needs_server_reset and prompt == self.current_prompt:
            return
        self.policy.infer(dict(reset=True, prompt=prompt))
        self.current_prompt = prompt
        self.needs_server_reset = False

    def _update_kv_cache_if_needed(self, prompt: str) -> None:
        if not self.pending_key_frames or self.last_action_tensor is None:
            return
        self.policy.infer(
            dict(
                obs=self.pending_key_frames,
                prompt=prompt,
                compute_kv_cache=True,
                imagine=False,
                state=self.last_action_tensor,
                save_visualization=self.save_visualization,
                task_name="calvin",
            )
        )
        self.pending_key_frames = []

    def _request_actions(self, obs: dict[str, np.ndarray], prompt: str) -> None:
        self._update_kv_cache_if_needed(prompt)
        ret = self.policy.infer(
            dict(
                obs=obs,
                prompt=prompt,
                save_visualization=self.save_visualization,
                task_name="calvin",
            )
        )
        self.last_action_tensor = ret["action"]
        self.action_queue = self._as_action_list(ret["action"], self.first_prediction)
        self.first_prediction = False
        self.steps_since_generation = 0

    def step(self, obs: dict[str, Any], goal: Any) -> np.ndarray:
        prompt = str(goal)
        lingbot_obs = extract_lingbot_obs(obs)
        self._reset_server_if_needed(prompt)

        if self.steps_since_generation > 0 and self.steps_since_generation % self.action_per_frame == 0:
            self.pending_key_frames.append(lingbot_obs)

        if not self.action_queue:
            self._request_actions(lingbot_obs, prompt)

        action = self.action_queue.pop(0)
        self.steps_since_generation += 1
        return action


def run(args: argparse.Namespace) -> None:
    add_calvin_to_path(args.calvin_root)
    eval_mod = importlib.import_module("calvin_agent.evaluation.evaluate_policy")

    if args.num_sequences > 0:
        eval_mod.NUM_SEQUENCES = args.num_sequences
    if args.ep_len > 0:
        eval_mod.EP_LEN = args.ep_len

    env = eval_mod.make_env(args.dataset_path)
    model = LingBotVACalvinModel(
        port=args.port,
        action_per_frame=args.action_per_frame,
        save_visualization=args.save_visualization,
    )
    eval_mod.evaluate_policy(
        model,
        env,
        epoch=args.epoch,
        eval_log_dir=args.out_dir,
        debug=args.debug,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Official CALVIN dataset root, e.g. .../task_ABC_D")
    parser.add_argument("--calvin-root", default=os.getenv("CALVIN_ROOT"), help="Path to the official mees/calvin checkout")
    parser.add_argument("--port", type=int, default=29057)
    parser.add_argument("--out-dir", default="outputs/calvin")
    parser.add_argument("--epoch", default="lingbot_va")
    parser.add_argument("--action-per-frame", type=int, default=4)
    parser.add_argument("--num-sequences", type=int, default=0, help="Override official NUM_SEQUENCES; 0 keeps default")
    parser.add_argument("--ep-len", type=int, default=0, help="Override official EP_LEN; 0 keeps default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
