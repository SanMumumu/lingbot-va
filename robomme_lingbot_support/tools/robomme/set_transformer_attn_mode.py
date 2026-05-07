#!/usr/bin/env python3
"""Set LingBot transformer config attn_mode for training or inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint root containing transformer/config.json")
    parser.add_argument("--mode", choices=["flex", "torch", "flashattn"], required=True)
    args = parser.parse_args()

    config_path = Path(args.ckpt) / "transformer" / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    old = data.get("attn_mode")
    data["attn_mode"] = args.mode
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{config_path}: attn_mode {old!r} -> {args.mode!r}")


if __name__ == "__main__":
    main()
