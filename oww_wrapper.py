#!/usr/bin/env python3
"""Bridge the repository pipeline to the NanoWakeWord training CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_backend_config(cfg: dict) -> dict:
    backend_cfg = dict(cfg)
    mode = str(cfg.get("mode", "wakeword")).lower()
    output_dir = Path(cfg["output_dir"])
    model_name = str(cfg["model_name"])
    model_dir = output_dir / model_name

    if mode == "vad":
        backend_cfg.setdefault("positive_data_path", str(model_dir / "positive_train"))
        backend_cfg.setdefault("negative_data_path", str(model_dir / "negative_train"))
        backend_cfg.setdefault("wake_word", "vad")

    return backend_cfg


def _nanowakeword_binary() -> str:
    binary = shutil.which("nanowakeword-train")
    if binary:
        return binary
    raise SystemExit(
        "nanowakeword-train was not found. Install dependencies with: "
        "pip install -r requirements.txt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--generate_clips", action="store_true")
    parser.add_argument("--augment_clips", action="store_true")
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_cfg = _load_config(Path(args.training_config))
    backend_cfg = _build_backend_config(source_cfg)

    backend_cfg["generate_clips"] = bool(args.generate_clips)
    backend_cfg["transform_clips"] = bool(args.augment_clips)
    backend_cfg["train_model"] = bool(args.train_model)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.dump(backend_cfg, handle, default_flow_style=False, sort_keys=False)
        temp_config = Path(handle.name)

    cmd = [_nanowakeword_binary(), "-c", str(temp_config)]

    try:
        subprocess.check_call(cmd)
    finally:
        temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
