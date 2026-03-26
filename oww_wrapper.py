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
    features_dir = model_dir / "features"

    if mode == "vad":
        positive_train = model_dir / "positive_train"
        positive_test = model_dir / "positive_test"
        negative_train = model_dir / "negative_train"
        negative_test = model_dir / "negative_test"

        backend_cfg["positive_data_path"] = str(positive_train)
        backend_cfg["negative_data_path"] = str(negative_train)
        backend_cfg.setdefault("wake_word", "vad")
        backend_cfg["feature_generation_manifest"] = {
            "positive_train_features": {
                "input_audio_dirs": [str(positive_train)],
                "output_filename": "positive_train.npy",
                "use_background_noise": True,
                "use_rir": True,
                "augmentation_rounds": int(cfg.get("augmentation_rounds", 1)),
            },
            "positive_test_features": {
                "input_audio_dirs": [str(positive_test)],
                "output_filename": "positive_test.npy",
                "use_background_noise": False,
                "use_rir": False,
                "augmentation_rounds": 1,
            },
            "negative_train_features": {
                "input_audio_dirs": [str(negative_train)],
                "output_filename": "negative_train.npy",
                "use_background_noise": False,
                "use_rir": False,
                "augmentation_rounds": 1,
            },
            "negative_test_features": {
                "input_audio_dirs": [str(negative_test)],
                "output_filename": "negative_test.npy",
                "use_background_noise": False,
                "use_rir": False,
                "augmentation_rounds": 1,
            },
        }

        background_paths = [str(p) for p in cfg.get("background_paths", []) if Path(p).exists()]
        if background_paths:
            backend_cfg["feature_generation_manifest"]["background_features"] = {
                "input_audio_dirs": background_paths,
                "output_filename": "background_noise.npy",
                "use_background_noise": False,
                "use_rir": False,
                "augmentation_rounds": 1,
            }

        feature_manifest = {
            "targets": {
                "t": str(features_dir / "positive_train.npy"),
            },
            "targets_val": {
                "t_v": str(features_dir / "positive_test.npy"),
            },
            "negatives": {
                "n": str(features_dir / "negative_train.npy"),
            },
            "negatives_val": {
                "n_v": str(features_dir / "negative_test.npy"),
            },
        }
        if background_paths:
            feature_manifest["negatives"]["b"] = str(features_dir / "background_noise.npy")
            feature_manifest["negatives_val"]["b_v"] = str(features_dir / "background_noise.npy")
        backend_cfg["feature_manifest"] = feature_manifest
        backend_cfg["batch_composition"] = {
            "t": 96,
            "n": 192,
            **({"b": 64} if background_paths else {}),
        }
        backend_cfg.pop("batch_n_per_class", None)

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
