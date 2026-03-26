#!/usr/bin/env python3
"""Download official ESPHome micro_wake_word models for Voice PE."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR / "export"
MODEL_ROOT = "https://raw.githubusercontent.com/esphome/micro-wake-word-models/main/models/v2"


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"Already present: {dest}")
        return

    if force and dest.exists():
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
    print(f"Downloaded {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="vad",
        help="Official ESPHome micro_wake_word model name to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXPORT_DIR),
        help="Directory where the .json and .tflite files will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for suffix in (".json", ".tflite"):
        filename = f"{args.model}{suffix}"
        download(f"{MODEL_ROOT}/{filename}", output_dir / filename, force=args.force)


if __name__ == "__main__":
    main()
