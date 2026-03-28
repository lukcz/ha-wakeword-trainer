#!/usr/bin/env python3
"""Train a Home Assistant Voice PE wake word or VAD model with microWakeWord."""

from __future__ import annotations

import argparse
import tarfile
import hashlib
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

import requests
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
THIRD_PARTY_DIR = SCRIPT_DIR / "third_party"
MWW_DIR = THIRD_PARTY_DIR / "micro-wake-word"
PIPER_MODELS_DIR = THIRD_PARTY_DIR / "piper-models"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
EXPORT_DIR = SCRIPT_DIR / "export"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "microwakeword_example.yaml"
TENSORBOARD_PIP_SPEC = "tensorboard>=2.20.0,<2.21.0"

CONFIG_FILE = DEFAULT_CONFIG

NEGATIVE_FEATURE_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"
DEFAULT_PIPER_MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
    "en_US-libritts_r-medium.pt"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_microwakeword")


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def _download(
    url: str,
    dest: Path,
    description: str,
    force: bool = False,
    retries: int = 5,
) -> None:
    if dest.exists() and not force:
        log.info("  Already present: %s", dest)
        return

    if force and dest.exists():
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        if tmp.exists():
            tmp.unlink()

        log.info("  Downloading %s (attempt %d/%d)", description, attempt, retries)
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dest)
            return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
            if attempt >= retries or not retryable:
                raise
            sleep_s = min(60, 2 ** (attempt - 1))
            log.warning("  Temporary HTTP failure while downloading %s: %s. Retrying in %ss.", description, exc, sleep_s)
            time.sleep(sleep_s)
        except requests.RequestException as exc:
            if attempt >= retries:
                raise
            sleep_s = min(60, 2 ** (attempt - 1))
            log.warning("  Temporary network failure while downloading %s: %s. Retrying in %ss.", description, exc, sleep_s)
            time.sleep(sleep_s)


def _ensure_python_module(module_name: str, pip_requirement: str) -> bool:
    if importlib.util.find_spec(module_name) is not None:
        return True

    log.warning("  Python module '%s' is missing. Installing %s", module_name, pip_requirement)
    try:
        _run([sys.executable, "-m", "pip", "install", pip_requirement])
    except subprocess.CalledProcessError:
        log.error("  Automatic installation failed for %s", pip_requirement)
        return False

    if importlib.util.find_spec(module_name) is None:
        log.error("  Module '%s' is still missing after installation", module_name)
        return False

    return True


def _patch_microwakeword_train_py() -> bool:
    train_py = MWW_DIR / "microwakeword" / "train.py"
    if not train_py.exists():
        log.error("  microWakeWord training module not found at %s", train_py)
        return False

    original = train_py.read_text(encoding="utf-8")
    updated = original

    helper = """
def _as_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


"""

    marker = "def validate_nonstreaming(config, data_processor, model, test_set):\n"
    if "_as_numpy(value):" not in updated:
        if marker not in updated:
            log.error("  Could not patch microWakeWord train.py: validate_nonstreaming marker not found")
            return False
        updated = updated.replace(marker, helper + marker, 1)

    replacements = {
        'result["fp"].numpy()': '_as_numpy(result["fp"])',
        'ambient_predictions["tp"].numpy()': '_as_numpy(ambient_predictions["tp"])',
        'ambient_predictions["fp"].numpy()': '_as_numpy(ambient_predictions["fp"])',
        'ambient_predictions["fn"].numpy()': '_as_numpy(ambient_predictions["fn"])',
        "np.trapz(": "np.trapezoid(",
    }
    for old, new in replacements.items():
        updated = updated.replace(old, new)

    if updated == original:
        return True

    train_py.write_text(updated, encoding="utf-8")
    log.info("  Patched microWakeWord metric compatibility in %s", train_py)
    return True


def _patch_microwakeword_test_py() -> bool:
    test_py = MWW_DIR / "microwakeword" / "test.py"
    if not test_py.exists():
        log.error("  microWakeWord test module not found at %s", test_py)
        return False

    original = test_py.read_text(encoding="utf-8")
    updated = original.replace("np.trapz(", "np.trapezoid(")

    if updated == original:
        return True

    test_py.write_text(updated, encoding="utf-8")
    log.info("  Patched microWakeWord test compatibility in %s", test_py)
    return True


def _load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _task(cfg: dict) -> str:
    return str(cfg.get("task", "wakeword")).strip().lower()


def _project_dir(cfg: dict) -> Path:
    return OUTPUT_DIR / cfg["model_name"]


def _generated_samples_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "generated_samples"


def _positive_features_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "generated_augmented_features"


def _background_negative_features_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "generated_background_negative_features"


def _negative_datasets_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "negative_datasets"


def _staged_positive_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "staged_positive_audio"


def _staged_background_dir(cfg: dict) -> Path:
    return _project_dir(cfg) / "staged_background_audio"


def _training_dir(cfg: dict) -> Path:
    raw = cfg.get("training", {}).get("train_dir") or f"output/{cfg['model_name']}/trained_model"
    path = Path(raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def _iter_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]


def _safe_iter_audio_files(root: Path) -> list[Path]:
    try:
        return _iter_audio_files(root)
    except OSError as exc:
        log.warning("  Failed to scan audio files in %s: %s", root, exc)
        return []


def _dir_has_entries(path: Path) -> bool:
    try:
        return path.exists() and any(path.iterdir())
    except OSError as exc:
        log.warning("  Directory check failed for %s: %s", path, exc)
        return False


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_augmentation_probabilities(raw_probabilities: dict) -> dict:
    try:
        import audiomentations
    except Exception:
        return dict(raw_probabilities)

    support_checks = {
        "RIR": "ApplyImpulseResponse",
    }

    resolved = {}
    missing = []
    for name, probability in (raw_probabilities or {}).items():
        attr_name = support_checks.get(name, name)
        if hasattr(audiomentations, attr_name):
            resolved[name] = probability
        else:
            missing.append(name)

    if missing:
        log.warning(
            "  Skipping unsupported audiomentations transforms: %s",
            ", ".join(sorted(missing)),
        )

    return resolved


def _runtime_cfg(cfg: dict) -> dict:
    return cfg.get("runtime", {}) or {}


def _append_env_flag(env: dict[str, str], key: str, flag: str) -> None:
    current = str(env.get(key, "")).strip()
    flags = current.split() if current else []
    if flag not in flags:
        flags.append(flag)
    env[key] = " ".join(flags)


def _build_training_env(cfg: dict, force_device: str | None = None) -> tuple[dict[str, str], str]:
    runtime = _runtime_cfg(cfg)
    env = os.environ.copy()

    device = str(force_device or runtime.get("device", "auto")).strip().lower()
    if device not in {"auto", "gpu", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, gpu, cpu")

    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    if bool(runtime.get("disable_xla_auto_jit", True)):
        _append_env_flag(env, "TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

    if not bool(runtime.get("xla_gpu_strict_conv_algorithm_picker", False)):
        _append_env_flag(env, "XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")

    if "tf_enable_onednn_opts" in runtime:
        env["TF_ENABLE_ONEDNN_OPTS"] = "1" if bool(runtime["tf_enable_onednn_opts"]) else "0"

    intra_threads = runtime.get("intra_op_threads")
    if intra_threads:
        env["TF_NUM_INTRAOP_THREADS"] = str(int(intra_threads))

    inter_threads = runtime.get("inter_op_threads")
    if inter_threads:
        env["TF_NUM_INTEROP_THREADS"] = str(int(inter_threads))

    return env, device


def _check_micro_wake_word_import() -> bool:
    try:
        import microwakeword  # noqa: F401
        import mmap_ninja  # noqa: F401
        return True
    except Exception as exc:
        log.error("  microWakeWord dependencies are not installed: %s", exc)
        log.error("  Run ./setup_environment.sh first.")
        return False


def _needs_piper(cfg: dict) -> bool:
    return _task(cfg) == "wakeword" and not str(cfg.get("positive_dataset_path", "")).strip()


def _ensure_piper_assets(cfg: dict) -> Path:
    model_filename = cfg.get("piper_model_filename", "en_US-libritts_r-medium.pt")
    model_url = cfg.get("piper_model_url", DEFAULT_PIPER_MODEL_URL)
    model_path = PIPER_MODELS_DIR / model_filename
    _download(model_url, model_path, f"Piper model {model_filename}")

    for extra in cfg.get("piper_extra_downloads", []) or []:
        if not isinstance(extra, dict) or "url" not in extra or "filename" not in extra:
            raise ValueError("Each entry in piper_extra_downloads must have url and filename")
        extra_path = PIPER_MODELS_DIR / str(extra["filename"])
        _download(str(extra["url"]), extra_path, f"Piper asset {extra['filename']}")

    return model_path


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _resolve_io_workers(cfg: dict | None = None, default: int = 4) -> int:
    if not cfg:
        return default
    return max(1, int(cfg.get("io_workers", default)))


def _resolve_bootstrap_workers(cfg: dict | None = None, default: int = 3) -> int:
    if not cfg:
        return default
    if "bootstrap_workers" in cfg:
        return max(1, int(cfg["bootstrap_workers"]))
    asset_cfg = cfg.get("asset_subsets", {}) or {}
    if "bootstrap_workers" in asset_cfg:
        return max(1, int(asset_cfg["bootstrap_workers"]))
    return default


def _write_audio_file(dest: Path, samples, sampling_rate: int) -> None:
    import soundfile as sf

    sf.write(dest, samples, sampling_rate)


def _segment_audio_samples(
    samples,
    sampling_rate: int,
    segment_duration_s: float | None,
    segment_overlap_s: float = 0.0,
    min_segment_duration_s: float | None = None,
):
    import numpy as np

    if not segment_duration_s or segment_duration_s <= 0:
        yield np.asarray(samples, dtype="float32")
        return

    total = len(samples)
    segment_len = max(1, int(segment_duration_s * sampling_rate))
    min_len = int((min_segment_duration_s or segment_duration_s) * sampling_rate)
    overlap_len = max(0, int(segment_overlap_s * sampling_rate))
    step = max(1, segment_len - overlap_len)

    if total <= segment_len:
        if total >= min_len:
            yield np.asarray(samples, dtype="float32")
        return

    start = 0
    while start + min_len <= total:
        end = min(total, start + segment_len)
        chunk = np.asarray(samples[start:end], dtype="float32")
        if len(chunk) >= min_len:
            yield chunk
        if end >= total:
            break
        start += step


def _write_dataset_audio(
    dataset,
    output_dir: Path,
    prefix: str,
    max_clips: int | None,
    io_workers: int,
    audio_column: str = "audio",
    segment_duration_s: float | None = None,
    segment_overlap_s: float = 0.0,
    min_segment_duration_s: float | None = None,
) -> int:
    import numpy as np

    count = 0
    futures = set()

    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as executor:
        for row in dataset:
            if max_clips is not None and count >= max_clips:
                break

            try:
                samples, sampling_rate = _extract_row_audio(row, audio_column)
            except Exception:
                continue

            for chunk in _segment_audio_samples(
                samples,
                sampling_rate,
                segment_duration_s=segment_duration_s,
                segment_overlap_s=segment_overlap_s,
                min_segment_duration_s=min_segment_duration_s,
            ):
                if max_clips is not None and count >= max_clips:
                    break

                dest = output_dir / f"{prefix}_{count:06d}.wav"
                futures.add(executor.submit(_write_audio_file, dest, chunk, sampling_rate))
                count += 1

            if len(futures) >= io_workers * 4:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()

        for future in futures:
            future.result()

    return count


def _extract_row_audio(row: dict, audio_column: str) -> tuple["np.ndarray", int]:
    import numpy as np
    import soundfile as sf

    audio = row[audio_column]
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        samples = np.asarray(audio["array"], dtype="float32")
        sampling_rate = int(audio["sampling_rate"])
    elif isinstance(audio, (bytes, bytearray)):
        samples, sampling_rate = sf.read(io.BytesIO(audio), dtype="float32")
        samples = np.asarray(samples, dtype="float32")
    else:
        raise ValueError(f"Unsupported audio payload type for column '{audio_column}'")

    if samples.ndim > 1:
        samples = np.mean(samples, axis=-1, dtype="float32")

    return np.asarray(samples, dtype="float32"), sampling_rate


def _normalize_filter_values(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return {str(value)}


def _row_matches_dataset_filters(row: dict, dataset_cfg: dict) -> bool:
    filter_column = dataset_cfg.get("filter_column")
    if filter_column:
        actual = str(row.get(str(filter_column), ""))
        allow_values = _normalize_filter_values(dataset_cfg.get("filter_allow_values"))
        deny_values = _normalize_filter_values(dataset_cfg.get("filter_deny_values"))
        if allow_values and actual not in allow_values:
            return False
        if deny_values and actual in deny_values:
            return False

    path_column = str(dataset_cfg.get("path_column", ""))
    if path_column:
        actual_path = str(row.get(path_column, ""))
        must_contain = _normalize_filter_values(dataset_cfg.get("path_must_contain"))
        deny_contains = _normalize_filter_values(dataset_cfg.get("path_deny_contains"))
        if must_contain and not any(fragment in actual_path for fragment in must_contain):
            return False
        if deny_contains and any(fragment in actual_path for fragment in deny_contains):
            return False

    return True


def _iter_filtered_dataset(dataset, dataset_cfg: dict):
    for row in dataset:
        if _row_matches_dataset_filters(row, dataset_cfg):
            yield row


def _download_hf_audio_dataset(dataset_cfg: dict) -> None:
    from datasets import load_dataset

    repo = str(dataset_cfg["hf_repo"])
    output_dir = _resolve_path(str(dataset_cfg["output_dir"]))
    max_clips = int(dataset_cfg.get("max_clips", 1200))
    split = str(dataset_cfg.get("split", "train"))
    hf_config = dataset_cfg.get("hf_config")
    audio_column = str(dataset_cfg.get("audio_column", "audio"))
    prefix = str(dataset_cfg.get("prefix", repo.split("/")[-1].replace("-", "_")))
    io_workers = _resolve_io_workers(dataset_cfg)
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", False))
    prefer_streaming = bool(dataset_cfg.get("streaming", True))
    fallback_to_non_streaming = bool(dataset_cfg.get("fallback_to_non_streaming", True))
    segment_duration_s = dataset_cfg.get("segment_duration_s")
    segment_overlap_s = float(dataset_cfg.get("segment_overlap_s", 0.0))
    min_segment_duration_s = dataset_cfg.get("min_segment_duration_s")

    if _dir_has_entries(output_dir):
        log.info("  HF audio dataset already present at %s", output_dir)
        return

    _reset_dir(output_dir)
    count = 0
    dataset_kwargs = {}
    if hf_config is not None and str(hf_config).strip():
        dataset_kwargs["name"] = str(hf_config)
    if trust_remote_code:
        dataset_kwargs["trust_remote_code"] = True

    def load(split_name: str, streaming: bool):
        return load_dataset(repo, split=split_name, streaming=streaming, **dataset_kwargs)

    for split_name in _split_spec_parts(split):
        if count >= max_clips:
            break

        dataset = None
        try:
            dataset = load(split_name, prefer_streaming)
        except Exception:
            if not fallback_to_non_streaming:
                raise
            dataset = load(split_name, False)

        count += _write_dataset_audio(
            _iter_filtered_dataset(dataset, dataset_cfg),
            output_dir,
            prefix,
            max_clips - count,
            io_workers,
            audio_column=audio_column,
            segment_duration_s=float(segment_duration_s) if segment_duration_s else None,
            segment_overlap_s=segment_overlap_s,
            min_segment_duration_s=float(min_segment_duration_s) if min_segment_duration_s else None,
        )

    log.info("  Saved %d clips from %s to %s", count, repo, output_dir)


def _split_spec_parts(split_spec: str) -> list[str]:
    normalized = split_spec.replace(",", "+")
    return [part.strip() for part in normalized.split("+") if part.strip()]


def _download_fleurs_dataset(dataset_cfg: dict) -> None:
    merged = {
        "hf_repo": "google/fleurs",
        "hf_config": dataset_cfg.get("hf_config", "pl_pl"),
        "output_dir": dataset_cfg.get("output_dir", "data/fleurs_pl"),
        "split": dataset_cfg.get("split", "train+validation"),
        "max_clips": dataset_cfg.get("max_clips", 1200),
        "prefix": dataset_cfg.get("prefix", "fleurs"),
        "streaming": dataset_cfg.get("streaming", True),
        "fallback_to_non_streaming": dataset_cfg.get("fallback_to_non_streaming", True),
        "audio_column": dataset_cfg.get("audio_column", "audio"),
    }
    if "io_workers" in dataset_cfg:
        merged["io_workers"] = dataset_cfg["io_workers"]
    _download_hf_audio_dataset(merged)


def _download_common_voice_dataset(dataset_cfg: dict) -> None:
    merged = {
        "hf_repo": "mozilla-foundation/common_voice_16_0",
        "hf_config": dataset_cfg.get("hf_config", "pl"),
        "output_dir": dataset_cfg.get("output_dir", "data/common_voice_pl"),
        "split": dataset_cfg.get("split", "train+validation"),
        "max_clips": dataset_cfg.get("max_clips", 1200),
        "prefix": dataset_cfg.get("prefix", "common_voice"),
        "streaming": dataset_cfg.get("streaming", True),
        "fallback_to_non_streaming": dataset_cfg.get("fallback_to_non_streaming", True),
        "audio_column": dataset_cfg.get("audio_column", "audio"),
    }
    if "io_workers" in dataset_cfg:
        merged["io_workers"] = dataset_cfg["io_workers"]
    _download_hf_audio_dataset(merged)


def _download_wham_dataset(dataset_cfg: dict) -> None:
    merged = {
        "hf_repo": dataset_cfg.get("hf_repo", "philgzl/wham"),
        "output_dir": dataset_cfg.get("output_dir", "data/wham_noise"),
        "split": dataset_cfg.get("split", "train"),
        "max_clips": dataset_cfg.get("max_clips", 4000),
        "prefix": dataset_cfg.get("prefix", "wham_noise"),
        "streaming": dataset_cfg.get("streaming", True),
        "fallback_to_non_streaming": dataset_cfg.get("fallback_to_non_streaming", True),
        "audio_column": dataset_cfg.get("audio_column", "audio"),
        "segment_duration_s": dataset_cfg.get("segment_duration_s", 5.0),
        "segment_overlap_s": dataset_cfg.get("segment_overlap_s", 2.5),
        "min_segment_duration_s": dataset_cfg.get("min_segment_duration_s", 4.0),
    }
    if "io_workers" in dataset_cfg:
        merged["io_workers"] = dataset_cfg["io_workers"]
    _download_hf_audio_dataset(merged)


def _download_bigos_dataset(dataset_cfg: dict) -> None:
    from datasets import load_dataset

    output_dir = _resolve_path(str(dataset_cfg.get("output_dir", "data/bigos")))
    max_clips = int(dataset_cfg.get("max_clips", 2000))
    split = str(dataset_cfg.get("split", "train"))
    hf_config = str(dataset_cfg.get("hf_config", "all"))
    io_workers = _resolve_io_workers(dataset_cfg)

    if _dir_has_entries(output_dir):
        log.info("  BIGOS dataset already present at %s", output_dir)
        return

    _reset_dir(output_dir)
    try:
        dataset = load_dataset(
            "amu-cai/pl-asr-bigos-v2",
            hf_config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        dataset = load_dataset(
            "amu-cai/pl-asr-bigos-v2",
            hf_config,
            split=split,
            streaming=False,
            trust_remote_code=True,
        )
    count = _write_dataset_audio(dataset, output_dir, "bigos", max_clips, io_workers)
    log.info("  Saved %d BIGOS clips to %s", count, output_dir)


def _bootstrap_positive_speech_datasets(cfg: dict) -> None:
    dataset_entries = [
        dataset_cfg
        for dataset_cfg in (cfg.get("bootstrap_speech_datasets", []) or [])
        if bool(dataset_cfg.get("enabled", True))
    ]

    if not dataset_entries:
        return

    def bootstrap(dataset_cfg: dict) -> None:
        kind = str(dataset_cfg.get("kind", "")).strip().lower()
        optional = bool(dataset_cfg.get("optional", False))

        try:
            if kind == "fleurs":
                _download_fleurs_dataset(dataset_cfg)
            elif kind == "bigos":
                _download_bigos_dataset(dataset_cfg)
            elif kind == "common_voice":
                _download_common_voice_dataset(dataset_cfg)
            elif kind == "wham":
                _download_wham_dataset(dataset_cfg)
            elif kind == "hf_audio":
                _download_hf_audio_dataset(dataset_cfg)
            else:
                raise ValueError(f"Unsupported bootstrap dataset kind: {kind}")
        except Exception as exc:
            if optional:
                log.warning("  Optional dataset bootstrap skipped for %s: %s", kind, exc)
            else:
                raise

    workers = min(_resolve_bootstrap_workers(cfg), len(dataset_entries))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(bootstrap, dataset_cfg) for dataset_cfg in dataset_entries]
        for future in futures:
            future.result()


def _bootstrap_background_audio_datasets(cfg: dict) -> None:
    dataset_entries = [
        dataset_cfg
        for dataset_cfg in (cfg.get("bootstrap_background_datasets", []) or [])
        if bool(dataset_cfg.get("enabled", True))
    ]

    if not dataset_entries:
        return

    def bootstrap(dataset_cfg: dict) -> None:
        kind = str(dataset_cfg.get("kind", "")).strip().lower()
        optional = bool(dataset_cfg.get("optional", False))

        try:
            if kind == "hf_audio":
                _download_hf_audio_dataset(dataset_cfg)
            elif kind == "wham":
                _download_wham_dataset(dataset_cfg)
            elif kind == "musan_openslr":
                _download_musan_openslr(_resolve_path(str(dataset_cfg.get("output_dir", "data/musan_noise_music"))))
            else:
                raise ValueError(f"Unsupported background dataset kind: {kind}")
        except Exception as exc:
            if optional:
                log.warning("  Optional background dataset bootstrap skipped for %s: %s", kind, exc)
            else:
                raise

    workers = min(_resolve_bootstrap_workers(cfg), len(dataset_entries))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(bootstrap, dataset_cfg) for dataset_cfg in dataset_entries]
        for future in futures:
            future.result()


def _resolve_positive_sources(cfg: dict) -> list[Path]:
    sources: list[Path] = []

    if str(cfg.get("positive_dataset_path", "")).strip():
        sources.append(_resolve_path(str(cfg["positive_dataset_path"])))

    for item in cfg.get("positive_dataset_paths", []) or []:
        if str(item).strip():
            sources.append(_resolve_path(str(item)))

    unique_sources: list[Path] = []
    seen: set[str] = set()
    for path in sources:
        key = str(path)
        if key not in seen:
            unique_sources.append(path)
            seen.add(key)
    return unique_sources


def _stage_audio_sources(
    staged_dir: Path,
    manifest_path: Path,
    source_dirs: list[Path],
    label: str,
) -> Path:
    source_manifest = [str(path) for path in source_dirs]

    if staged_dir.exists() and manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_manifest = None
        if current_manifest == source_manifest and _safe_iter_audio_files(staged_dir):
            log.info("  Using staged %s audio at %s", label, staged_dir)
            return staged_dir

    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    for source_dir in source_dirs:
        files = _safe_iter_audio_files(source_dir)
        for audio_file in files:
            dest = staged_dir / f"{source_dir.name}_{index:06d}{audio_file.suffix.lower()}"
            _link_or_copy(audio_file, dest)
            index += 1

    if index == 0:
        raise ValueError(f"No {label} audio files were found in the configured source directories")

    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    log.info("  Staged %d %s audio files into %s", index, label, staged_dir)
    return staged_dir


def _stage_positive_sources(cfg: dict, source_dirs: list[Path]) -> Path:
    return _stage_audio_sources(
        staged_dir=_staged_positive_dir(cfg),
        manifest_path=_project_dir(cfg) / "staged_positive_sources.json",
        source_dirs=source_dirs,
        label="positive",
    )


def _stage_background_sources(cfg: dict, source_dirs: list[Path]) -> Path:
    return _stage_audio_sources(
        staged_dir=_staged_background_dir(cfg),
        manifest_path=_project_dir(cfg) / "staged_background_sources.json",
        source_dirs=source_dirs,
        label="background",
    )


def _generated_background_negatives_cfg(cfg: dict) -> dict:
    return cfg.get("generated_background_negatives", {}) or {}


def _generated_background_negatives_enabled(cfg: dict) -> bool:
    return _task(cfg) == "vad" and bool(_generated_background_negatives_cfg(cfg).get("enabled", False))


def _generated_background_negative_manifest_path(cfg: dict) -> Path:
    return _project_dir(cfg) / "generated_background_negative_features.json"


def _has_any_mmap_dir(root: Path) -> bool:
    return any(root.glob("**/*_mmap"))


def _background_negative_pack_ready(root: Path) -> bool:
    required = (
        root / "training",
        root / "validation_ambient",
        root / "testing_ambient",
    )
    return all(directory.exists() and _has_any_mmap_dir(directory) for directory in required)


def step_check_env() -> bool:
    log.info("  Python: %s", sys.executable)
    if sys.version_info < (3, 10):
        log.error("  Python 3.10+ is required")
        return False
    return _check_micro_wake_word_import()


def step_prepare_tools() -> bool:
    if not MWW_DIR.exists():
        log.error("  Missing micro-wake-word checkout at %s", MWW_DIR)
        log.error("  Run ./setup_environment.sh first.")
        return False

    cfg = _load_config()
    if not _needs_piper(cfg):
        log.info("  Piper model not required for this config")
        return True

    try:
        _ensure_piper_assets(cfg)
        return True
    except Exception as exc:
        log.error("  Failed to prepare Piper assets: %s", exc)
        return False


def _download_mit_rirs(dest: Path, io_workers: int = 4) -> None:
    from datasets import load_dataset

    if dest.exists() and any(dest.iterdir()):
        log.info("  MIT RIRs already present")
        return

    dest.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    count = _write_dataset_audio(dataset, dest, "rir", None, io_workers)
    log.info("  Saved %d RIR files", count)


def _download_musan_openslr(dest: Path) -> None:
    archive_dir = DATA_DIR / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "musan.tar.gz"

    if _dir_has_entries(dest):
        log.info("  MUSAN already present at %s", dest)
        return

    _download(
        "https://openslr.trmal.net/resources/17/musan.tar.gz",
        archive_path,
        "MUSAN corpus (OpenSLR)",
    )

    extract_dir = archive_dir / "musan_extract"
    _reset_dir(extract_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_dir)

    musan_root = extract_dir / "musan"
    if not musan_root.exists():
        raise FileNotFoundError(f"Expected MUSAN root not found in archive: {musan_root}")

    _reset_dir(dest)
    copied = 0
    for category in ("music", "noise"):
        source_dir = musan_root / category
        if not source_dir.exists():
            continue
        for audio_file in _safe_iter_audio_files(source_dir):
            rel = audio_file.relative_to(musan_root)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_file, target)
            copied += 1

    shutil.rmtree(extract_dir, ignore_errors=True)
    log.info("  Prepared %d MUSAN music/noise files in %s", copied, dest)


def _download_audioset_subset(dest: Path, limit: int = 300, io_workers: int = 4) -> None:
    from datasets import load_dataset

    if dest.exists() and any(dest.iterdir()):
        log.info("  AudioSet subset already present")
        return

    dest.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "agkphysics/AudioSet",
        "unbalanced",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    count = _write_dataset_audio(dataset, dest, "audioset", limit, io_workers)
    log.info("  Saved %d AudioSet clips", count)


def _download_fma_subset(dest: Path, limit: int = 200, io_workers: int = 4) -> None:
    from datasets import load_dataset

    if dest.exists() and any(dest.iterdir()):
        log.info("  FMA subset already present")
        return

    dest.mkdir(parents=True, exist_ok=True)

    count = 0

    try:
        dataset = load_dataset(
            "rudraml/fma",
            name="small",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count = _write_dataset_audio(dataset, dest, "fma", limit, io_workers)
    except Exception as exc:
        log.warning("  Streaming FMA download failed, retrying non-streaming subset: %s", exc)
        dataset = load_dataset(
            "rudraml/fma",
            name="small",
            split=f"train[:{limit}]",
            streaming=False,
            trust_remote_code=True,
        )
        count = _write_dataset_audio(dataset, dest, "fma", limit, io_workers)

    log.info("  Saved %d FMA clips", count)


def _download_negative_feature_pack(dest_root: Path, name: str) -> None:
    from mmap_ninja.ragged import RaggedMmap

    def validate_pack(pack_dir: Path) -> tuple[bool, str | None]:
        mmap_dirs = [path for path in pack_dir.glob("**/*_mmap") if path.is_dir()]
        if not mmap_dirs:
            return False, "no mmap directories found"

        for mmap_dir in mmap_dirs:
            try:
                RaggedMmap(str(mmap_dir))
            except Exception as exc:
                return False, f"{mmap_dir}: {exc}"

        return True, None

    target_dir = dest_root / name
    zip_path = dest_root / f"{name}.zip"
    if target_dir.exists() and any(target_dir.iterdir()):
        valid, reason = validate_pack(target_dir)
        if valid:
            log.info("  Negative feature pack already present: %s", name)
            return

        log.warning("  Negative feature pack %s is corrupted (%s). Re-downloading.", name, reason)
        shutil.rmtree(target_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink()

    _download(
        f"{NEGATIVE_FEATURE_ROOT}/{name}.zip",
        zip_path,
        f"negative feature pack {name}",
    )
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_root)

    valid, reason = validate_pack(target_dir)
    if not valid:
        raise ValueError(f"negative feature pack {name} is still invalid after download: {reason}")

    log.info("  Extracted negative feature pack: %s", name)


def _negative_feature_names(cfg: dict) -> list[str]:
    names = cfg.get("negative_feature_sets")
    if names:
        return [str(name) for name in names]
    return ["speech", "dinner_party", "no_speech", "dinner_party_eval"]


def _resolve_background_audio_paths(cfg: dict) -> list[str]:
    defaults = [DATA_DIR / "fma_16k", DATA_DIR / "audioset_16k"]
    configured = [_resolve_path(str(path)) for path in cfg.get("background_audio_paths", []) or [] if str(path).strip()]

    resolved: list[str] = []
    seen: set[str] = set()
    for path in [*defaults, *configured]:
        try:
            has_entries = _dir_has_entries(path)
        except Exception:
            has_entries = False
        if not has_entries:
            continue
        key = str(path)
        if key not in seen:
            resolved.append(key)
            seen.add(key)
    return resolved


def step_download_assets() -> bool:
    cfg = _load_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    asset_cfg = cfg.get("asset_subsets", {}) or {}
    audioset_max_clips = int(asset_cfg.get("audioset_max_clips", 300))
    fma_max_clips = int(asset_cfg.get("fma_max_clips", 200))
    io_workers = _resolve_io_workers(asset_cfg)

    try:
        _download_mit_rirs(DATA_DIR / "mit_rirs", io_workers=io_workers)
        _download_audioset_subset(
            DATA_DIR / "audioset_16k",
            limit=audioset_max_clips,
            io_workers=io_workers,
        )
        _download_fma_subset(
            DATA_DIR / "fma_16k",
            limit=fma_max_clips,
            io_workers=io_workers,
        )
        _bootstrap_background_audio_datasets(cfg)

        neg_root = _negative_datasets_dir(cfg)
        for name in _negative_feature_names(cfg):
            _download_negative_feature_pack(neg_root, name)
        return True
    except Exception as exc:
        log.error("  Asset download failed: %s", exc)
        return False


def _generate_piper_samples(cfg: dict) -> Path:
    sample_dir = _generated_samples_dir(cfg)
    if sample_dir.exists() and any(sample_dir.glob("*.wav")):
        log.info("  Piper samples already present at %s", sample_dir)
        return sample_dir

    model_path = _ensure_piper_assets(cfg)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        str(cfg["sample_text"]),
        "--model",
        str(model_path),
        "--max-samples",
        str(cfg.get("generated_sample_count", 1000)),
        "--batch-size",
        str(cfg.get("generated_sample_batch_size", 100)),
        "--output-dir",
        str(sample_dir),
    ]
    for item in cfg.get("piper_extra_args", []) or []:
        cmd.append(str(item))
    _run(cmd)
    return sample_dir


def step_prepare_positives() -> bool:
    cfg = _load_config()

    try:
        if _task(cfg) == "vad":
            _bootstrap_positive_speech_datasets(cfg)

        source_dirs = [path for path in _resolve_positive_sources(cfg) if path.exists()]
        if source_dirs:
            total_files = sum(len(_safe_iter_audio_files(path)) for path in source_dirs)
            if total_files == 0:
                log.error("  Positive source directories exist, but no audio files were found")
                return False
            if len(source_dirs) == 1:
                log.info("  Using %d positive clips from %s", total_files, source_dirs[0])
            else:
                _stage_positive_sources(cfg, source_dirs)
            return True

        if _task(cfg) == "wakeword":
            _generate_piper_samples(cfg)
            return True

        log.error("  No positive speech datasets are available for VAD training")
        return False
    except Exception as exc:
        log.error("  Positive sample preparation failed: %s", exc)
        return False


def _positive_source(cfg: dict) -> tuple[Path, str]:
    source_dirs = [path for path in _resolve_positive_sources(cfg) if path.exists()]

    if len(source_dirs) > 1:
        return _stage_positive_sources(cfg, source_dirs), "*.*"
    if len(source_dirs) == 1:
        return source_dirs[0], str(cfg.get("positive_file_pattern", "**/*.wav"))

    if _task(cfg) == "wakeword":
        return _generated_samples_dir(cfg), "*.wav"

    raise ValueError("No positive audio source is available")


def _clip_entry_path(entry: dict) -> str:
    audio = entry.get("audio")
    if isinstance(audio, dict):
        for key in ("path", "filename"):
            value = audio.get(key)
            if value:
                return str(value)
    return str(audio)


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _intersection_size(left: set[str], right: set[str]) -> int:
    return len(left.intersection(right))


def step_audit_validation() -> bool:
    if not _check_micro_wake_word_import():
        return False

    cfg = _load_config()
    try:
        from mmap_ninja.ragged import RaggedMmap
        from microwakeword.audio.clips import Clips
    except Exception as exc:
        log.error("  Audit dependencies are not available: %s", exc)
        return False

    try:
        positive_dir, file_pattern = _positive_source(cfg)
    except Exception as exc:
        log.error("  Could not resolve positive source for audit: %s", exc)
        return False

    try:
        clips = Clips(
            input_directory=str(positive_dir),
            file_pattern=file_pattern,
            max_clip_duration_s=None,
            remove_silence=bool(cfg.get("remove_silence", False)),
            random_split_seed=int(cfg.get("random_split_seed", 10)),
            split_count=float(cfg.get("split_count", 0.1)),
        )
    except Exception as exc:
        log.error("  Failed to build clip splits for audit: %s", exc)
        return False

    if not hasattr(clips, "split_clips"):
        log.error("  Clips object did not expose split datasets; random_split_seed may be missing.")
        return False

    split_paths: dict[str, list[str]] = {}
    split_hashes: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        dataset = clips.split_clips[split_name]
        paths = [_clip_entry_path(entry) for entry in dataset]
        split_paths[split_name] = paths

        hashes: set[str] = set()
        for path_str in paths:
            path = Path(path_str)
            if path.exists():
                try:
                    hashes.add(_hash_file(path))
                except OSError as exc:
                    log.warning("  Could not hash %s: %s", path, exc)
        split_hashes[split_name] = hashes

    log.info("  Positive split sizes:")
    for split_name in ("train", "validation", "test"):
        log.info("    %-10s %6d clips", split_name, len(split_paths[split_name]))

    path_overlaps = {
        "train/validation": _intersection_size(set(split_paths["train"]), set(split_paths["validation"])),
        "train/test": _intersection_size(set(split_paths["train"]), set(split_paths["test"])),
        "validation/test": _intersection_size(set(split_paths["validation"]), set(split_paths["test"])),
    }
    hash_overlaps = {
        "train/validation": _intersection_size(split_hashes["train"], split_hashes["validation"]),
        "train/test": _intersection_size(split_hashes["train"], split_hashes["test"]),
        "validation/test": _intersection_size(split_hashes["validation"], split_hashes["test"]),
    }

    log.info("  Exact path overlaps between positive splits:")
    for label, count in path_overlaps.items():
        log.info("    %-18s %d", label, count)

    log.info("  Content-hash overlaps between positive splits:")
    for label, count in hash_overlaps.items():
        log.info("    %-18s %d", label, count)

    background_paths = _resolve_background_audio_paths(cfg)
    log.info("  Background audio directories used for augmentation:")
    if background_paths:
        for path_str in background_paths:
            path = Path(path_str)
            log.info("    %-50s %6d files", path.name, len(_safe_iter_audio_files(path)))
    else:
        log.warning("    none")

    neg_root = _negative_datasets_dir(cfg)
    log.info("  Negative feature pack ambient stats:")
    found_any = False
    for name in _negative_feature_names(cfg):
        feature_dir = neg_root / name
        validation_ambient_dir = feature_dir / "validation_ambient"
        validation_dir = feature_dir / "validation"
        training_dir = feature_dir / "training"
        testing_ambient_dir = feature_dir / "testing_ambient"

        counts = {
            "training": 0,
            "validation": 0,
            "validation_ambient": 0,
            "testing_ambient": 0,
        }

        for label, directory in (
            ("training", training_dir),
            ("validation", validation_dir),
            ("validation_ambient", validation_ambient_dir),
            ("testing_ambient", testing_ambient_dir),
        ):
            if not directory.exists():
                continue
            for mmap_path in directory.glob("**/*_mmap/"):
                try:
                    counts[label] += len(RaggedMmap(str(mmap_path)))
                except Exception as exc:
                    log.warning("  Could not inspect %s: %s", mmap_path, exc)

        if any(counts.values()):
            found_any = True
            log.info(
                "    %-18s training=%d validation=%d validation_ambient=%d testing_ambient=%d",
                name,
                counts["training"],
                counts["validation"],
                counts["validation_ambient"],
                counts["testing_ambient"],
            )

    if not found_any:
        log.warning("  No negative feature pack stats were found under %s", neg_root)

    generated_background_root = _background_negative_features_dir(cfg)
    if _generated_background_negatives_enabled(cfg):
        counts = {
            "training": 0,
            "validation_ambient": 0,
            "testing_ambient": 0,
        }
        if _background_negative_pack_ready(generated_background_root):
            for label, directory in (
                ("training", generated_background_root / "training"),
                ("validation_ambient", generated_background_root / "validation_ambient"),
                ("testing_ambient", generated_background_root / "testing_ambient"),
            ):
                for mmap_path in directory.glob("**/*_mmap/"):
                    try:
                        counts[label] += len(RaggedMmap(str(mmap_path)))
                    except Exception as exc:
                        log.warning("  Could not inspect %s: %s", mmap_path, exc)
            log.info(
                "  Generated background negatives: training=%d validation_ambient=%d testing_ambient=%d",
                counts["training"],
                counts["validation_ambient"],
                counts["testing_ambient"],
            )
        else:
            log.warning(
                "  Generated background negatives are enabled, but no mmap pack was found under %s",
                generated_background_root,
            )

    suspicious = []
    if any(path_overlaps.values()):
        suspicious.append("exact path overlap between positive splits")
    if any(hash_overlaps.values()):
        suspicious.append("content-duplicate overlap between positive splits")

    if suspicious:
        log.warning("  Audit found suspicious validation issues: %s", ", ".join(suspicious))
    else:
        log.info("  Audit found no overlap between positive splits.")

    return True


def step_generate_positive_features() -> bool:
    if not _check_micro_wake_word_import():
        return False

    cfg = _load_config()
    features_dir = _positive_features_dir(cfg)
    training_mmap = features_dir / "training" / "wakeword_mmap"
    if training_mmap.exists():
        log.info("  Positive features already present at %s", training_mmap)
        return True

    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    positive_dir, file_pattern = _positive_source(cfg)
    features_dir.mkdir(parents=True, exist_ok=True)

    clips = Clips(
        input_directory=str(positive_dir),
        file_pattern=file_pattern,
        max_clip_duration_s=None,
        remove_silence=bool(cfg.get("remove_silence", False)),
        random_split_seed=int(cfg.get("random_split_seed", 10)),
        split_count=float(cfg.get("split_count", 0.1)),
    )

    aug_cfg = cfg.get("augmentation", {})
    background_paths = _resolve_background_audio_paths(cfg)
    if not background_paths:
        log.warning("  No background audio directories were found for augmentation.")
    augmenter = Augmentation(
        augmentation_duration_s=float(aug_cfg.get("duration_s", 3.2)),
        augmentation_probabilities=_resolve_augmentation_probabilities(
            aug_cfg.get("probabilities", {})
        ),
        impulse_paths=[str(DATA_DIR / "mit_rirs")],
        background_paths=background_paths,
        background_min_snr_db=int(aug_cfg.get("background_min_snr_db", -5)),
        background_max_snr_db=int(aug_cfg.get("background_max_snr_db", 10)),
        min_jitter_s=float(aug_cfg.get("min_jitter_s", 0.195)),
        max_jitter_s=float(aug_cfg.get("max_jitter_s", 0.205)),
    )

    for split in ("training", "validation", "testing"):
        out_dir = features_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        split_name = "train"
        repetition = 2
        slide_frames = 10
        if split == "validation":
            split_name = "validation"
            repetition = 1
        elif split == "testing":
            split_name = "test"
            repetition = 1
            slide_frames = 1

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name,
                repeat=repetition,
            ),
            batch_size=100,
            verbose=True,
        )

    log.info("  Generated positive feature datasets at %s", features_dir)
    return True


def step_generate_background_negative_features() -> bool:
    if not _check_micro_wake_word_import():
        return False

    cfg = _load_config()
    neg_cfg = _generated_background_negatives_cfg(cfg)
    if not _generated_background_negatives_enabled(cfg):
        log.info("  Generated background negatives are disabled for this config")
        return True

    source_dirs = [Path(path) for path in _resolve_background_audio_paths(cfg)]
    if not source_dirs:
        log.warning("  No background audio directories were found for generated negative features.")
        return True

    staged_dir = _stage_background_sources(cfg, source_dirs)
    features_dir = _background_negative_features_dir(cfg)
    manifest_path = _generated_background_negative_manifest_path(cfg)

    feature_manifest = {
        "sources": [str(path) for path in source_dirs],
        "duration_s": float(neg_cfg.get("duration_s", cfg.get("augmentation", {}).get("duration_s", 3.2))),
        "training_repeat": int(neg_cfg.get("training_repeat", 1)),
        "training_slide_frames": int(neg_cfg.get("training_slide_frames", 5)),
        "eval_slide_frames": int(neg_cfg.get("eval_slide_frames", 1)),
        "random_split_seed": int(cfg.get("random_split_seed", 10)),
        "split_count": float(cfg.get("split_count", 0.1)),
        "probabilities": dict(neg_cfg.get("probabilities", {})),
        "use_rirs": bool(neg_cfg.get("use_rirs", False)),
    }

    if _background_negative_pack_ready(features_dir) and manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_manifest = None
        if current_manifest == feature_manifest:
            log.info("  Generated background negative features already present at %s", features_dir)
            return True

    if features_dir.exists():
        shutil.rmtree(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    clips = Clips(
        input_directory=str(staged_dir),
        file_pattern="*.*",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=int(cfg.get("random_split_seed", 10)),
        split_count=float(cfg.get("split_count", 0.1)),
    )

    impulse_paths: list[str] = []
    if bool(neg_cfg.get("use_rirs", False)) and _dir_has_entries(DATA_DIR / "mit_rirs"):
        impulse_paths = [str(DATA_DIR / "mit_rirs")]

    augmenter = Augmentation(
        augmentation_duration_s=float(neg_cfg.get("duration_s", cfg.get("augmentation", {}).get("duration_s", 3.2))),
        augmentation_probabilities=_resolve_augmentation_probabilities(neg_cfg.get("probabilities", {})),
        impulse_paths=impulse_paths,
        background_paths=[],
        background_min_snr_db=0,
        background_max_snr_db=0,
        min_jitter_s=0.0,
        max_jitter_s=0.0,
    )

    split_plan = (
        ("training", "train", int(neg_cfg.get("training_repeat", 1)), int(neg_cfg.get("training_slide_frames", 5))),
        ("validation_ambient", "validation", 1, int(neg_cfg.get("eval_slide_frames", 1))),
        ("testing_ambient", "test", 1, int(neg_cfg.get("eval_slide_frames", 1))),
    )

    for split_name, clip_split, repetition, slide_frames in split_plan:
        out_dir = features_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=str(out_dir / "background_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=clip_split,
                repeat=repetition,
            ),
            batch_size=100,
            verbose=True,
        )

    manifest_path.write_text(json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8")
    log.info("  Generated background negative feature datasets at %s", features_dir)
    return True


def _write_training_config(cfg: dict) -> Path:
    feature_root = _positive_features_dir(cfg)
    neg_root = _negative_datasets_dir(cfg)
    train_cfg = cfg.get("training", {})

    negative_entries = []
    weights = cfg.get("negative_feature_weights", {})
    truncation = cfg.get("negative_feature_truncation", {})
    for name in _negative_feature_names(cfg):
        negative_entries.append(
            {
                "features_dir": str(neg_root / name),
                "sampling_weight": float(weights.get(name, 10.0 if name != "dinner_party_eval" else 0.0)),
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": str(truncation.get(name, "split" if name == "dinner_party_eval" else "random")),
                "type": "mmap",
            }
        )

    generated_background_root = _background_negative_features_dir(cfg)
    generated_background_cfg = _generated_background_negatives_cfg(cfg)
    if _generated_background_negatives_enabled(cfg) and _background_negative_pack_ready(generated_background_root):
        negative_entries.append(
            {
                "features_dir": str(generated_background_root),
                "sampling_weight": float(generated_background_cfg.get("sampling_weight", 8.0)),
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": str(generated_background_cfg.get("truncation_strategy", "random")),
                "type": "mmap",
            }
        )

    config = {
        "window_step_ms": 10,
        "train_dir": str(_training_dir(cfg)),
        "features": [
            {
                "features_dir": str(feature_root),
                "sampling_weight": float(cfg.get("positive_sampling_weight", 2.0)),
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": str(cfg.get("positive_truncation_strategy", "truncate_start")),
                "type": "mmap",
            },
            *negative_entries,
        ],
        "training_steps": train_cfg.get("training_steps", [10000]),
        "positive_class_weight": train_cfg.get("positive_class_weight", [1]),
        "negative_class_weight": train_cfg.get("negative_class_weight", [20]),
        "learning_rates": train_cfg.get("learning_rates", [0.001]),
        "batch_size": int(train_cfg.get("batch_size", 128)),
        "time_mask_max_size": train_cfg.get("time_mask_max_size", [0]),
        "time_mask_count": train_cfg.get("time_mask_count", [0]),
        "freq_mask_max_size": train_cfg.get("freq_mask_max_size", [0]),
        "freq_mask_count": train_cfg.get("freq_mask_count", [0]),
        "eval_step_interval": int(train_cfg.get("eval_step_interval", 500)),
        "clip_duration_ms": int(train_cfg.get("clip_duration_ms", 1500)),
        "target_minimization": train_cfg.get("target_minimization", 0.9),
        "minimization_metric": train_cfg.get("minimization_metric"),
        "maximization_metric": train_cfg.get("maximization_metric", "average_viable_recall"),
    }

    out = _project_dir(cfg) / "training_parameters.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return out


def step_train() -> bool:
    if not _check_micro_wake_word_import():
        return False
    if not _ensure_python_module("tensorboard", TENSORBOARD_PIP_SPEC):
        return False
    if not _patch_microwakeword_train_py():
        return False
    if not _patch_microwakeword_test_py():
        return False

    cfg = _load_config()
    try:
        neg_root = _negative_datasets_dir(cfg)
        for name in _negative_feature_names(cfg):
            _download_negative_feature_pack(neg_root, name)

        if _generated_background_negatives_enabled(cfg) and not _background_negative_pack_ready(_background_negative_features_dir(cfg)):
            log.info("  Generated background negative features are missing; building them before training.")
            if not step_generate_background_negative_features():
                return False

        training_config = _write_training_config(cfg)
        cmd = [
            sys.executable,
            "-m",
            "microwakeword.model_train_eval",
            f"--training_config={training_config}",
            "--train",
            "1",
            "--restore_checkpoint",
            "1",
            "--test_tf_nonstreaming",
            "0",
            "--test_tflite_nonstreaming",
            "0",
            "--test_tflite_nonstreaming_quantized",
            "0",
            "--test_tflite_streaming",
            "0",
            "--test_tflite_streaming_quantized",
            "1",
            "--use_weights",
            "best_weights",
            "mixednet",
            "--pointwise_filters",
            "64,64,64,64",
            "--repeat_in_block",
            "1,1,1,1",
            "--mixconv_kernel_sizes",
            "[5],[7,11],[9,15],[23]",
            "--residual_connection",
            "0,0,0,0",
            "--first_conv_filters",
            "32",
            "--first_conv_kernel_size",
            "5",
            "--stride",
            "3",
        ]
        runtime = _runtime_cfg(cfg)
        train_env, primary_device = _build_training_env(cfg)
        log.info("  Training runtime device preference: %s", primary_device)
        if train_env.get("TF_XLA_FLAGS"):
            log.info("  TF_XLA_FLAGS=%s", train_env["TF_XLA_FLAGS"])
        if train_env.get("XLA_FLAGS"):
            log.info("  XLA_FLAGS=%s", train_env["XLA_FLAGS"])

        try:
            _run(cmd, env=train_env)
        except subprocess.CalledProcessError as exc:
            allow_cpu_fallback = bool(runtime.get("allow_cpu_fallback", True))
            if primary_device != "cpu" and allow_cpu_fallback:
                log.warning(
                    "  GPU/auto training failed with exit code %d. Retrying once on CPU.",
                    exc.returncode,
                )
                cpu_env, _ = _build_training_env(cfg, force_device="cpu")
                log.info("  CPU fallback sets CUDA_VISIBLE_DEVICES=%s", cpu_env["CUDA_VISIBLE_DEVICES"])
                _run(cmd, env=cpu_env)
            else:
                raise
        return True
    except subprocess.CalledProcessError as exc:
        log.error("  Training failed with exit code %d", exc.returncode)
        return False
    except ValueError as exc:
        log.error("  Training runtime configuration is invalid: %s", exc)
        return False


def step_export() -> bool:
    cfg = _load_config()
    train_dir = _training_dir(cfg)
    source = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if not source.exists():
        log.error("  Expected trained TFLite file not found: %s", source)
        return False

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    dest_model = EXPORT_DIR / f"{cfg['model_name']}.tflite"
    shutil.copy2(source, dest_model)

    manifest_cfg = cfg.get("esphome_manifest", {})
    manifest = {
        "type": "micro",
        "wake_word": cfg.get("wake_word", cfg["model_name"]),
        "author": manifest_cfg.get("author", "ha-wakeword-trainer"),
        "website": manifest_cfg.get("website", "https://github.com/lukcz/ha-wakeword-trainer"),
        "model": dest_model.name,
        "trained_languages": manifest_cfg.get("trained_languages", ["en"]),
        "version": 2,
        "micro": {
            "probability_cutoff": float(manifest_cfg.get("probability_cutoff", 0.97)),
            "feature_step_size": int(manifest_cfg.get("feature_step_size", 10)),
            "sliding_window_size": int(manifest_cfg.get("sliding_window_size", 5)),
            "tensor_arena_size": int(manifest_cfg.get("tensor_arena_size", 26080)),
            "minimum_esphome_version": manifest_cfg.get("minimum_esphome_version", "2024.7.0"),
        },
    }
    dest_manifest = EXPORT_DIR / f"{cfg['model_name']}.json"
    dest_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("  Exported %s and %s", dest_model, dest_manifest)
    return True


STEPS: list[tuple[str, Callable[[], bool], str]] = [
    ("check-env", step_check_env, "Verify Python and microWakeWord dependencies"),
    ("prepare-tools", step_prepare_tools, "Verify microWakeWord and Piper tooling"),
    ("download-assets", step_download_assets, "Download augmentation data and negative feature packs"),
    ("prepare-positives", step_prepare_positives, "Prepare positive speech or wake-word samples"),
    ("generate-positive-features", step_generate_positive_features, "Generate positive Ragged Mmap features"),
    ("generate-background-negative-features", step_generate_background_negative_features, "Generate negative Ragged Mmap features from background audio"),
    ("audit-validation", step_audit_validation, "Audit split integrity and ambient validation data"),
    ("train", step_train, "Train and quantize the microWakeWord model"),
    ("export", step_export, "Export the TFLite model and ESPHome manifest"),
]
STEP_NAMES = [name for name, _, _ in STEPS]


def _print_steps() -> None:
    print("\nAvailable steps:\n")
    for index, (name, _, description) in enumerate(STEPS, 1):
        print(f"  {index:2d}. {name:<28s} {description}")
    print()


def run_pipeline(from_step: str | None = None, single_step: str | None = None) -> bool:
    if single_step:
        matches = [(name, fn, desc) for name, fn, desc in STEPS if name == single_step]
        if not matches:
            log.error("Unknown step: %s", single_step)
            _print_steps()
            return False
        name, fn, description = matches[0]
        log.info("STEP: %s - %s", name, description)
        return fn()

    steps_to_run = STEPS
    if from_step:
        if from_step not in STEP_NAMES:
            log.error("Unknown step: %s", from_step)
            _print_steps()
            return False
        steps_to_run = STEPS[STEP_NAMES.index(from_step):]

    total = len(steps_to_run)
    for index, (name, fn, description) in enumerate(steps_to_run, 1):
        log.info("")
        log.info("=" * 60)
        log.info("[%d/%d] %s - %s", index, total, name, description)
        log.info("=" * 60)
        if not fn():
            log.error("[%d/%d] %s FAILED", index, total, name)
            log.error("Pipeline stopped. Resume with:")
            log.error("  python train_microwakeword.py --from %s", name)
            return False
        log.info("[%d/%d] %s PASSED", index, total, name)

    log.info("All steps complete.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Home Assistant Voice PE wake word or VAD model with microWakeWord.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python train_microwakeword.py
              python train_microwakeword.py --config configs/microwakeword_example.yaml
              python train_microwakeword.py --config configs/polish_vad.yaml
              python train_microwakeword.py --step download-assets
              python train_microwakeword.py --from train
              python train_microwakeword.py --list-steps
            """
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the microWakeWord training config YAML.",
    )
    parser.add_argument("--step", default=None, help="Run a single step.")
    parser.add_argument("--from", dest="from_step", default=None, help="Resume from a step.")
    parser.add_argument("--list-steps", action="store_true", help="Print the step list and exit.")
    args = parser.parse_args()

    if args.list_steps:
        _print_steps()
        return

    global CONFIG_FILE
    CONFIG_FILE = Path(args.config).resolve()

    ok = run_pipeline(from_step=args.from_step, single_step=args.step)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
