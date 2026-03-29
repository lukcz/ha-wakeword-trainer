#!/usr/bin/env python3
"""Train a Home Assistant Voice PE wake word or VAD model with microWakeWord."""

from __future__ import annotations

import argparse
import tarfile
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import re
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
MDC_PIP_SPEC = "datacollective>=0.4.5"
ZIPFILE_INFLATE64_PIP_SPEC = "zipfile-inflate64>=0.1"
COMMON_VOICE_MDC_ORG_URL = "https://datacollective.mozillafoundation.org/organization/cmfh0j9o10006ns07jq45h7xk"
BOOTSTRAP_MANIFEST_NAME = ".bootstrap_manifest.json"

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


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _supports_inline_progress() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_INLINE_PROGRESS")


def _write_inline_progress(message: str) -> None:
    if not _supports_inline_progress():
        return
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()


def _finish_inline_progress() -> None:
    if not _supports_inline_progress():
        return
    sys.stdout.write("\n")
    sys.stdout.flush()


def _extract_zip_with_external_tools(archive_path: Path, extract_dir: Path) -> bool:
    candidates = [
        ["7z", "x", str(archive_path), f"-o{extract_dir}", "-y"],
        ["7zz", "x", str(archive_path), f"-o{extract_dir}", "-y"],
        ["bsdtar", "-xf", str(archive_path), "-C", str(extract_dir)],
        ["unzip", "-o", str(archive_path), "-d", str(extract_dir)],
    ]

    for cmd in candidates:
        executable = shutil.which(cmd[0])
        if not executable:
            continue
        try:
            log.info("  Extracting %s with external tool %s", archive_path.name, cmd[0])
            subprocess.check_call([executable, *cmd[1:]])
            return True
        except subprocess.CalledProcessError as exc:
            log.warning("  External extractor %s failed for %s: %s", cmd[0], archive_path.name, exc)
            _reset_dir(extract_dir)

    return False


def _extract_members_with_progress(
    *,
    description: str,
    members: list,
    extract_one: Callable[[object], None],
    size_getter: Callable[[object], int],
) -> None:
    total_members = len(members)
    total_bytes = sum(max(0, int(size_getter(member))) for member in members)
    if total_members == 0:
        log.info("  Archive %s is empty", description)
        return

    log.info(
        "  Extracting %s: %d entries totaling %s",
        description,
        total_members,
        _format_bytes(total_bytes),
    )

    extracted_members = 0
    extracted_bytes = 0
    start_ts = time.time()
    last_log_ts = start_ts
    progress_log_interval_s = 5.0
    last_logged_bucket = -1
    inline_progress = _supports_inline_progress()

    for member in members:
        extract_one(member)
        extracted_members += 1
        extracted_bytes += max(0, int(size_getter(member)))

        now = time.time()
        should_log = False
        if total_bytes > 0:
            bucket = int((extracted_bytes / total_bytes) * 100)
            if bucket > last_logged_bucket:
                last_logged_bucket = bucket
                should_log = True
        elif extracted_members == total_members:
            should_log = True

        if now - last_log_ts >= progress_log_interval_s:
            should_log = True

        if should_log:
            elapsed = max(now - start_ts, 0.001)
            rate = extracted_bytes / elapsed if extracted_bytes > 0 else 0.0
            if total_bytes > 0:
                pct = min(100.0, (extracted_bytes / total_bytes) * 100.0)
                remaining_bytes = max(total_bytes - extracted_bytes, 0)
                eta_seconds = remaining_bytes / rate if rate > 0 else 0.0
                progress_message = (
                    f"  Extracting {description}: "
                    f"{extracted_members}/{total_members} files, "
                    f"{_format_bytes(extracted_bytes)} / {_format_bytes(total_bytes)} "
                    f"({pct:.1f}%) at {_format_bytes(int(rate))}/s, ETA {_format_duration(eta_seconds)}"
                )
            else:
                progress_message = (
                    f"  Extracting {description}: "
                    f"{extracted_members}/{total_members} files"
                )

            if inline_progress:
                _write_inline_progress(progress_message)
            else:
                log.info(progress_message)
            last_log_ts = now

    if inline_progress:
        _finish_inline_progress()

    elapsed = max(time.time() - start_ts, 0.001)
    rate = extracted_bytes / elapsed if extracted_bytes > 0 else 0.0
    log.info(
        "  Finished extracting %s: %d entries, %s in %.1fs (%s/s)",
        description,
        total_members,
        _format_bytes(extracted_bytes),
        elapsed,
        _format_bytes(int(rate)),
    )


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
                total_bytes = int(response.headers.get("content-length", "0") or "0")
                if total_bytes > 0:
                    log.info("  Expected download size for %s: %s", description, _format_bytes(total_bytes))

                bytes_written = 0
                start_ts = time.time()
                last_log_ts = start_ts
                last_logged_bucket = -1
                progress_log_interval_s = 5.0
                inline_progress = _supports_inline_progress()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                            bytes_written += len(chunk)

                            now = time.time()
                            should_log = False
                            if total_bytes > 0:
                                bucket = int((bytes_written / total_bytes) * 100)
                                if bucket > last_logged_bucket:
                                    last_logged_bucket = bucket
                                    should_log = True
                                if now - last_log_ts >= progress_log_interval_s:
                                    should_log = True
                            elif now - last_log_ts >= progress_log_interval_s:
                                should_log = True

                            if should_log:
                                elapsed = max(now - start_ts, 0.001)
                                rate = bytes_written / elapsed
                                if total_bytes > 0:
                                    pct = min(100.0, (bytes_written / total_bytes) * 100.0)
                                    remaining_bytes = max(total_bytes - bytes_written, 0)
                                    eta_seconds = remaining_bytes / rate if rate > 0 else 0.0
                                    progress_message = (
                                        f"  Download progress for {description}: "
                                        f"{_format_bytes(bytes_written)} / {_format_bytes(total_bytes)} "
                                        f"({pct:.1f}%) at {_format_bytes(int(rate))}/s, ETA {_format_duration(eta_seconds)}"
                                    )
                                    if inline_progress:
                                        _write_inline_progress(progress_message)
                                    else:
                                        log.info(
                                            "  Download progress for %s: %s / %s (%.1f%%) at %s/s, ETA %s",
                                            description,
                                            _format_bytes(bytes_written),
                                            _format_bytes(total_bytes),
                                            pct,
                                            _format_bytes(int(rate)),
                                            _format_duration(eta_seconds),
                                        )
                                else:
                                    progress_message = (
                                        f"  Download progress for {description}: "
                                        f"{_format_bytes(bytes_written)} at {_format_bytes(int(rate))}/s"
                                    )
                                    if inline_progress:
                                        _write_inline_progress(progress_message)
                                    else:
                                        log.info(
                                            "  Download progress for %s: %s at %s/s",
                                            description,
                                            _format_bytes(bytes_written),
                                            _format_bytes(int(rate)),
                                        )
                                last_log_ts = now

                if inline_progress:
                    _finish_inline_progress()

                elapsed = max(time.time() - start_ts, 0.001)
                log.info(
                    "  Finished downloading %s: %s in %.1fs (%s/s)",
                    description,
                    _format_bytes(bytes_written),
                    elapsed,
                    _format_bytes(int(bytes_written / elapsed)),
                )
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
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
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


def _audio_file_count(path: Path) -> int:
    return len(_safe_iter_audio_files(path))


def _bootstrap_manifest_path(path: Path) -> Path:
    return path / BOOTSTRAP_MANIFEST_NAME


def _write_bootstrap_manifest(path: Path, *, description: str, expected_audio_files: int, metadata: dict | None = None) -> None:
    manifest = {
        "description": description,
        "expected_audio_files": int(expected_audio_files),
    }
    if metadata:
        manifest["metadata"] = metadata
    _bootstrap_manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _bootstrap_audio_dir_verified(path: Path, *, description: str) -> bool:
    if not path.exists():
        return False

    manifest_path = _bootstrap_manifest_path(path)
    if not manifest_path.exists():
        if _audio_file_count(path) > 0:
            log.warning("  Existing %s at %s has no bootstrap manifest. Rebuilding to verify completeness.", description, path)
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_audio_files = int(manifest.get("expected_audio_files", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("  Bootstrap manifest for %s is invalid. Rebuilding.", path)
        return False

    actual_audio_files = _audio_file_count(path)
    if actual_audio_files != expected_audio_files or actual_audio_files == 0:
        if actual_audio_files == 0:
            reason = "no audio files found"
        elif actual_audio_files < expected_audio_files:
            reason = f"too few audio files ({actual_audio_files}/{expected_audio_files})"
        elif actual_audio_files > expected_audio_files:
            reason = f"too many audio files ({actual_audio_files}/{expected_audio_files})"
        else:
            reason = f"audio file count mismatch ({actual_audio_files}/{expected_audio_files})"
        log.warning(
            "  Existing %s at %s does not match the expected bootstrap output: %s. Rebuilding.",
            description,
            path,
            reason,
        )
        return False

    log.info("  Verified existing %s at %s (%d audio files)", description, path, actual_audio_files)
    return True


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

    if _bootstrap_audio_dir_verified(output_dir, description=f"HF audio dataset {repo}"):
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

    _write_bootstrap_manifest(
        output_dir,
        description=f"HF audio dataset {repo}",
        expected_audio_files=count,
        metadata={"repo": repo, "split": split},
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
    if not _ensure_python_module("datacollective", MDC_PIP_SPEC):
        raise RuntimeError("Mozilla Data Collective Python SDK is required for Common Voice downloads.")

    from datacollective import download_dataset

    output_dir = _resolve_path(str(dataset_cfg.get("output_dir", "data/common_voice_pl")))
    max_clips = int(dataset_cfg.get("max_clips", 1200))
    prefix = str(dataset_cfg.get("prefix", "common_voice"))
    locale = str(dataset_cfg.get("locale", dataset_cfg.get("hf_config", "pl"))).strip().lower()
    version = str(dataset_cfg.get("version", "25.0")).strip()
    archive_dir = DATA_DIR / "_archives" / "common_voice"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if _bootstrap_audio_dir_verified(output_dir, description="Common Voice dataset"):
        return

    if not os.environ.get("MDC_API_KEY"):
        raise RuntimeError(
            "Common Voice now downloads from Mozilla Data Collective. Set MDC_API_KEY and accept dataset terms in the browser first."
        )

    dataset_id = _resolve_common_voice_dataset_id(locale=locale, version=version, dataset_cfg=dataset_cfg)
    log.info("  Downloading Common Voice %s (%s) from Mozilla Data Collective", version, locale)
    show_progress = bool(dataset_cfg.get("show_progress", True))
    archive_path = download_dataset(
        dataset_id,
        download_directory=str(archive_dir),
        show_progress=show_progress,
        overwrite_existing=False,
        enable_logging=False,
    )

    extract_dir = archive_dir / f"extract_{locale}_{version.replace('.', '_')}"
    _extract_archive(Path(archive_path), extract_dir, description=f"Common Voice {version} ({locale}) archive")
    dataset_root = _find_common_voice_dataset_root(extract_dir)
    _reset_dir(output_dir)
    count = _copy_common_voice_audio_subset(dataset_root, output_dir, prefix=prefix, max_clips=max_clips)
    _write_bootstrap_manifest(
        output_dir,
        description="Common Voice dataset",
        expected_audio_files=count,
        metadata={"locale": locale, "version": version, "dataset_id": dataset_id},
    )
    shutil.rmtree(extract_dir, ignore_errors=True)
    log.info("  Saved %d Common Voice clips to %s", count, output_dir)


def _resolve_common_voice_dataset_id(*, locale: str, version: str, dataset_cfg: dict) -> str:
    configured_id = str(dataset_cfg.get("dataset_id", "")).strip()
    if configured_id:
        return configured_id

    catalog_url = str(dataset_cfg.get("catalog_url", COMMON_VOICE_MDC_ORG_URL))
    response = requests.get(catalog_url, timeout=120)
    response.raise_for_status()
    html = response.text

    escaped_version = re.escape(version)
    escaped_locale = re.escape(locale)

    row_pattern = re.compile(
        rf"<tr\b[^>]*>.*?</tr>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    title_pattern = re.compile(
        rf'Common Voice Scripted Speech {escaped_version} - [^<"]+',
        flags=re.IGNORECASE,
    )
    link_pattern = re.compile(r'href="/datasets/([^"]+)"', flags=re.IGNORECASE)
    locale_cell_pattern = re.compile(
        rf"<td\b[^>]*>\s*{escaped_locale}\s*</td>",
        flags=re.IGNORECASE,
    )

    for row_match in row_pattern.finditer(html):
        row_html = row_match.group(0)
        if not title_pattern.search(row_html):
            continue
        if not locale_cell_pattern.search(row_html):
            continue
        link_match = link_pattern.search(row_html)
        if link_match:
            dataset_id = link_match.group(1)
            log.info("  Resolved Common Voice %s (%s) dataset id: %s", version, locale, dataset_id)
            return dataset_id

    json_pattern = re.compile(
        rf'"id":"([^"]+)","name":"Common Voice Scripted Speech {escaped_version} - [^"]+","licenseAbbreviation":"[^"]+","locale":"{escaped_locale}"',
        flags=re.IGNORECASE,
    )
    json_match = json_pattern.search(html)
    if json_match:
        dataset_id = json_match.group(1)
        log.info("  Resolved Common Voice %s (%s) dataset id: %s", version, locale, dataset_id)
        return dataset_id

    raise RuntimeError(f"Could not resolve Common Voice dataset id for locale '{locale}' and version '{version}' from Mozilla Data Collective.")


def _extract_archive(archive_path: Path, extract_dir: Path, description: str | None = None, reset_dir: bool = True) -> None:
    if reset_dir:
        _reset_dir(extract_dir)
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)
    label = description or archive_path.name
    archive_str = str(archive_path)
    if zipfile.is_zipfile(archive_path):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _extract_members_with_progress(
                    description=label,
                    members=archive.infolist(),
                    extract_one=lambda member: archive.extract(member, path=extract_dir),
                    size_getter=lambda member: getattr(member, "file_size", 0),
                )
        except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as exc:
            log.warning("  Python zip extraction failed for %s: %s", archive_path.name, exc)
            if _ensure_python_module("zipfile_inflate64", ZIPFILE_INFLATE64_PIP_SPEC):
                import zipfile_inflate64  # noqa: F401

                if reset_dir:
                    _reset_dir(extract_dir)
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        _extract_members_with_progress(
                            description=label,
                            members=archive.infolist(),
                            extract_one=lambda member: archive.extract(member, path=extract_dir),
                            size_getter=lambda member: getattr(member, "file_size", 0),
                        )
                    return
                except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as inflate_exc:
                    log.warning("  zipfile-inflate64 extraction failed for %s: %s", archive_path.name, inflate_exc)
                    if reset_dir:
                        _reset_dir(extract_dir)
            if not _extract_zip_with_external_tools(archive_path, extract_dir):
                raise
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            _extract_members_with_progress(
                description=label,
                members=archive.getmembers(),
                extract_one=lambda member: archive.extract(member, path=extract_dir),
                size_getter=lambda member: getattr(member, "size", 0),
            )
        return
    try:
        shutil.unpack_archive(archive_str, str(extract_dir))
    except (shutil.ReadError, ValueError) as exc:
        log.warning("  Generic archive extraction failed for %s: %s", archive_path.name, exc)
        if archive_path.suffix.lower() == ".zip" and _extract_zip_with_external_tools(archive_path, extract_dir):
            return
        raise


def _find_common_voice_dataset_root(extract_dir: Path) -> Path:
    if (extract_dir / "clips").exists():
        return extract_dir

    candidates: list[Path] = []
    for clips_dir in extract_dir.rglob("clips"):
        if clips_dir.is_dir():
            candidates.append(clips_dir.parent)

    if not candidates:
        raise FileNotFoundError(f"Could not find Common Voice clips directory inside {extract_dir}")

    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0]


def _iter_common_voice_relative_paths(dataset_root: Path) -> list[str]:
    validated_tsv = dataset_root / "validated.tsv"
    if validated_tsv.exists():
        with open(validated_tsv, encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames and "path" in reader.fieldnames:
                return [str(row["path"]).strip() for row in reader if str(row.get("path", "")).strip()]

    clips_root = dataset_root / "clips"
    return [path.name for path in sorted(_safe_iter_audio_files(clips_root))]


def _copy_common_voice_audio_subset(dataset_root: Path, output_dir: Path, *, prefix: str, max_clips: int) -> int:
    clips_root = dataset_root / "clips"
    if not clips_root.exists():
        raise FileNotFoundError(f"Expected Common Voice clips directory not found: {clips_root}")

    count = 0
    seen_paths: set[Path] = set()
    for relative_path in _iter_common_voice_relative_paths(dataset_root):
        if count >= max_clips:
            break
        source = clips_root / relative_path
        if not source.exists() or source in seen_paths:
            continue
        suffix = source.suffix.lower() or ".mp3"
        dest = output_dir / f"{prefix}_{count:06d}{suffix}"
        shutil.copy2(source, dest)
        seen_paths.add(source)
        count += 1

    if count == 0:
        raise RuntimeError(f"No Common Voice audio clips were copied from {dataset_root}")

    return count


def _download_voxpopuli_dataset(dataset_cfg: dict) -> None:
    merged = {
        "hf_repo": "facebook/voxpopuli",
        "hf_config": dataset_cfg.get("hf_config", "pl"),
        "output_dir": dataset_cfg.get("output_dir", "data/voxpopuli_pl"),
        "split": dataset_cfg.get("split", "train"),
        "max_clips": dataset_cfg.get("max_clips", 3000),
        "prefix": dataset_cfg.get("prefix", "voxpopuli"),
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


def _download_mls_polish_dataset(dataset_cfg: dict) -> None:
    output_dir = _resolve_path(str(dataset_cfg.get("output_dir", "data/mls_polish")))
    max_clips = int(dataset_cfg.get("max_clips", 4000))
    archive_dir = DATA_DIR / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "mls_polish.tar.gz"

    if _bootstrap_audio_dir_verified(output_dir, description="MLS Polish dataset"):
        return

    _download(
        str(dataset_cfg.get("url", "https://dl.fbaipublicfiles.com/mls/mls_polish.tar.gz")),
        archive_path,
        "MLS Polish corpus (OpenSLR)",
    )

    extract_dir = archive_dir / "mls_polish_extract"
    _extract_archive(archive_path, extract_dir, description="MLS Polish corpus archive")

    root = extract_dir / "mls_polish"
    if not root.exists():
        raise FileNotFoundError(f"Expected MLS Polish root not found in archive: {root}")

    audio_root = root / "train" / "audio"
    if not audio_root.exists():
        raise FileNotFoundError(f"Expected MLS Polish audio root not found: {audio_root}")

    _reset_dir(output_dir)
    count = 0
    for audio_file in _safe_iter_audio_files(audio_root):
        if count >= max_clips:
            break
        dest = output_dir / f"mls_polish_{count:06d}{audio_file.suffix.lower()}"
        shutil.copy2(audio_file, dest)
        count += 1

    _write_bootstrap_manifest(
        output_dir,
        description="MLS Polish dataset",
        expected_audio_files=count,
        metadata={"max_clips": max_clips},
    )
    shutil.rmtree(extract_dir, ignore_errors=True)
    log.info("  Saved %d MLS Polish clips to %s", count, output_dir)


def _download_sounds_of_home_dataset(dataset_cfg: dict) -> None:
    output_dir = _resolve_path(str(dataset_cfg.get("output_dir", "data/sounds_of_home")))
    archive_dir = DATA_DIR / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "sounds_of_home_light.zip"

    if _bootstrap_audio_dir_verified(output_dir, description="Sounds of Home dataset"):
        return

    _download(
        str(dataset_cfg.get("url", "https://www.cvssp.org/data/ai4s/sounds_of_home/VITALISE_data_light.zip")),
        archive_path,
        "Sounds of Home dataset (light)",
    )

    _extract_archive(archive_path, output_dir, description="Sounds of Home dataset (light)")

    file_count = len(_safe_iter_audio_files(output_dir))
    _write_bootstrap_manifest(
        output_dir,
        description="Sounds of Home dataset",
        expected_audio_files=file_count,
    )
    log.info("  Extracted %d Sounds of Home audio files into %s", file_count, output_dir)


def _download_bigos_dataset(dataset_cfg: dict) -> None:
    from datasets import load_dataset

    output_dir = _resolve_path(str(dataset_cfg.get("output_dir", "data/bigos")))
    max_clips = int(dataset_cfg.get("max_clips", 2000))
    split = str(dataset_cfg.get("split", "train"))
    hf_config = str(dataset_cfg.get("hf_config", "all"))
    io_workers = _resolve_io_workers(dataset_cfg)

    if _bootstrap_audio_dir_verified(output_dir, description="BIGOS dataset"):
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
    _write_bootstrap_manifest(
        output_dir,
        description="BIGOS dataset",
        expected_audio_files=count,
        metadata={"split": split, "hf_config": hf_config},
    )
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
            elif kind == "voxpopuli":
                _download_voxpopuli_dataset(dataset_cfg)
            elif kind == "mls_polish":
                _download_mls_polish_dataset(dataset_cfg)
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
            elif kind == "sounds_of_home":
                _download_sounds_of_home_dataset(dataset_cfg)
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
    if not _background_segmentation_enabled(cfg):
        return _stage_audio_sources(
            staged_dir=_staged_background_dir(cfg),
            manifest_path=_project_dir(cfg) / "staged_background_sources.json",
            source_dirs=source_dirs,
            label="background",
        )

    staged_dir = _staged_background_dir(cfg)
    manifest_path = _project_dir(cfg) / "staged_background_sources.json"
    seg_cfg = _background_segmentation_cfg(cfg)
    source_manifest = {
        "sources": [str(path) for path in source_dirs],
        "segment_duration_s": float(seg_cfg.get("segment_duration_s", 5.0)),
        "segment_overlap_s": float(seg_cfg.get("segment_overlap_s", 2.5)),
        "min_segment_duration_s": float(seg_cfg.get("min_segment_duration_s", 4.0)),
    }

    if staged_dir.exists() and manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_manifest = None
        if current_manifest == source_manifest and _safe_iter_audio_files(staged_dir):
            log.info("  Using segmented staged background audio at %s", staged_dir)
            return staged_dir

    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_dir in source_dirs:
        for audio_file in _safe_iter_audio_files(source_dir):
            count += _segment_file_into_dir(
                audio_file,
                staged_dir,
                prefix=f"{source_dir.name}_{count:06d}",
                segment_duration_s=float(seg_cfg.get("segment_duration_s", 5.0)),
                segment_overlap_s=float(seg_cfg.get("segment_overlap_s", 2.5)),
                min_segment_duration_s=float(seg_cfg.get("min_segment_duration_s", 4.0)),
            )

    if count == 0:
        raise ValueError("No background audio files were found in the configured source directories")

    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    log.info("  Segmented %d background audio clips into %s", count, staged_dir)
    return staged_dir


def _generated_background_negatives_cfg(cfg: dict) -> dict:
    return cfg.get("generated_background_negatives", {}) or {}


def _generated_background_negatives_enabled(cfg: dict) -> bool:
    return _task(cfg) == "vad" and bool(_generated_background_negatives_cfg(cfg).get("enabled", False))


def _generated_background_negative_manifest_path(cfg: dict) -> Path:
    return _project_dir(cfg) / "generated_background_negative_features.json"


def _positive_segmentation_cfg(cfg: dict) -> dict:
    return cfg.get("positive_segmentation", {}) or {}


def _positive_segmentation_enabled(cfg: dict) -> bool:
    return _task(cfg) == "vad" and bool(_positive_segmentation_cfg(cfg).get("enabled", False))


def _segmented_positive_root(cfg: dict) -> Path:
    return _project_dir(cfg) / "segmented_positive_audio"


def _segmented_positive_manifest_path(cfg: dict) -> Path:
    return _project_dir(cfg) / "segmented_positive_audio.json"


def _background_segmentation_cfg(cfg: dict) -> dict:
    return cfg.get("background_segmentation", {}) or {}


def _background_segmentation_enabled(cfg: dict) -> bool:
    return bool(_background_segmentation_cfg(cfg).get("enabled", False))


def _split_dir_name(split_name: str) -> str:
    return "training" if split_name == "train" else split_name


def _extract_audio_from_file(path: Path) -> tuple["np.ndarray", int]:
    import numpy as np
    import soundfile as sf

    samples, sampling_rate = sf.read(str(path), dtype="float32")
    samples = np.asarray(samples, dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=-1, dtype="float32")
    return np.asarray(samples, dtype="float32"), int(sampling_rate)


def _segment_file_into_dir(
    src: Path,
    dest_dir: Path,
    prefix: str,
    segment_duration_s: float,
    segment_overlap_s: float = 0.0,
    min_segment_duration_s: float | None = None,
) -> int:
    samples, sampling_rate = _extract_audio_from_file(src)
    count = 0
    for count, chunk in enumerate(
        _segment_audio_samples(
            samples,
            sampling_rate,
            segment_duration_s=segment_duration_s,
            segment_overlap_s=segment_overlap_s,
            min_segment_duration_s=min_segment_duration_s,
        ),
        start=1,
    ):
        _write_audio_file(dest_dir / f"{prefix}_{count:03d}.wav", chunk, sampling_rate)
    return count


def _split_source_files(files: list[Path], split_count: float, seed: int) -> dict[str, list[Path]]:
    shuffled = sorted(files, key=lambda path: str(path))
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    holdout = int(round(total * split_count))
    validation_count = min(holdout, total)
    test_count = min(holdout, max(0, total - validation_count))
    train_count = max(0, total - validation_count - test_count)

    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count:train_count + validation_count],
        "test": shuffled[train_count + validation_count:train_count + validation_count + test_count],
    }


def _segmented_positive_split_dirs(cfg: dict) -> dict[str, Path]:
    root = _segmented_positive_root(cfg)
    return {
        "train": root / "training",
        "validation": root / "validation",
        "test": root / "testing",
    }


def _prepare_segmented_positive_splits(cfg: dict, source_dirs: list[Path]) -> dict[str, Path]:
    seg_cfg = _positive_segmentation_cfg(cfg)
    split_dirs = _segmented_positive_split_dirs(cfg)
    manifest_path = _segmented_positive_manifest_path(cfg)

    source_manifest = {
        "sources": [str(path) for path in source_dirs],
        "segment_duration_s": float(seg_cfg.get("segment_duration_s", 5.0)),
        "segment_overlap_s": float(seg_cfg.get("segment_overlap_s", 2.5)),
        "min_segment_duration_s": float(seg_cfg.get("min_segment_duration_s", 4.0)),
        "split_count": float(cfg.get("split_count", 0.1)),
        "random_split_seed": int(cfg.get("random_split_seed", 10)),
    }

    if all(_dir_has_entries(path) for path in split_dirs.values()) and manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_manifest = None
        if current_manifest == source_manifest:
            log.info("  Using segmented positive splits at %s", _segmented_positive_root(cfg))
            return split_dirs

    root = _segmented_positive_root(cfg)
    if root.exists():
        shutil.rmtree(root)
    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for source_dir in source_dirs:
        files.extend(_safe_iter_audio_files(source_dir))

    if not files:
        raise ValueError("No positive audio files were found for segmentation")

    splits = _split_source_files(
        files,
        split_count=float(cfg.get("split_count", 0.1)),
        seed=int(cfg.get("random_split_seed", 10)),
    )

    segment_duration_s = float(seg_cfg.get("segment_duration_s", 5.0))
    segment_overlap_s = float(seg_cfg.get("segment_overlap_s", 2.5))
    min_segment_duration_s = float(seg_cfg.get("min_segment_duration_s", 4.0))

    segment_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for split_name, split_files in splits.items():
        dest_dir = split_dirs[split_name]
        for index, audio_file in enumerate(split_files):
            segment_counts[split_name] += _segment_file_into_dir(
                audio_file,
                dest_dir,
                prefix=f"{audio_file.stem}_{index:06d}",
                segment_duration_s=segment_duration_s,
                segment_overlap_s=segment_overlap_s,
                min_segment_duration_s=min_segment_duration_s,
            )

    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8")
    log.info(
        "  Segmented positive splits at %s: train=%d validation=%d test=%d",
        root,
        segment_counts["train"],
        segment_counts["validation"],
        segment_counts["test"],
    )
    return split_dirs


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

    if _bootstrap_audio_dir_verified(dest, description="MIT RIR dataset"):
        return

    dest.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    count = _write_dataset_audio(dataset, dest, "rir", None, io_workers)
    _write_bootstrap_manifest(dest, description="MIT RIR dataset", expected_audio_files=count)
    log.info("  Saved %d RIR files", count)


def _download_musan_openslr(dest: Path) -> None:
    archive_dir = DATA_DIR / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "musan.tar.gz"

    if _bootstrap_audio_dir_verified(dest, description="MUSAN music/noise dataset"):
        return

    _download(
        "https://openslr.trmal.net/resources/17/musan.tar.gz",
        archive_path,
        "MUSAN corpus (OpenSLR)",
    )

    extract_dir = archive_dir / "musan_extract"
    _extract_archive(archive_path, extract_dir, description="MUSAN corpus archive")

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

    _write_bootstrap_manifest(dest, description="MUSAN music/noise dataset", expected_audio_files=copied)
    shutil.rmtree(extract_dir, ignore_errors=True)
    log.info("  Prepared %d MUSAN music/noise files in %s", copied, dest)


def _download_audioset_subset(dest: Path, limit: int = 300, io_workers: int = 4) -> None:
    from datasets import load_dataset

    if _bootstrap_audio_dir_verified(dest, description="AudioSet subset"):
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
    _write_bootstrap_manifest(dest, description="AudioSet subset", expected_audio_files=count, metadata={"limit": limit})
    log.info("  Saved %d AudioSet clips", count)


def _download_fma_subset(dest: Path, limit: int = 200, io_workers: int = 4) -> None:
    from datasets import load_dataset

    if _bootstrap_audio_dir_verified(dest, description="FMA subset"):
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

    _write_bootstrap_manifest(dest, description="FMA subset", expected_audio_files=count, metadata={"limit": limit})
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
    _extract_archive(zip_path, dest_root, description=f"negative feature pack {name}", reset_dir=False)

    valid, reason = validate_pack(target_dir)
    if not valid:
        raise ValueError(f"negative feature pack {name} is still invalid after download: {reason}")

    log.info("  Extracted negative feature pack: %s", name)


def _negative_feature_names(cfg: dict) -> list[str]:
    names = cfg.get("negative_feature_sets")
    if names:
        return [str(name) for name in names]
    return ["speech", "dinner_party", "no_speech", "dinner_party_eval"]


def _base_background_audio_paths(cfg: dict) -> list[str]:
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


def _resolve_background_audio_paths(cfg: dict) -> list[str]:
    resolved = _base_background_audio_paths(cfg)

    if _background_segmentation_enabled(cfg) and resolved:
        staged = _stage_background_sources(cfg, [Path(path) for path in resolved])
        return [str(staged)]

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
            if _positive_segmentation_enabled(cfg):
                _prepare_segmented_positive_splits(cfg, source_dirs)
                return True
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
    except Exception as exc:
        log.error("  Audit dependencies are not available: %s", exc)
        return False

    split_paths: dict[str, list[str]] = {}
    split_hashes: dict[str, set[str]] = {}
    if _positive_segmentation_enabled(cfg):
        split_dirs = _segmented_positive_split_dirs(cfg)
        if not all(_dir_has_entries(path) for path in split_dirs.values()):
            try:
                source_dirs = [path for path in _resolve_positive_sources(cfg) if path.exists()]
                _prepare_segmented_positive_splits(cfg, source_dirs)
            except Exception as exc:
                log.error("  Failed to prepare segmented positive splits for audit: %s", exc)
                return False

        for split_name, directory in split_dirs.items():
            paths = [str(path) for path in _safe_iter_audio_files(directory)]
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
    else:
        try:
            from microwakeword.audio.clips import Clips
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

    features_dir.mkdir(parents=True, exist_ok=True)

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

    if _positive_segmentation_enabled(cfg):
        source_dirs = [path for path in _resolve_positive_sources(cfg) if path.exists()]
        split_source_dirs = _prepare_segmented_positive_splits(cfg, source_dirs)
        split_plan = (
            ("training", split_source_dirs["train"], 1, 10),
            ("validation", split_source_dirs["validation"], 1, 10),
            ("testing", split_source_dirs["test"], 1, 1),
        )
    else:
        positive_dir, file_pattern = _positive_source(cfg)
        clips = Clips(
            input_directory=str(positive_dir),
            file_pattern=file_pattern,
            max_clip_duration_s=None,
            remove_silence=bool(cfg.get("remove_silence", False)),
            random_split_seed=int(cfg.get("random_split_seed", 10)),
            split_count=float(cfg.get("split_count", 0.1)),
        )
        split_plan = None

    for split in ("training", "validation", "testing"):
        out_dir = features_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        repetition = 2
        slide_frames = 10
        clip_split = "train"
        split_dir = None
        if split == "validation":
            repetition = 1
            clip_split = "validation"
        elif split == "testing":
            repetition = 1
            slide_frames = 1
            clip_split = "test"

        if split_plan is not None:
            for split_name_key, split_source_dir, split_repetition, split_slide_frames in split_plan:
                if split_name_key == split:
                    split_dir = split_source_dir
                    repetition = split_repetition
                    slide_frames = split_slide_frames
                    clip_split = "train"
                    break

            if split_dir is None:
                raise ValueError(f"Missing segmented positive directory for split {split}")

            clips = Clips(
                input_directory=str(split_dir),
                file_pattern="*.*",
                max_clip_duration_s=None,
                remove_silence=bool(cfg.get("remove_silence", False)),
                random_split_seed=int(cfg.get("random_split_seed", 10)),
                # Upstream Clips requires a positive holdout size, even when we already
                # split source files ourselves. Using an integer of 1 keeps the
                # secondary split negligible while avoiding leakage across source files.
                split_count=1,
            )

        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectrograms.spectrogram_generator(
                split=clip_split,
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

    source_dirs = [Path(path) for path in _base_background_audio_paths(cfg)]
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
