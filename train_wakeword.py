#!/usr/bin/env python3
"""
train_wakeword.py — Granular custom wake word training pipeline.

Each stage has a **do** step and a **verify** step so problems surface
immediately instead of cascading.  You can run the full pipeline, resume
from any step, or run a single step in isolation.

Usage:
    python train_wakeword.py                                  # full pipeline
    python train_wakeword.py --config configs/wakeword_example.yaml
    python train_wakeword.py --from augment                   # resume
    python train_wakeword.py --step verify-clips              # one step
    python train_wakeword.py --verify-only                    # check state
    python train_wakeword.py --list-steps                     # show steps

Run inside WSL2 with CUDA support.
See README.md for full setup instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Callable

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # project root
DATA_DIR   = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "wakeword_example.yaml"
RESOLVED_CONFIG = OUTPUT_DIR / "_resolved_config.yaml"
OWW_WRAPPER = SCRIPT_DIR / "oww_wrapper.py"

# Will be set by CLI --config flag or default
CONFIG_FILE: Path = DEFAULT_CONFIG

# Remote URLs — all public, no auth required
URLS = {
    "acav100m_features": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features"
        "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    ),
    "validation_features": (
        "https://huggingface.co/datasets/davidscripka/openwakeword_features"
        "/resolve/main/validation_set_features.npy"
    ),
    "piper_model": (
        "https://github.com/rhasspy/piper-sample-generator/releases/download"
        "/v2.0.0/en_US-libritts_r-medium.pt"
    ),
    "piper_repo": "https://github.com/rhasspy/piper-sample-generator.git",
}

# Minimum expected file sizes (bytes) for data verification
MIN_SIZES = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": 5_000_000_000,   # ~7.5 GB
    "validation_set_features.npy":                         30_000_000,      # ~40 MB
    "piper-sample-generator/models/en_US-libritts_r-medium.pt": 600_000_000,  # ~800 MB
}

DATASET_REGISTRY = {
    "mc_speech": {
        "kind": "positive",
        "default_path": DATA_DIR / "mc_speech",
        "description": "Polish speech clips (MC Speech, user-supplied)",
        "placeholder": True,
    },
    "bigos": {
        "kind": "positive",
        "default_path": DATA_DIR / "bigos",
        "description": "Polish speech clips (BIGOS, user-supplied)",
        "placeholder": False,
    },
    "pl_speech": {
        "kind": "positive",
        "default_path": DATA_DIR / "pl_speech",
        "description": "Additional Polish speech clips (user-supplied)",
        "placeholder": True,
    },
    "no_speech": {
        "kind": "negative",
        "default_path": DATA_DIR / "no_speech",
        "description": "Silence / HVAC / room tone negatives",
        "placeholder": False,
    },
    "dinner_party": {
        "kind": "negative",
        "default_path": DATA_DIR / "dinner_party",
        "description": "Crowd / babble / kitchen ambience negatives",
        "placeholder": False,
    },
    "musan": {
        "kind": "negative",
        "default_path": DATA_DIR / "musan",
        "description": "MUSAN music/noise/speech negatives",
        "placeholder": False,
    },
    "fma": {
        "kind": "negative",
        "default_path": DATA_DIR / "fma_small",
        "description": "FMA music negatives",
        "placeholder": False,
    },
    "audioset": {
        "kind": "negative",
        "default_path": DATA_DIR / "audioset_16k",
        "description": "AudioSet background negatives",
        "placeholder": False,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_wakeword")


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run(cmd: list[str] | str, cwd: str | Path | None = None, **kw) -> None:
    """Run a subprocess, streaming output.  Raises on failure."""
    log.info("$ %s", cmd if isinstance(cmd, str) else " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, **kw)


def _download(url: str, dest: Path, description: str = "") -> None:
    """Download *url* → *dest* with progress.  Skips if *dest* exists."""
    if dest.exists():
        log.info("  Already downloaded: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = description or dest.name
    log.info("  Downloading %s …", label)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(".part")
    downloaded = 0
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / (1 << 20)
                total_mb = total / (1 << 20)
                print(f"\r  {label}: {mb:.0f}/{total_mb:.0f} MB ({pct}%)",
                      end="", flush=True)
    print()
    tmp.rename(dest)
    log.info("  Saved %s", dest)


def _clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        log.info("  Repo already cloned: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)])


def _load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_FILE) as f:
        return yaml.safe_load(f)


def _load_active_config() -> dict:
    """Prefer the resolved config once available so later steps see CLI overrides."""
    if RESOLVED_CONFIG.exists():
        return _load_config(RESOLVED_CONFIG)
    return _load_config(CONFIG_FILE)


def _require_local_helper(path: Path, purpose: str) -> bool:
    """Fail with a clear message when required helper files are missing."""
    if path.exists():
        return True
    log.error("  Missing required helper: %s", path)
    log.error("  %s", purpose)
    log.error("  This repo contains the pipeline entrypoint, but that helper is not present.")
    return False


def _dataset_overrides_from_env() -> dict[str, str]:
    raw = os.environ.get("OWW_DATASET_PATHS", "").strip()
    overrides = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        overrides[name.strip()] = value.strip()
    return overrides


def _use_large_acav_features(cfg: dict) -> bool:
    raw = os.environ.get("OWW_USE_ACAV100M_FEATURES")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(cfg.get("use_acav100m_features", False))


def _get_mode(cfg: dict) -> str:
    return str(cfg.get("mode") or cfg.get("training_type") or "wakeword").lower()


def _get_dataset_names(cfg: dict, kind: str) -> list[str]:
    datasets = cfg.get("datasets", {}) or {}
    names = datasets.get(kind, []) or []
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return [str(n).strip() for n in names if str(n).strip()]


def _resolve_dataset_paths(cfg: dict, kind: str) -> list[tuple[str, Path]]:
    overrides = {**(cfg.get("dataset_paths", {}) or {}), **_dataset_overrides_from_env()}
    resolved = []
    for name in _get_dataset_names(cfg, kind):
        meta = DATASET_REGISTRY.get(name, {})
        raw_path = overrides.get(name)
        if raw_path:
            path = Path(raw_path).expanduser()
        else:
            path = meta.get("default_path", DATA_DIR / name)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()
        resolved.append((name, path))
    return resolved


def _ensure_placeholder_dataset(name: str, path: Path, description: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    readme = path / "README.txt"
    if not readme.exists():
        readme.write_text(
            f"Placeholder dataset: {name}\n\n{description}\n\n"
            "Put .wav/.flac/.mp3 files here or pass --dataset-path name=/abs/path.\n",
            encoding="utf-8",
        )


def _dataset_has_audio(path: Path) -> bool:
    return path.exists() and any(_iter_audio_files(path))


def _ensure_vad_dataset_ready(name: str, path: Path) -> None:
    if _dataset_has_audio(path):
        return

    if name == "bigos":
        _download_bigos_subset(path)
    elif name == "no_speech":
        _generate_no_speech_dataset(path)
    elif name == "dinner_party":
        _generate_dinner_party_dataset(path)


# ═══════════════════════════════════════════════════════════════════════════
# STEP IMPLEMENTATIONS
# Each returns True on success, False on failure.
# ═══════════════════════════════════════════════════════════════════════════


# ── 1. check-env ──────────────────────────────────────────────────────────

def step_check_env() -> bool:
    """Verify Python ≥3.10, CUDA availability, and critical imports."""
    ok = True

    # Python version
    v = sys.version_info
    log.info("  Python %d.%d.%d  (%s)", v.major, v.minor, v.micro, sys.executable)
    if v < (3, 10):
        log.error("  Python ≥3.10 required")
        ok = False

    # Platform
    import platform
    log.info("  Platform: %s", platform.platform())
    if platform.system() != "Linux":
        log.error("  This script must run inside WSL2 (Linux)")
        ok = False

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            log.info("  CUDA: %s  (%.1f GB)", gpu, mem)
        else:
            log.warning("  CUDA not available — training will be very slow on CPU")
    except ImportError:
        log.error("  PyTorch not installed")
        ok = False

    # Critical imports
    for mod in ["yaml", "requests", "soundfile", "numpy", "scipy"]:
        try:
            __import__(mod)
        except ImportError:
            log.error("  Missing: %s", mod)
            ok = False

    # Config file
    log.info("  Config: %s  (%s)", CONFIG_FILE, "exists" if CONFIG_FILE.exists() else "MISSING")
    if not CONFIG_FILE.exists():
        log.error("  Config file not found: %s", CONFIG_FILE)
        ok = False

    return ok


# ── 2. apply-patches ─────────────────────────────────────────────────────

def step_apply_patches() -> bool:
    """Apply compatibility monkey-patches and verify they work."""
    compat_path = SCRIPT_DIR / "compat.py"
    if not _require_local_helper(
        compat_path,
        "The apply-patches step needs compat.py to patch torchaudio, speechbrain, and Piper.",
    ):
        return False
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import compat

    log.info("  Applying patches …")
    results = compat.apply_all()
    failed_apply = [k for k, v in results.items() if "FAIL" in v]
    if failed_apply:
        log.error("  Patch application failed: %s", failed_apply)
        return False

    log.info("  Verifying patches …")
    checks = compat.verify_all()
    failed_verify = [k for k, v in checks.items() if not v]
    if failed_verify:
        log.error("  Patch verification failed: %s", failed_verify)
        return False

    log.info("  All patches applied and verified")
    return True


# ── 3. download ──────────────────────────────────────────────────────────

def step_download() -> bool:
    """Download all datasets, tools, and models.  Idempotent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    mode = _get_mode(cfg)

    # 3a — Piper Sample Generator repo (wake word mode only)
    piper_dir = DATA_DIR / "piper-sample-generator"
    if mode != "vad":
        _clone_repo(URLS["piper_repo"], piper_dir)

        piper_marker = piper_dir / ".installed"
        if not piper_marker.exists():
            log.info("  Installing piper-sample-generator (editable) …")
            _run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=piper_dir)
            piper_marker.touch()

        # 3b — Piper TTS model
        piper_models_dir = piper_dir / "models"
        piper_models_dir.mkdir(exist_ok=True)
        _download(
            URLS["piper_model"],
            piper_models_dir / "en_US-libritts_r-medium.pt",
            "Piper LibriTTS model (~800 MB)",
        )
    else:
        log.info("  VAD mode: skipping Piper download (positives come from datasets)")

    # 3c — ACAV100M pre-computed negative features
    if mode != "vad" or _use_large_acav_features(cfg):
        _download(
            URLS["acav100m_features"],
            DATA_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "ACAV100M negative features (~7.5 GB)",
        )
    else:
        log.info("  VAD mode: skipping ACAV100M negative features download (set OWW_USE_ACAV100M_FEATURES=1 to enable)")

    # 3d — Validation features
    _download(
        URLS["validation_features"],
        DATA_DIR / "validation_set_features.npy",
        "Validation features (~40 MB)",
    )

    # 3e — MIT Room Impulse Responses
    rir_dir = DATA_DIR / "mit_rirs"
    if not rir_dir.exists():
        _download_mit_rirs(rir_dir)
    else:
        log.info("  MIT RIRs already present")

    selected_negative = set(_get_dataset_names(cfg, "negative"))
    if not selected_negative or "audioset" in selected_negative:
        audioset_dir = DATA_DIR / "audioset_16k"
        if not audioset_dir.exists():
            _download_audioset_subset(audioset_dir)
        else:
            log.info("  AudioSet subset already present")

    if not selected_negative or "fma" in selected_negative:
        fma_dir = DATA_DIR / "fma_small"
        if not fma_dir.exists():
            _download_fma_subset(fma_dir)
        else:
            log.info("  FMA subset already present")

    if "musan" in selected_negative:
        musan_dir = DATA_DIR / "musan"
        if not musan_dir.exists():
            _download_musan_subset(musan_dir)
        else:
            log.info("  MUSAN subset already present")

    for name, path in _resolve_dataset_paths(cfg, "positive"):
        _ensure_vad_dataset_ready(name, path)

    for name, path in _resolve_dataset_paths(cfg, "negative"):
        _ensure_vad_dataset_ready(name, path)

    for kind in ("positive", "negative"):
        for name, path in _resolve_dataset_paths(cfg, kind):
            meta = DATASET_REGISTRY.get(name, {})
            if meta.get("placeholder"):
                _ensure_placeholder_dataset(name, path, meta.get("description", name))

    return True


# ── 4. verify-data ───────────────────────────────────────────────────────

def step_verify_data() -> bool:
    """Check every expected download exists with minimum file sizes."""
    ok = True
    cfg = _load_config()
    mode = _get_mode(cfg)

    # Large feature / model files
    for relpath, min_bytes in MIN_SIZES.items():
        if mode == "vad" and relpath.endswith("en_US-libritts_r-medium.pt"):
            continue
        if mode == "vad" and relpath.endswith("openwakeword_features_ACAV100M_2000_hrs_16bit.npy") and not _use_large_acav_features(cfg):
            continue
        fp = DATA_DIR / relpath
        if not fp.exists():
            log.error("  MISSING: %s", fp)
            ok = False
        elif fp.stat().st_size < min_bytes:
            log.error(
                "  TOO SMALL: %s  (%d bytes, expected ≥%d)",
                fp, fp.stat().st_size, min_bytes,
            )
            ok = False
        else:
            sz_mb = fp.stat().st_size / (1 << 20)
            log.info("  OK: %-60s  %.0f MB", relpath, sz_mb)

    # Directories that should contain files
    for name in ["mit_rirs", "audioset_16k", "fma_small"]:
        d = DATA_DIR / name
        if not d.is_dir():
            log.error("  MISSING dir: %s", d)
            ok = False
        else:
            n = len(list(d.iterdir()))
            if n == 0:
                log.error("  EMPTY dir: %s", d)
                ok = False
            else:
                log.info("  OK: %-60s  %d files", name + "/", n)

    # Piper install marker
    marker = DATA_DIR / "piper-sample-generator" / ".installed"
    if mode != "vad":
        if not marker.exists():
            log.error("  Piper not installed (run download step)")
            ok = False
        else:
            log.info("  OK: piper-sample-generator installed")
    else:
        log.info("  VAD mode: Piper install not required")

    for kind in ("positive", "negative"):
        for name, path in _resolve_dataset_paths(cfg, kind):
            if path.exists():
                n = len(list(path.rglob("*")))
                log.info("  Dataset %-18s %s  (%d filesystem entries)", name, path, n)
            else:
                log.warning("  Dataset %-18s missing at %s", name, path)

    return ok


# ── 5. resolve-config ────────────────────────────────────────────────────

def step_resolve_config() -> bool:
    """Read config YAML, resolve relative paths → absolute, write output."""
    cfg = _load_config()
    mode = _get_mode(cfg)
    cfg["mode"] = mode

    if "piper_sample_generator_path" in cfg:
        cfg["piper_sample_generator_path"] = str(
            (SCRIPT_DIR / cfg["piper_sample_generator_path"]).resolve()
        )
    cfg["output_dir"] = str((SCRIPT_DIR / cfg["output_dir"]).resolve())
    os.makedirs(cfg["output_dir"], exist_ok=True)

    cfg["rir_paths"] = [
        str((SCRIPT_DIR / p).resolve()) for p in cfg.get("rir_paths", [])
    ]
    cfg["background_paths"] = [
        str((SCRIPT_DIR / p).resolve()) for p in cfg.get("background_paths", [])
    ]
    resolved_features = {}
    for key, relpath in cfg.get("feature_data_files", {}).items():
        resolved_features[key] = str((SCRIPT_DIR / relpath).resolve())
    cfg["feature_data_files"] = resolved_features

    if mode == "vad" and not _use_large_acav_features(cfg):
        cfg["feature_data_files"] = {}

    if "false_positive_validation_data_path" in cfg:
        cfg["false_positive_validation_data_path"] = str(
            (SCRIPT_DIR / cfg["false_positive_validation_data_path"]).resolve()
        )

    dataset_paths = {}
    for kind in ("positive", "negative"):
        dataset_paths[kind] = {name: str(path) for name, path in _resolve_dataset_paths(cfg, kind)}
    cfg["resolved_dataset_paths"] = dataset_paths

    export_manifest = cfg.get("export_manifest", {}) or {}
    if mode == "vad":
        export_manifest.setdefault("esphome_vad", True)
        export_manifest.setdefault("wake_word", "vad")
        export_manifest.setdefault("trained_languages", ["pl"])
        export_manifest.setdefault("author", "ha-wakeword-trainer")
    cfg["export_manifest"] = export_manifest

    RESOLVED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    log.info("  Resolved config → %s", RESOLVED_CONFIG)

    # Quick sanity: print key paths
    for key in ["piper_sample_generator_path", "output_dir",
                "false_positive_validation_data_path"]:
        if key in cfg:
            p = Path(cfg[key])
            exists = p.exists()
            log.info("    %-40s %s  %s", key, "✓" if exists else "✗", p)
            if not exists:
                log.warning("    ^ path does not exist yet (may be created later)")

    return True


# ── 6. generate ──────────────────────────────────────────────────────────

def _iter_audio_files(root: Path) -> list[Path]:
    exts = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts]


def _resample_audio_file(src: Path, dest: Path) -> None:
    import numpy as np
    import soundfile as sf
    from scipy import signal

    data, sr = sf.read(str(src), always_2d=False)
    if hasattr(data, 'ndim') and data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000 and len(data) > 0:
        target_len = max(1, round(len(data) * 16000 / sr))
        data = signal.resample(data, target_len).astype(np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), data, 16000)


def _build_vad_fallback_sentences() -> list[str]:
    prompts = [
        "Wlacz swiatlo w salonie.",
        "Wylacz telewizor w sypialni.",
        "Jaka bedzie pogoda jutro rano?",
        "Ustaw temperature na dwadziescia dwa stopnie.",
        "Odtworz spokojna muzyke.",
        "Zamknij rolety w kuchni.",
        "Czy frontowe drzwi sa zamkniete?",
        "Nastepny alarm ustaw na siodma rano.",
        "Wlacz odkurzacz za dziesiec minut.",
        "Pokaz zuzycie energii dzisiaj.",
        "Przelacz lampke nocna na cieply kolor.",
        "Czy pralka skonczyla pranie?",
    ]
    variants = [
        "Prosze {base}",
        "{base}",
        "Hej asystencie, {base}",
        "Mozesz {base}",
        "Sprawdz prosze: {base}",
    ]
    sentences = []
    for base in prompts:
        for variant in variants:
            sentences.append(variant.format(base=base.lower()))
    return sentences


async def _generate_edge_tts_dataset(dest: Path, count: int = 120) -> int:
    import edge_tts

    voices = [
        "pl-PL-MarekNeural",
        "pl-PL-ZofiaNeural",
    ]
    sentences = _build_vad_fallback_sentences()
    random.seed(42)
    dest.mkdir(parents=True, exist_ok=True)

    generated = 0
    for idx in range(count):
        text = random.choice(sentences)
        voice = voices[idx % len(voices)]
        out_path = dest / f"synthetic_pl_{idx:04d}.mp3"
        if out_path.exists():
            generated += 1
            continue
        try:
            await edge_tts.Communicate(text=text, voice=voice).save(str(out_path))
            generated += 1
        except Exception as exc:
            log.warning("  Edge TTS fallback failed for sample %d: %s", idx, exc)
            break
    return generated


def _ensure_vad_positive_fallback(min_count: int = 120) -> list[Path]:
    fallback_dir = DATA_DIR / "synthetic_pl_speech"
    existing = _iter_audio_files(fallback_dir) if fallback_dir.exists() else []
    if len(existing) >= min_count:
        log.info("  Using bundled fallback speech dataset: %d clips", len(existing))
        return existing

    log.warning("  No positive speech datasets found. Generating fallback Polish speech with Edge TTS...")
    try:
        generated = asyncio.run(_generate_edge_tts_dataset(fallback_dir, count=min_count))
    except ImportError:
        log.error("  edge-tts is not installed, so fallback speech generation is unavailable.")
        return []
    except Exception as exc:
        log.error("  Failed to generate fallback Polish speech dataset: %s", exc)
        return []

    if generated == 0:
        return []

    return _iter_audio_files(fallback_dir)


def _prepare_vad_training_clips(cfg: dict) -> bool:
    model_name = cfg.get('model_name', 'vad')
    model_dir = OUTPUT_DIR / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)
    splits = {
        'positive_train': [],
        'positive_test': [],
        'negative_train': [],
        'negative_test': [],
    }

    pos_limit = int(cfg.get('vad_positive_samples', cfg.get('n_samples', 12000)))
    neg_limit = int(cfg.get('vad_negative_samples', max(pos_limit, 12000)))
    val_limit = int(cfg.get('vad_validation_samples', cfg.get('n_samples_val', max(1000, pos_limit // 10))))

    positive_files = []
    for name, path in _resolve_dataset_paths(cfg, 'positive'):
        _ensure_vad_dataset_ready(name, path)
        if path.exists():
            positive_files.extend(_iter_audio_files(path))
        else:
            log.warning('  Positive dataset missing: %s -> %s', name, path)

    if len(positive_files) < 2:
        fallback_positive = _ensure_vad_positive_fallback()
        if fallback_positive:
            positive_files.extend(fallback_positive)

    negative_files = []
    for name, path in _resolve_dataset_paths(cfg, 'negative'):
        _ensure_vad_dataset_ready(name, path)
        if path.exists():
            negative_files.extend(_iter_audio_files(path))
        else:
            log.warning('  Negative dataset missing: %s -> %s', name, path)

    if len(positive_files) < 2:
        log.error('  VAD mode needs real speech audio. Found only %d positive files.', len(positive_files))
        return False
    if len(negative_files) < 2:
        log.error('  VAD mode needs non-speech audio. Found only %d negative files.', len(negative_files))
        return False

    pos_train = positive_files[:pos_limit]
    pos_test = positive_files[pos_limit:pos_limit + val_limit] or positive_files[:val_limit]
    neg_train = negative_files[:neg_limit]
    neg_test = negative_files[neg_limit:neg_limit + val_limit] or negative_files[:val_limit]

    for split_name, files in {
        'positive_train': pos_train,
        'positive_test': pos_test,
        'negative_train': neg_train,
        'negative_test': neg_test,
    }.items():
        split_dir = model_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(files):
            dest = split_dir / f'{split_name}_{idx:05d}.wav'
            _resample_audio_file(src, dest)
        log.info('  Prepared %-16s %5d clips', split_name, len(files))

    return True


def step_generate() -> bool:
    """Generate positive + negative clips via Piper TTS or prepare VAD datasets."""
    if not RESOLVED_CONFIG.exists():
        log.error("  Resolved config not found — run 'resolve-config' first")
        return False

    cfg = _load_config(RESOLVED_CONFIG)
    if _get_mode(cfg) == 'vad':
        log.info('  VAD mode: preparing speech/non-speech clips from datasets …')
        return _prepare_vad_training_clips(cfg)

    log.info("  Generating clips via openwakeword + Piper TTS …")
    log.info("  (Longest step — ~10 min on GPU, hours on CPU)")

    if not _require_local_helper(
        OWW_WRAPPER,
        "Wake word clip generation depends on oww_wrapper.py calling the openWakeWord backend.",
    ):
        return False

    try:
        _run([
            sys.executable, str(OWW_WRAPPER),
            "--training_config", str(RESOLVED_CONFIG),
            "--generate_clips",
        ])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("  Clip generation failed (exit code %d)", exc.returncode)
        return False


# ── 7. resample-clips ────────────────────────────────────────────────────

def step_resample_clips() -> bool:
    """Verify clips exist and note sample rates.

    Actual resampling is handled on-the-fly by the patched torchaudio.load
    in compat.py (applied during the ``apply-patches`` step).  This avoids
    the extremely slow bulk rewrite of 100k+ WAV files.

    This step just spot-checks a few files and warns if rates differ from 16 kHz.
    """
    import soundfile as sf

    wav_files = list(OUTPUT_DIR.rglob("*.wav"))
    if not wav_files:
        log.warning("  No WAV files found in %s", OUTPUT_DIR)
        return True

    log.info("  Found %d WAV files in output/", len(wav_files))

    # Spot-check first 5 files from each subdirectory
    checked = 0
    non_16k = 0
    for d in sorted(set(f.parent for f in wav_files)):
        samples = sorted(d.glob("*.wav"))[:5]
        for wav in samples:
            try:
                info = sf.info(str(wav))
                if info.samplerate != 16000:
                    non_16k += 1
                    if non_16k <= 3:
                        log.info("    %s → %d Hz (will be resampled on-the-fly)", wav.name, info.samplerate)
                checked += 1
            except Exception as exc:
                log.warning("    Error reading %s: %s", wav.name, exc)

    if non_16k > 0:
        log.info("  %d/%d spot-checked files are not 16 kHz — compat patch will handle this", non_16k, checked)
    else:
        log.info("  All %d spot-checked files are 16 kHz", checked)

    return True


# ── 8. verify-clips ──────────────────────────────────────────────────────

def step_verify_clips() -> bool:
    """Verify clip counts and sample rates in output/."""
    import soundfile as sf

    ok = True

    cfg = _load_active_config()

    expected_positive = cfg.get("n_samples", 50000)
    model_name = cfg.get("model_name", "my_wakeword")

    # Clips live in output/<model_name>/{positive_train, positive_test, ...}
    model_dir = OUTPUT_DIR / model_name
    if not model_dir.is_dir():
        log.error("  Model output directory not found: %s", model_dir)
        return False

    clip_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not clip_dirs:
        log.error("  No clip subdirectories found in %s", model_dir)
        return False

    log.info("  Clip directories in %s/:", model_name)
    total_clips = 0
    for d in sorted(clip_dirs):
        wavs = list(d.glob("*.wav"))
        n = len(wavs)
        total_clips += n
        log.info("    %-40s %6d clips", d.name + "/", n)

        # Spot-check sample rate on first file
        if wavs:
            try:
                info = sf.info(str(wavs[0]))
                sr_note = "" if info.samplerate == 16000 else f"  (⚠ {info.samplerate} Hz — compat patch will resample)"
                log.info("    %-40s SR=%d%s", "", info.samplerate, sr_note)
            except Exception as exc:
                log.warning("    ^ Could not read %s: %s", wavs[0].name, exc)

    if total_clips == 0:
        log.error("  No clips generated")
        ok = False
    else:
        log.info("  Total clips: %d", total_clips)
        if total_clips < expected_positive:
            log.warning(
                "  Fewer clips than expected (%d < %d) — may still work",
                total_clips, expected_positive,
            )

    return ok


# ── 9. augment ───────────────────────────────────────────────────────────

def step_augment() -> bool:
    """Run augmentation (noise, RIR) and mel-spectrogram feature extraction."""
    if not RESOLVED_CONFIG.exists():
        log.error("  Resolved config not found — run 'resolve-config' first")
        return False

    log.info("  Augmenting clips & extracting features …")
    if not _require_local_helper(
        OWW_WRAPPER,
        "Feature augmentation depends on oww_wrapper.py calling the training backend.",
    ):
        return False
    try:
        _run([
            sys.executable, str(OWW_WRAPPER),
            "--training_config", str(RESOLVED_CONFIG),
            "--augment_clips", "--overwrite",
        ])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("  Augmentation failed (exit code %d)", exc.returncode)
        return False


# ── 10. verify-features ──────────────────────────────────────────────────

def step_verify_features() -> bool:
    """Check that .npy feature files were produced with reasonable shapes."""
    import numpy as np

    ok = True
    expected_features = [
        "positive_features_train.npy",
        "positive_features_test.npy",
        "negative_features_train.npy",
        "negative_features_test.npy",
    ]

    for name in expected_features:
        fp = OUTPUT_DIR / name
        if not fp.exists():
            # Also check subdirectories
            found = list(OUTPUT_DIR.rglob(name))
            if found:
                fp = found[0]
            else:
                log.error("  MISSING: %s", name)
                ok = False
                continue

        arr = np.load(str(fp), mmap_mode="r")
        log.info("  OK: %-45s shape=%s  dtype=%s", name, arr.shape, arr.dtype)

        if arr.shape[0] == 0:
            log.error("    ^ empty array!")
            ok = False

    return ok


# ── 11. train ────────────────────────────────────────────────────────────

def step_train() -> bool:
    """Train the DNN model."""
    if not RESOLVED_CONFIG.exists():
        log.error("  Resolved config not found — run 'resolve-config' first")
        return False

    with open(RESOLVED_CONFIG) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    output_dir = Path(cfg["output_dir"])
    model_path = output_dir / f"{model_name}.onnx"

    if model_path.exists():
        log.info("  Model already exists: %s", model_path)
        log.info("  Delete it to retrain.")
        return True

    steps = cfg.get("steps", 50000)
    log.info("  Training %s for %d steps …", model_name, steps)

    if not _require_local_helper(
        OWW_WRAPPER,
        "Model training depends on oww_wrapper.py calling the training backend.",
    ):
        return False
    try:
        _run([
            sys.executable, str(OWW_WRAPPER),
            "--training_config", str(RESOLVED_CONFIG),
            "--train_model",
        ])
        return True
    except subprocess.CalledProcessError as exc:
        log.error("  Training failed (exit code %d)", exc.returncode)
        return False


# ── 12. verify-model ─────────────────────────────────────────────────────

def step_verify_model() -> bool:
    """Verify the ONNX model was produced and can be loaded."""
    cfg = _load_active_config()

    model_name = cfg["model_name"]
    model_path = OUTPUT_DIR / f"{model_name}.onnx"

    # Search for any .onnx file if exact name not found
    if not model_path.exists():
        onnx_files = list(OUTPUT_DIR.rglob("*.onnx"))
        if onnx_files:
            model_path = onnx_files[0]
            log.info("  Found model at: %s (expected %s.onnx)", model_path, model_name)
        else:
            log.error("  No .onnx model found in %s", OUTPUT_DIR)
            return False

    size_mb = model_path.stat().st_size / (1 << 20)
    log.info("  Model: %s  (%.2f MB)", model_path.name, size_mb)

    # Check for companion .data file
    data_file = model_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_mb = data_file.stat().st_size / (1 << 20)
        log.info("  External data: %s  (%.2f MB)", data_file.name, data_mb)

    # Try loading with ONNX runtime
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(model_path))
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        log.info("  ONNX inputs:  %s", [(i.name, i.shape) for i in inputs])
        log.info("  ONNX outputs: %s", [(o.name, o.shape) for o in outputs])

        # Quick inference test with silence
        inp = {inputs[0].name: np.zeros((1, *inputs[0].shape[1:]), dtype=np.float32)}
        result = sess.run(None, inp)
        log.info("  Inference test passed (output shape: %s)", result[0].shape)
    except ImportError:
        log.warning("  onnxruntime not installed — skipping load test")
    except Exception as exc:
        log.warning("  ONNX load test failed: %s", exc)

    return True


# ── 13. export ───────────────────────────────────────────────────────────

def _write_esphome_vad_manifest(cfg: dict, export_dir: Path, model_name: str) -> Path:
    export_manifest = cfg.get('export_manifest', {}) or {}
    manifest = {
        'type': 'micro',
        'wake_word': export_manifest.get('wake_word', 'vad'),
        'author': export_manifest.get('author', 'ha-wakeword-trainer'),
        'website': export_manifest.get('website', 'https://github.com/lgpearson1771/openwakeword-trainer'),
        'model': export_manifest.get('model', f'{model_name}.tflite'),
        'trained_languages': export_manifest.get('trained_languages', ['pl']),
        'version': int(export_manifest.get('version', 2)),
        'micro': {
            'probability_cutoff': float(export_manifest.get('probability_cutoff', 0.5)),
            'tensor_arena_size': int(export_manifest.get('tensor_arena_size', 24000)),
            'feature_step_size': int(export_manifest.get('feature_step_size', 10)),
            'sliding_window_size': int(export_manifest.get('sliding_window_size', 5)),
            'minimum_esphome_version': export_manifest.get('minimum_esphome_version', '2024.7.0'),
        },
    }
    dest = export_dir / f'{model_name}.json'
    dest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return dest


def step_export() -> bool:
    """Copy the trained model to the export/ directory for easy retrieval."""
    cfg = _load_active_config()

    model_name = cfg["model_name"]
    model_path = OUTPUT_DIR / f"{model_name}.onnx"

    if not model_path.exists():
        onnx_files = list(OUTPUT_DIR.rglob("*.onnx"))
        if onnx_files:
            model_path = onnx_files[0]
        else:
            log.error("  No .onnx model found — run 'train' step first")
            return False

    export_dir = SCRIPT_DIR / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    dest = export_dir / f"{model_name}.onnx"
    shutil.copy2(model_path, dest)
    log.info("  Model exported → %s", dest)

    # ONNX models exported with external data have a companion .data file
    data_file = model_path.with_suffix(".onnx.data")
    if data_file.exists():
        dest_data = dest.with_suffix(".onnx.data")
        shutil.copy2(data_file, dest_data)
        log.info("  External data  → %s", dest_data)

    manifest_dest = None
    if _get_mode(cfg) == 'vad' or (cfg.get('export_manifest', {}) or {}).get('esphome_vad'):
        manifest_dest = _write_esphome_vad_manifest(cfg, export_dir, model_name)
        log.info("  ESPHome VAD manifest → %s", manifest_dest)

    log.info("")
    log.info("=" * 60)
    log.info("  DONE!  Your trained model is at:")
    log.info("    %s", dest)
    if data_file.exists():
        log.info("    %s", dest_data)
    if manifest_dest is not None:
        log.info("    %s", manifest_dest)
    log.info("")
    log.info("  To use with openWakeWord:")
    log.info("")
    log.info("    from openwakeword.model import Model")
    log.info('    oww = Model(wakeword_models=["%s"])', dest.name)
    log.info("")
    log.info("  Copy the model file(s) to your project and update your config.")
    log.info("=" * 60)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Download helpers (AudioSet, FMA, MIT RIR, synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _load_audio_mono16k(src: Path) -> "np.ndarray":
    import numpy as np
    import soundfile as sf
    from scipy import signal

    data, sr = sf.read(str(src), always_2d=False)
    if hasattr(data, "ndim") and data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000 and len(data) > 0:
        target_len = max(1, round(len(data) * 16000 / sr))
        data = signal.resample(data, target_len).astype(np.float32)
    return data


def _write_audio_mono16k(dest: Path, data: "np.ndarray") -> None:
    import numpy as np
    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    sf.write(str(dest), clipped, 16000)


def _download_bigos_subset(dest: Path, n: int = 1200) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if _dataset_has_audio(dest):
        log.info("  BIGOS-style speech dataset already present")
        return

    try:
        from datasets import load_dataset
        import soundfile as sf

        count = 0
        for split in ("train", "validation", "test"):
            ds = load_dataset(
                "amu-cai/pl-asr-bigos-v2",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            for row in ds:
                if count >= n:
                    break
                try:
                    audio = row["audio"]
                    sf.write(str(dest / f"bigos_{count:05d}.wav"), audio["array"], audio["sampling_rate"])
                    count += 1
                except Exception:
                    continue
            if count >= n:
                break
        if count > 0:
            log.info("  Saved %d BIGOS clips", count)
            return
        raise RuntimeError("stream yielded no decodable audio")
    except Exception as exc:
        log.warning("  BIGOS download failed: %s", exc)

    try:
        from datasets import load_dataset
        import soundfile as sf

        count = 0
        for split in ("train", "validation", "test"):
            ds = load_dataset(
                "google/fleurs",
                "pl_pl",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            for row in ds:
                if count >= n:
                    break
                try:
                    audio = row["audio"]
                    sf.write(str(dest / f"pl_speech_{count:05d}.wav"), audio["array"], audio["sampling_rate"])
                    count += 1
                except Exception:
                    continue
            if count >= n:
                break
        if count > 0:
            log.info("  BIGOS fallback: saved %d Polish FLEURS clips", count)
            return
        raise RuntimeError("stream yielded no decodable audio")
    except Exception as exc:
        log.warning("  Public Polish speech download failed: %s", exc)

    fallback_files = _ensure_vad_positive_fallback(min_count=min(200, n))
    for idx, src in enumerate(fallback_files[:n]):
        dest_file = dest / f"bigos_fallback_{idx:05d}{src.suffix}"
        if not dest_file.exists():
            shutil.copy2(src, dest_file)
    if _dataset_has_audio(dest):
        log.info("  BIGOS fallback: copied synthetic Polish speech into %s", dest)


def _generate_no_speech_dataset(dest: Path, n: int = 240) -> None:
    import numpy as np

    if _dataset_has_audio(dest):
        log.info("  no_speech dataset already present")
        return

    log.info("  Generating no_speech fallback dataset ...")
    rng = np.random.default_rng(7)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        duration = rng.uniform(3.0, 12.0)
        samples = int(16000 * duration)
        t = np.linspace(0.0, duration, samples, endpoint=False, dtype=np.float32)
        hum = (
            0.006 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, np.pi))
            + 0.003 * np.sin(2 * np.pi * 100 * t + rng.uniform(0, np.pi))
            + 0.002 * np.sin(2 * np.pi * 150 * t + rng.uniform(0, np.pi))
        )
        air = rng.normal(0.0, rng.uniform(0.0005, 0.006), samples).astype(np.float32)
        envelope = np.linspace(rng.uniform(0.3, 0.8), rng.uniform(0.3, 0.8), samples, dtype=np.float32)
        clip = (hum + air) * envelope
        _write_audio_mono16k(dest / f"no_speech_{i:04d}.wav", clip)
    log.info("  Generated %d no_speech clips", n)


def _generate_dinner_party_dataset(dest: Path, n: int = 240) -> None:
    import numpy as np

    if _dataset_has_audio(dest):
        log.info("  dinner_party dataset already present")
        return

    speech_pool = []
    for candidate in (
        DATA_DIR / "bigos",
        DATA_DIR / "mc_speech",
        DATA_DIR / "pl_speech",
        DATA_DIR / "synthetic_pl_speech",
    ):
        if candidate.exists():
            speech_pool.extend(_iter_audio_files(candidate))

    if not speech_pool:
        _download_bigos_subset(DATA_DIR / "bigos", n=400)
        speech_pool.extend(_iter_audio_files(DATA_DIR / "bigos"))

    if not speech_pool:
        speech_pool.extend(_ensure_vad_positive_fallback(min_count=120))

    background_pool = []
    for candidate in (
        DATA_DIR / "audioset_16k",
        DATA_DIR / "fma_small",
        DATA_DIR / "musan",
    ):
        if candidate.exists():
            background_pool.extend(_iter_audio_files(candidate))

    if not background_pool:
        _generate_synthetic_noise(DATA_DIR / "audioset_16k", n=120, label="ambient")
        background_pool.extend(_iter_audio_files(DATA_DIR / "audioset_16k"))

    log.info("  Generating dinner_party fallback dataset ...")
    rng = np.random.default_rng(9)
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        duration = rng.uniform(5.0, 12.0)
        samples = int(16000 * duration)
        mix = np.zeros(samples, dtype=np.float32)

        if background_pool:
            bg = _load_audio_mono16k(Path(rng.choice(background_pool)))
            if len(bg) >= samples:
                start = int(rng.integers(0, max(1, len(bg) - samples + 1)))
                bg = bg[start:start + samples]
            else:
                reps = int(np.ceil(samples / max(1, len(bg))))
                bg = np.tile(bg, reps)[:samples]
            mix += 0.25 * bg.astype(np.float32)

        speaker_count = int(rng.integers(2, 5))
        for _ in range(speaker_count):
            if not speech_pool:
                break
            speech = _load_audio_mono16k(Path(rng.choice(speech_pool)))
            if len(speech) == 0:
                continue
            gain = float(rng.uniform(0.08, 0.22))
            offset = int(rng.integers(0, max(1, samples - 1)))
            available = min(len(speech), samples - offset)
            if available <= 0:
                continue
            mix[offset:offset + available] += gain * speech[:available]

        mix += rng.normal(0.0, 0.003, samples).astype(np.float32)
        peak = max(0.05, float(np.max(np.abs(mix))))
        mix = 0.8 * (mix / peak)
        _write_audio_mono16k(dest / f"dinner_party_{i:04d}.wav", mix)
    log.info("  Generated %d dinner_party clips", n)


def _download_mit_rirs(dest: Path) -> None:
    try:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train", trust_remote_code=True,
        )
        dest.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(ds):
            audio = row["audio"]
            sf.write(str(dest / f"rir_{i:04d}.wav"), audio["array"], audio["sampling_rate"])
        log.info("  Saved %d RIR files", len(ds))
    except Exception as exc:
        log.warning("  Could not download MIT RIRs: %s", exc)
        log.info("  Creating empty RIR directory — training will proceed without RIRs")
        dest.mkdir(parents=True, exist_ok=True)


def _download_audioset_subset(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("agkphysics/AudioSet", "unbalanced", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            if count >= 500:
                break
            try:
                audio = row["audio"]
                sf.write(str(dest / f"audioset_{count:04d}.wav"),
                         audio["array"], audio["sampling_rate"])
                count += 1
            except Exception:
                continue
        log.info("  Saved %d AudioSet clips", count)
    except Exception as exc:
        log.warning("  AudioSet download failed: %s", exc)
        _generate_synthetic_noise(dest, n=200, label="audioset")


def _download_fma_subset(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("rudraml/fma", name="small", split="train",
                          streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            if count >= 200:
                break
            try:
                audio = row["audio"]
                sf.write(str(dest / f"fma_{count:04d}.wav"),
                         audio["array"], audio["sampling_rate"])
                count += 1
            except Exception:
                continue
        log.info("  Saved %d FMA clips", count)
    except Exception as exc:
        log.warning("  FMA download failed: %s", exc)
        _generate_synthetic_noise(dest, n=100, label="fma")


def _download_musan_subset(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("mushanWang/MUSAN", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            if count >= 300:
                break
            try:
                audio = row.get("audio")
                if not audio:
                    continue
                sf.write(str(dest / f"musan_{count:04d}.wav"), audio["array"], audio["sampling_rate"])
                count += 1
            except Exception:
                continue
        if count == 0:
            raise RuntimeError("stream yielded no decodable audio")
        log.info("  Saved %d MUSAN clips", count)
    except Exception as exc:
        log.warning("  MUSAN download failed: %s", exc)
        _generate_synthetic_noise(dest, n=150, label="musan")


def _generate_synthetic_noise(dest: Path, n: int = 200, label: str = "noise") -> None:
    import numpy as np
    import soundfile as sf

    log.info("  Generating %d synthetic noise clips as fallback …", n)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        duration = rng.uniform(3, 10)
        samples = int(16000 * duration)
        white = rng.normal(0, rng.uniform(0.01, 0.15), samples).astype(np.float32)
        sf.write(str(dest / f"{label}_{i:04d}.wav"), white, 16000)
    log.info("  Generated %d noise clips in %s", n, dest)


# ═══════════════════════════════════════════════════════════════════════════
# Step registry & pipeline runner
# ═══════════════════════════════════════════════════════════════════════════

STEPS: list[tuple[str, Callable[[], bool], str]] = [
    ("check-env",        step_check_env,        "Verify Python ≥3.10, CUDA, critical imports"),
    ("apply-patches",    step_apply_patches,     "Apply torchaudio/speechbrain/piper compat patches"),
    ("download",         step_download,          "Download datasets, Piper TTS model, tools"),
    ("verify-data",      step_verify_data,       "Check all data files present & minimum sizes"),
    ("resolve-config",   step_resolve_config,    "Resolve config paths → _resolved_config.yaml"),
    ("generate",         step_generate,          "Generate positive + negative clips via Piper TTS"),
    ("resample-clips",   step_resample_clips,    "Spot-check clip sample rates (resampling is on-the-fly)"),
    ("verify-clips",     step_verify_clips,      "Verify clip counts and sample rates"),
    ("augment",          step_augment,           "Augment clips & extract mel features"),
    ("verify-features",  step_verify_features,   "Check .npy feature files exist & shapes"),
    ("train",            step_train,             "Train DNN model (50k steps, ~30 min on GPU)"),
    ("verify-model",     step_verify_model,      "Verify ONNX model produced & loadable"),
    ("export",           step_export,            "Copy model to export/ directory"),
]

STEP_NAMES = [s[0] for s in STEPS]


def _print_steps() -> None:
    print("\nAvailable steps:\n")
    for i, (name, _, desc) in enumerate(STEPS, 1):
        print(f"  {i:2d}. {name:<20s}  {desc}")
    print()


def run_pipeline(
    *,
    from_step: str | None = None,
    single_step: str | None = None,
    verify_only: bool = False,
) -> bool:
    """Execute steps and stop on first failure."""

    if single_step:
        # Run exactly one step
        matches = [(n, fn, d) for n, fn, d in STEPS if n == single_step]
        if not matches:
            log.error("Unknown step: %s", single_step)
            _print_steps()
            return False
        name, fn, desc = matches[0]
        log.info("=" * 60)
        log.info("STEP: %s  —  %s", name, desc)
        log.info("=" * 60)
        ok = fn()
        status = "PASSED" if ok else "FAILED"
        log.info("Result: %s\n", status)
        return ok

    # Determine which steps to run
    steps_to_run = STEPS
    if from_step:
        try:
            idx = STEP_NAMES.index(from_step)
            steps_to_run = STEPS[idx:]
        except ValueError:
            log.error("Unknown step: %s", from_step)
            _print_steps()
            return False

    if verify_only:
        steps_to_run = [(n, fn, d) for n, fn, d in steps_to_run if n.startswith("verify")]

    total = len(steps_to_run)
    for i, (name, fn, desc) in enumerate(steps_to_run, 1):
        log.info("")
        log.info("=" * 60)
        log.info("[%d/%d]  %s  —  %s", i, total, name, desc)
        log.info("=" * 60)

        ok = fn()

        if ok:
            log.info("[%d/%d]  %s  ✓ PASSED", i, total, name)
        else:
            log.error("[%d/%d]  %s  ✗ FAILED", i, total, name)
            log.error("")
            log.error("Pipeline stopped.  Fix the issue above, then resume:")
            log.error("  python train_wakeword.py --from %s", name)
            return False

    log.info("")
    log.info("=" * 60)
    log.info("  ALL STEPS COMPLETE")
    log.info("=" * 60)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a custom wake word model using openWakeWord.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python train_wakeword.py                                    # full pipeline
              python train_wakeword.py --config configs/wakeword_example.yaml
              python train_wakeword.py --from augment                     # resume
              python train_wakeword.py --step verify-clips                # run one step
              python train_wakeword.py --verify-only                      # check state
              python train_wakeword.py --list-steps                       # show all steps
        """),
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="FILE",
        help="Path to training config YAML (default: configs/wakeword_example.yaml).",
    )
    parser.add_argument(
        "--step", type=str, default=None, metavar="NAME",
        help="Run a single step by name.",
    )
    parser.add_argument(
        "--from", type=str, default=None, dest="from_step", metavar="NAME",
        help="Run from this step onward (skip earlier steps).",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Run only verify-* steps (status check without side effects).",
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="Print all available steps and exit.",
    )
    parser.add_argument(
        "--mode", choices=["wakeword", "vad"], default=None,
        help="Override training mode declared in config.",
    )
    parser.add_argument(
        "--positive-datasets", default=None, metavar="CSV",
        help="Comma-separated positive dataset names for VAD mode (e.g. mc_speech,bigos).",
    )
    parser.add_argument(
        "--negative-datasets", default=None, metavar="CSV",
        help="Comma-separated negative dataset names for VAD mode (e.g. no_speech,dinner_party,musan,fma).",
    )
    parser.add_argument(
        "--dataset-path", action="append", default=[], metavar="NAME=PATH",
        help="Override dataset location. Can be repeated, e.g. --dataset-path mc_speech=/data/mc_speech",
    )
    args = parser.parse_args()

    if args.list_steps:
        _print_steps()
        return

    # Set config file globally
    global CONFIG_FILE
    if args.config:
        CONFIG_FILE = Path(args.config).resolve()
    elif not DEFAULT_CONFIG.exists():
        # Try to find any .yaml in configs/
        configs_dir = SCRIPT_DIR / "configs"
        if configs_dir.exists():
            yamls = sorted(configs_dir.glob("*.yaml"))
            if yamls:
                CONFIG_FILE = yamls[0]
                log.info("Using config: %s", CONFIG_FILE)

    if args.mode or args.positive_datasets or args.negative_datasets or args.dataset_path:
        cfg = _load_config(CONFIG_FILE)
        if args.mode:
            cfg['mode'] = args.mode
        datasets = cfg.setdefault('datasets', {})
        if args.positive_datasets:
            datasets['positive'] = [x.strip() for x in args.positive_datasets.split(',') if x.strip()]
        if args.negative_datasets:
            datasets['negative'] = [x.strip() for x in args.negative_datasets.split(',') if x.strip()]
        dataset_paths = cfg.setdefault('dataset_paths', {})
        for item in args.dataset_path:
            if '=' not in item:
                parser.error(f'Invalid --dataset-path value: {item!r}')
            name, value = item.split('=', 1)
            dataset_paths[name.strip()] = value.strip()
        tmp_path = OUTPUT_DIR / '_cli_config.yaml'
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding='utf-8')
        CONFIG_FILE = tmp_path
        log.info('Using CLI-overridden config: %s', CONFIG_FILE)

    ok = run_pipeline(
        from_step=args.from_step,
        single_step=args.step,
        verify_only=args.verify_only,
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
