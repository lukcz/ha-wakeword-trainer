#!/usr/bin/env python3
"""Autonomous VAD experiment runner based on the current leaderboard."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REPO_ROOT / "configs"
AUTO_CONFIGS_DIR = CONFIGS_DIR / "auto"
RESULTS_DIR = REPO_ROOT / "output" / "_results"
LOGS_DIR = REPO_ROOT / "output" / "_logs"
STATE_PATH = RESULTS_DIR / "_auto_vad_research_state.json"
LEADERBOARD_PATH = RESULTS_DIR / "_leaderboard.json"
RUNNER_STATUS_PATH = RESULTS_DIR / "_auto_vad_research_status.json"
_MIRROR_ROOT_UNSET = object()
_MIRROR_ROOT_CACHE: Path | None | object = _MIRROR_ROOT_UNSET

STATIC_PRIORITY = [
    "polish_vad_public_research_gainwide_earlydense",
    "polish_vad_public_research_gainwide_safetune",
    "polish_vad_public_research_homeweighted",
    "polish_vad_public_research_gainwide",
    "polish_vad_public_research_normlite",
    "polish_vad_public_research_homeweighted_gainwide",
    "polish_vad_public_research_homeweighted_balancedbg",
    "polish_vad_public_research_speechnoise_heavy",
    "polish_vad_public_research_musan_down",
    "polish_vad_public_research_cutoffsafe",
    "polish_vad_public_research_earlydense",
    "polish_vad_public_research_bigos_plus_homeweighted",
]


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _mirror_file(path)


def _mirror_repo_root() -> Path | None:
    global _MIRROR_ROOT_CACHE

    if _MIRROR_ROOT_CACHE is not _MIRROR_ROOT_UNSET:
        if isinstance(_MIRROR_ROOT_CACHE, Path):
            return _MIRROR_ROOT_CACHE
        return None

    candidates: list[Path] = []
    env_value = os.environ.get("HAWW_WINDOWS_MIRROR_ROOT")
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(Path("/mnt/d/Github/ha-wakeword-trainer"))

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved == REPO_ROOT:
            continue
        if candidate.exists() and (candidate / ".git").exists():
            _MIRROR_ROOT_CACHE = candidate
            return candidate

    _MIRROR_ROOT_CACHE = None
    return None


def _mirror_path(local_path: Path) -> Path | None:
    mirror_root = _mirror_repo_root()
    if mirror_root is None:
        return None
    try:
        relative_path = local_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    return mirror_root / relative_path


def _mirror_file(local_path: Path) -> None:
    if not local_path.exists():
        return
    dest_path = _mirror_path(local_path)
    if dest_path is None:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(local_path.read_bytes())


def _load_state() -> dict:
    return _load_json(STATE_PATH, {"launched": [], "generated": []})


def _save_state(state: dict) -> None:
    _write_json(STATE_PATH, state)


def _write_runner_status(**fields) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **fields,
    }
    _write_json(RUNNER_STATUS_PATH, payload)


def _load_leaderboard() -> list[dict]:
    return _load_json(LEADERBOARD_PATH, [])


def _result_record_for_preset(stem: str) -> dict | None:
    path = RESULTS_DIR / f"{stem}.json"
    if not path.exists():
        return None
    return _load_json(path, None)


def _preset_has_finished(stem: str) -> bool:
    record = _result_record_for_preset(stem)
    return bool(record and record.get("status") in {"success", "failed"})


def _find_config_path(stem: str) -> Path | None:
    for root in (CONFIGS_DIR, AUTO_CONFIGS_DIR):
        candidate = root / f"{stem}.yaml"
        if candidate.exists():
            return candidate
    return None


def _running_training_processes() -> list[str]:
    lines: list[str] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
        if not argv:
            continue
        joined = " ".join(argv)
        if "auto_vad_research.py" in joined:
            continue
        if "train_microwakeword.py" not in joined:
            continue
        lines.append(f"{entry.name} {joined}")
    return lines


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _model_name_from_stem(stem: str) -> str:
    suffix = stem.removeprefix("polish_vad_")
    return f"vad_pl_{suffix}"


def _prepare_generated_config(base_cfg: dict, stem: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["model_name"] = _model_name_from_stem(stem)
    training = cfg.setdefault("training", {})
    training["train_dir"] = f"output/{cfg['model_name']}/trained_model"
    return cfg


def _default_early_stopping() -> dict:
    return {
        "enabled": True,
        "metric": "average_viable_recall",
        "min_training_steps": 3000,
        "patience_evals": 5,
        "min_delta_absolute": 0.004,
        "min_delta_relative": 0.005,
    }


def _ensure_early_stopping(cfg: dict, *, patience_evals: int = 5) -> None:
    training = cfg.setdefault("training", {})
    early = copy.deepcopy(_default_early_stopping())
    early["patience_evals"] = patience_evals
    training["early_stopping"] = early


def _mutate_gainwide_wider(cfg: dict) -> None:
    augmentation = cfg.setdefault("augmentation", {})
    augmentation["gain_min_db"] = -22.0
    augmentation["gain_max_db"] = 8.0
    augmentation["gain_transition_min_db"] = -14.0
    augmentation["gain_transition_max_db"] = 14.0
    probabilities = augmentation.setdefault("probabilities", {})
    probabilities["GainTransition"] = 0.4
    _ensure_early_stopping(cfg, patience_evals=5)


def _mutate_gainwide_safetune(cfg: dict) -> None:
    weights = cfg.setdefault("negative_feature_weights", {})
    weights["speech"] = 10.5
    weights["dinner_party"] = 16.0
    weights["no_speech"] = 10.5
    cfg["positive_sampling_weight"] = 1.7
    augmentation = cfg.setdefault("augmentation", {})
    probabilities = augmentation.setdefault("probabilities", {})
    probabilities["AddBackgroundNoise"] = 0.8
    probabilities["RIR"] = 0.5
    training = cfg.setdefault("training", {})
    training["negative_class_weight"] = [21]
    _ensure_early_stopping(cfg, patience_evals=5)


def _mutate_earlydense(cfg: dict) -> None:
    training = cfg.setdefault("training", {})
    training["eval_step_interval"] = 250
    _ensure_early_stopping(cfg, patience_evals=6)


def _candidate_specs_for_leader(leader_preset: str) -> list[tuple[str, str, list[str]]]:
    if "gainwide" in leader_preset:
        return [
            (
                "polish_vad_auto_gainwide_safetune_earlydense",
                "Combine the strongest current family with denser evals and slightly safer regularization.",
                ["safetune", "earlydense"],
            ),
            (
                "polish_vad_auto_gainwide_wider",
                "Widen gain augmentation around the current leader to see if extra amplitude diversity helps.",
                ["wider"],
            ),
            (
                "polish_vad_auto_gainwide_wider_safetune",
                "Test the wider gain range with a slightly safer negative mix.",
                ["wider", "safetune"],
            ),
        ]
    return [
        (
            "polish_vad_auto_gainwide_wider",
            "Shift search toward the currently strongest gain-based family.",
            ["wider"],
        ),
        (
            "polish_vad_auto_gainwide_safetune_earlydense",
            "Search a safer, denser-eval variant around gainwide.",
            ["safetune", "earlydense"],
        ),
    ]


def _generate_dynamic_preset(state: dict) -> tuple[Path | None, str]:
    leaderboard = _load_leaderboard()
    if not leaderboard:
        return None, "Leaderboard is empty"

    leader = leaderboard[0]
    leader_preset = str(leader.get("preset", ""))
    leader_config = leader.get("config_file")
    if not leader_config:
        return None, "Leader has no config path"

    base_path = Path(leader_config)
    if not base_path.exists():
        return None, f"Leader config missing: {base_path}"

    generated = set(state.get("generated", []))
    for stem, reason, mutations in _candidate_specs_for_leader(leader_preset):
        if stem in generated or _preset_has_finished(stem) or _find_config_path(stem):
            continue
        cfg = _prepare_generated_config(_load_yaml(base_path), stem)
        for mutation in mutations:
            if mutation == "wider":
                _mutate_gainwide_wider(cfg)
            elif mutation == "safetune":
                _mutate_gainwide_safetune(cfg)
            elif mutation == "earlydense":
                _mutate_earlydense(cfg)
        dest = AUTO_CONFIGS_DIR / f"{stem}.yaml"
        _write_yaml(dest, cfg)
        state.setdefault("generated", []).append(stem)
        _save_state(state)
        return dest, reason
    return None, "No dynamic candidates remain"


def _next_static_candidate() -> tuple[Path | None, str]:
    for stem in STATIC_PRIORITY:
        path = _find_config_path(stem)
        if path is None:
            continue
        if not _preset_has_finished(stem):
            return path, "Queued static research preset"
    return None, "No static candidates remain"


def _next_candidate(state: dict) -> tuple[Path | None, str]:
    path, reason = _next_static_candidate()
    if path is not None:
        return path, reason
    return _generate_dynamic_preset(state)


def _launch_training(config_path: Path, *, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        "train_microwakeword.py",
        "--config",
        str(config_path),
        "--from",
        "download-assets",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous VAD preset tuner.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-launches", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    AUTO_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_state()

    while True:
        running = _running_training_processes()
        if running:
            _write_runner_status(
                active=True,
                phase="waiting_for_training",
                running_training_processes=running,
                launched_count=len(state.get("launched", [])),
            )
            print(f"[{time.strftime('%F %T')}] Training already running; waiting {args.poll_seconds}s", flush=True)
            time.sleep(args.poll_seconds)
            continue

        launched = state.get("launched", [])
        if len(launched) >= args.max_launches:
            _write_runner_status(
                active=False,
                phase="idle",
                reason=f"Reached max launches ({args.max_launches})",
                launched_count=len(launched),
            )
            print(f"[{time.strftime('%F %T')}] Reached max launches ({args.max_launches}); exiting", flush=True)
            return 0

        config_path, reason = _next_candidate(state)
        if config_path is None:
            _write_runner_status(
                active=False,
                phase="idle",
                reason=reason,
                launched_count=len(launched),
            )
            print(f"[{time.strftime('%F %T')}] {reason}; exiting", flush=True)
            return 0

        stem = config_path.stem
        _write_runner_status(
            active=True,
            phase="launching",
            next_preset=stem,
            reason=reason,
            launched_count=len(launched),
        )
        print(f"[{time.strftime('%F %T')}] Next preset: {stem} ({reason})", flush=True)
        state.setdefault("launched", []).append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "preset": stem,
                "config_path": str(config_path),
                "reason": reason,
            }
        )
        _save_state(state)

        returncode = _launch_training(config_path, dry_run=args.dry_run)
        _write_runner_status(
            active=True,
            phase="post_run",
            last_preset=stem,
            last_returncode=returncode,
            launched_count=len(state.get("launched", [])),
        )
        print(f"[{time.strftime('%F %T')}] Finished {stem} with rc={returncode}", flush=True)

        if args.dry_run:
            return 0

        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
