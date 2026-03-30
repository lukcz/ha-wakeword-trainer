#!/usr/bin/env python3
"""Preset-selection helpers for autonomous VAD research."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = REPO_ROOT / "configs"
AUTO_CONFIGS_DIR = CONFIGS_DIR / "auto"
RESULTS_DIR = REPO_ROOT / "output" / "_results"
LOGS_DIR = REPO_ROOT / "output" / "_logs"
LEADERBOARD_PATH = RESULTS_DIR / "_leaderboard.json"

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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def load_leaderboard() -> list[dict]:
    return _load_json(LEADERBOARD_PATH, [])


def result_record_for_preset(stem: str) -> dict | None:
    path = RESULTS_DIR / f"{stem}.json"
    if not path.exists():
        return None
    return _load_json(path, None)


def preset_has_finished(stem: str) -> bool:
    record = result_record_for_preset(stem)
    return bool(record and record.get("status") in {"success", "failed"})


def find_config_path(stem: str) -> Path | None:
    for root in (CONFIGS_DIR, AUTO_CONFIGS_DIR):
        candidate = root / f"{stem}.yaml"
        if candidate.exists():
            return candidate
    return None


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


def _is_blocked(stem: str, blocked_presets: set[str]) -> bool:
    return stem in blocked_presets


def next_static_candidate(blocked_presets: set[str] | None = None) -> tuple[Path | None, str, str]:
    blocked = blocked_presets or set()
    for stem in STATIC_PRIORITY:
        path = find_config_path(stem)
        if path is None:
            continue
        if _is_blocked(stem, blocked) or preset_has_finished(stem):
            continue
        return path, "Queued static research preset", "static"
    return None, "No static candidates remain", "static"


def generate_dynamic_candidate(
    generated_presets: set[str] | None = None,
    blocked_presets: set[str] | None = None,
) -> tuple[Path | None, str, str]:
    leaderboard = load_leaderboard()
    if not leaderboard:
        return None, "Leaderboard is empty", "dynamic"

    leader = leaderboard[0]
    leader_preset = str(leader.get("preset", ""))
    leader_config = leader.get("config_file")
    if not leader_config:
        return None, "Leader has no config path", "dynamic"

    base_path = Path(leader_config)
    if not base_path.exists():
        return None, f"Leader config missing: {base_path}", "dynamic"

    generated = generated_presets or set()
    blocked = blocked_presets or set()
    for stem, reason, mutations in _candidate_specs_for_leader(leader_preset):
        if stem in generated or _is_blocked(stem, blocked) or preset_has_finished(stem) or find_config_path(stem):
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
        return dest, reason, "dynamic"

    return None, "No dynamic candidates remain", "dynamic"


def next_candidate(
    generated_presets: set[str] | None = None,
    blocked_presets: set[str] | None = None,
) -> tuple[Path | None, str, str]:
    path, reason, source = next_static_candidate(blocked_presets)
    if path is not None:
        return path, reason, source
    return generate_dynamic_candidate(generated_presets, blocked_presets)
