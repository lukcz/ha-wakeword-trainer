#!/usr/bin/env python3
"""Reliable single-process orchestrator for autonomous VAD research."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from vad_research_planner import LOGS_DIR, REPO_ROOT, RESULTS_DIR, next_candidate

ORCHESTRATOR_STATUS_PATH = RESULTS_DIR / "_orchestrator_status.json"
CURRENT_RUN_PATH = RESULTS_DIR / "_current_run.json"
QUEUE_PATH = RESULTS_DIR / "_queue.json"
RUNS_HISTORY_PATH = RESULTS_DIR / "_runs.jsonl"
COMPAT_RUNNER_STATUS_PATH = RESULTS_DIR / "_auto_vad_research_status.json"
COMPAT_RUNNER_STATE_PATH = RESULTS_DIR / "_auto_vad_research_state.json"
COMPAT_TRAINING_STATUS_PATH = RESULTS_DIR / "_training_status.json"
ORCHESTRATOR_LOG_PATH = LOGS_DIR / "vad-orchestrator.log"

_MIRROR_ROOT_UNSET = object()
_MIRROR_ROOT_CACHE: Path | None | object = _MIRROR_ROOT_UNSET

_VALIDATION_RE = re.compile(
    r"Step (?P<step>\d+) \(nonstreaming\): Validation: "
    r"recall at no faph = (?P<recall_at_no_faph>[-0-9.]+) "
    r"with cutoff (?P<cutoff>[-0-9.]+), "
    r"accuracy = (?P<accuracy>[-0-9.]+)%, "
    r"recall = (?P<recall>[-0-9.]+)%, "
    r"precision = (?P<precision>[-0-9.]+)%, "
    r"ambient false positives = (?P<ambient_false_positives>\d+), "
    r"estimated false positives per hour = (?P<estimated_false_positives_per_hour>[-0-9.]+), "
    r"loss = (?P<loss>[-0-9.]+), "
    r"auc = (?P<auc>[-0-9.]+), "
    r"average viable recall = (?P<average_viable_recall>[-0-9.]+)"
)
_TRAIN_STEP_RE = re.compile(
    r"Step #(?P<step>\d+): rate (?P<rate>[-0-9.]+), accuracy (?P<accuracy>[-0-9.]+)%, "
    r"recall (?P<recall>[-0-9.]+)%, precision (?P<precision>[-0-9.]+)%, cross entropy (?P<cross_entropy>[-0-9.]+)"
)

log = logging.getLogger("vad_orchestrator")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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
    shutil.copy2(local_path, dest_path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")
    _mirror_file(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _mirror_file(path)


def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ORCHESTRATOR_LOG_PATH, encoding="utf-8"),
    ]
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def _default_queue() -> dict:
    return {"updated_at": _timestamp(), "entries": []}


def _default_current_run() -> dict:
    return {
        "timestamp": _timestamp(),
        "active": False,
        "status": "idle",
        "run_id": None,
        "preset": None,
        "config_path": None,
        "pid": None,
        "started_at": None,
        "started_at_epoch": None,
        "session_log": None,
        "result_file": None,
        "reason": None,
        "last_metric_snapshot": None,
        "last_training_step": None,
        "last_error": None,
    }


def _load_queue() -> dict:
    payload = _load_json(QUEUE_PATH, _default_queue())
    if not isinstance(payload, dict):
        return _default_queue()
    payload.setdefault("updated_at", _timestamp())
    payload.setdefault("entries", [])
    return payload


def _save_queue(queue: dict) -> None:
    queue["updated_at"] = _timestamp()
    _write_json(QUEUE_PATH, queue)


def _load_current_run() -> dict:
    payload = _load_json(CURRENT_RUN_PATH, _default_current_run())
    if not isinstance(payload, dict):
        return _default_current_run()
    return {**_default_current_run(), **payload}


def _save_current_run(current_run: dict) -> None:
    current_run["timestamp"] = _timestamp()
    _write_json(CURRENT_RUN_PATH, current_run)


def _queue_counts(queue: dict) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "finished": 0, "failed": 0, "abandoned": 0}
    for entry in queue.get("entries", []):
        status = str(entry.get("status", "pending"))
        if status in counts:
            counts[status] += 1
    return counts


def _write_compatibility_files(orchestrator_status: dict, current_run: dict, queue: dict) -> None:
    queue_entries = queue.get("entries", [])
    pending_entries = [entry for entry in queue_entries if entry.get("status") == "pending"]
    launched_entries = [entry for entry in queue_entries if entry.get("run_id")]
    generated = sorted(entry["preset"] for entry in queue_entries if entry.get("source") == "dynamic")

    runner_status = {
        "timestamp": orchestrator_status.get("heartbeat_at"),
        "active": bool(orchestrator_status.get("active")),
        "phase": orchestrator_status.get("phase"),
        "current_preset": orchestrator_status.get("current_preset"),
        "next_preset": pending_entries[0]["preset"] if pending_entries else None,
        "reason": orchestrator_status.get("reason"),
        "queue_counts": orchestrator_status.get("queue_counts"),
    }
    _write_json(COMPAT_RUNNER_STATUS_PATH, runner_status)

    runner_state = {
        "generated": generated,
        "launched": [
            {
                "timestamp": entry.get("last_started_at") or entry.get("created_at"),
                "preset": entry.get("preset"),
                "config_path": entry.get("config_path"),
                "reason": entry.get("reason"),
                "status": entry.get("status"),
                "run_id": entry.get("run_id"),
            }
            for entry in launched_entries
        ],
    }
    _write_json(COMPAT_RUNNER_STATE_PATH, runner_state)

    metric_snapshot = current_run.get("last_metric_snapshot") or {}
    training_step = current_run.get("last_training_step") or {}
    training_status = {
        "timestamp": orchestrator_status.get("heartbeat_at"),
        "active": bool(current_run.get("active")),
        "pipeline_status": current_run.get("status", "idle"),
        "preset": current_run.get("preset"),
        "config_file": current_run.get("config_path"),
        "pid": current_run.get("pid"),
        "current_step": metric_snapshot.get("step") or training_step.get("step"),
        "current_step_index": None,
        "total_steps": None,
        "session_log": current_run.get("session_log"),
        "result_file": current_run.get("result_file"),
        "error": current_run.get("last_error"),
    }
    _write_json(COMPAT_TRAINING_STATUS_PATH, training_status)


def _write_orchestrator_status(queue: dict, current_run: dict, **fields) -> None:
    status = {
        "timestamp": _timestamp(),
        "active": True,
        "pid": os.getpid(),
        "heartbeat_at": _timestamp(),
        "phase": fields.pop("phase", "idle"),
        "current_run_id": current_run.get("run_id"),
        "current_preset": current_run.get("preset"),
        "queue_counts": _queue_counts(queue),
        **fields,
    }
    _write_json(ORCHESTRATOR_STATUS_PATH, status)
    _write_compatibility_files(status, current_run, queue)
    _mirror_file(ORCHESTRATOR_LOG_PATH)


def _result_path_for_preset(preset: str | None) -> Path | None:
    if not preset:
        return None
    return RESULTS_DIR / f"{preset}.json"


def _queue_entry(queue: dict, preset: str) -> dict | None:
    for entry in queue.get("entries", []):
        if entry.get("preset") == preset:
            return entry
    return None


def _pending_entry(queue: dict) -> dict | None:
    for entry in queue.get("entries", []):
        if entry.get("status") == "pending":
            return entry
    return None


def _ensure_queue_entry(queue: dict, config_path: Path, *, reason: str, source: str) -> dict:
    preset = config_path.stem
    entry = _queue_entry(queue, preset)
    if entry is not None:
        return entry
    entry = {
        "preset": preset,
        "config_path": str(config_path),
        "reason": reason,
        "source": source,
        "status": "pending",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "run_id": None,
        "last_started_at": None,
        "last_finished_at": None,
        "result_file": None,
        "best_validation": None,
        "last_error": None,
    }
    queue.setdefault("entries", []).append(entry)
    return entry


def _generated_presets(queue: dict) -> set[str]:
    return {
        str(entry.get("preset"))
        for entry in queue.get("entries", [])
        if entry.get("source") == "dynamic"
    }


def _blocked_presets(queue: dict) -> set[str]:
    return {
        str(entry.get("preset"))
        for entry in queue.get("entries", [])
        if entry.get("status") in {"pending", "running"}
    }


def _list_training_processes() -> list[dict]:
    processes: list[dict] = []
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
        if not argv or "train_microwakeword.py" not in " ".join(argv):
            continue

        config_path = None
        for index, arg in enumerate(argv[:-1]):
            if arg == "--config":
                config_path = argv[index + 1]
                break
        preset = Path(config_path).stem if config_path else None
        processes.append(
            {
                "pid": int(entry.name),
                "argv": argv,
                "config_path": config_path,
                "preset": preset,
            }
        )
    return sorted(processes, key=lambda item: item["pid"])


def _find_session_log(preset: str | None, started_at_epoch: float | None, existing_path: str | None) -> str | None:
    if existing_path and Path(existing_path).exists():
        return existing_path
    if not preset:
        return existing_path

    matches = sorted(
        LOGS_DIR.glob(f"{preset}-*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    lower_bound = (started_at_epoch or 0.0) - 30.0
    for path in matches:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= lower_bound:
            return str(path)
    return existing_path


def _tail_lines(path: Path, max_lines: int = 400) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def _latest_metrics_from_log(session_log: str | None) -> tuple[dict | None, dict | None]:
    if not session_log:
        return None, None
    lines = _tail_lines(Path(session_log))

    last_validation = None
    last_training_step = None
    for line in lines:
        validation_match = _VALIDATION_RE.search(line)
        if validation_match:
            values = validation_match.groupdict()
            last_validation = {
                "step": int(values["step"]),
                "recall_at_no_faph": float(values["recall_at_no_faph"]),
                "cutoff": float(values["cutoff"]),
                "accuracy": float(values["accuracy"]),
                "recall": float(values["recall"]),
                "precision": float(values["precision"]),
                "ambient_false_positives": int(values["ambient_false_positives"]),
                "estimated_false_positives_per_hour": float(values["estimated_false_positives_per_hour"]),
                "loss": float(values["loss"]),
                "auc": float(values["auc"]),
                "average_viable_recall": float(values["average_viable_recall"]),
            }
        step_match = _TRAIN_STEP_RE.search(line)
        if step_match:
            values = step_match.groupdict()
            last_training_step = {
                "step": int(values["step"]),
                "rate": float(values["rate"]),
                "accuracy": float(values["accuracy"]),
                "recall": float(values["recall"]),
                "precision": float(values["precision"]),
                "cross_entropy": float(values["cross_entropy"]),
            }
    return last_validation, last_training_step


def _load_result_record(result_path: Path | None) -> dict | None:
    if result_path is None or not result_path.exists():
        return None
    payload = _load_json(result_path, None)
    return payload if isinstance(payload, dict) else None


def _append_run_record_if_needed(queue_entry: dict, record: dict) -> None:
    if queue_entry.get("history_recorded_at"):
        return
    _append_jsonl(RUNS_HISTORY_PATH, record)
    queue_entry["history_recorded_at"] = _timestamp()


def _launch_training(queue_entry: dict) -> dict:
    cmd = [
        sys.executable,
        "train_microwakeword.py",
        "--config",
        str(queue_entry["config_path"]),
        "--from",
        "download-assets",
    ]
    log.info("$ %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    now = time.time()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{queue_entry['preset']}"
    queue_entry["status"] = "running"
    queue_entry["run_id"] = run_id
    queue_entry["last_started_at"] = _timestamp()
    queue_entry["updated_at"] = _timestamp()
    queue_entry["last_error"] = None
    return {
        "timestamp": _timestamp(),
        "active": True,
        "status": "running",
        "run_id": run_id,
        "preset": queue_entry["preset"],
        "config_path": queue_entry["config_path"],
        "pid": process.pid,
        "started_at": _timestamp(),
        "started_at_epoch": now,
        "session_log": None,
        "result_file": str(_result_path_for_preset(queue_entry["preset"])),
        "reason": queue_entry.get("reason"),
        "last_metric_snapshot": None,
        "last_training_step": None,
        "last_error": None,
    }


def _adopt_running_process(queue: dict, current_run: dict, process: dict) -> dict:
    preset = process.get("preset")
    config_path = process.get("config_path")
    if not preset or not config_path:
        adopted = {
            **_default_current_run(),
            "active": True,
            "status": "running",
            "run_id": f"adopted-{process['pid']}",
            "preset": preset,
            "config_path": config_path,
            "pid": process["pid"],
            "started_at": current_run.get("started_at") or _timestamp(),
            "started_at_epoch": current_run.get("started_at_epoch") or time.time(),
            "session_log": current_run.get("session_log"),
            "result_file": str(_result_path_for_preset(preset)),
            "reason": current_run.get("reason") or "Adopted pre-existing training process",
        }
        return adopted

    queue_entry = _ensure_queue_entry(
        queue,
        Path(config_path),
        reason=current_run.get("reason") or "Adopted pre-existing training process",
        source="adopted",
    )
    if not current_run.get("active") or current_run.get("pid") != process["pid"]:
        run_id = current_run.get("run_id") if current_run.get("pid") == process["pid"] else f"adopted-{process['pid']}"
        current_run = {
            **_default_current_run(),
            "active": True,
            "status": "running",
            "run_id": run_id,
            "preset": preset,
            "config_path": config_path,
            "pid": process["pid"],
            "started_at": current_run.get("started_at") or _timestamp(),
            "started_at_epoch": current_run.get("started_at_epoch") or time.time(),
            "session_log": current_run.get("session_log"),
            "result_file": str(_result_path_for_preset(preset)),
            "reason": queue_entry.get("reason") or "Adopted pre-existing training process",
            "last_metric_snapshot": current_run.get("last_metric_snapshot"),
            "last_training_step": current_run.get("last_training_step"),
            "last_error": None,
        }
    queue_entry["status"] = "running"
    queue_entry["run_id"] = current_run["run_id"]
    queue_entry["last_started_at"] = current_run["started_at"]
    queue_entry["updated_at"] = _timestamp()
    return current_run


def _refresh_current_run(queue: dict, current_run: dict, process: dict | None) -> dict:
    if process is not None:
        current_run = _adopt_running_process(queue, current_run, process)
    current_run["session_log"] = _find_session_log(
        current_run.get("preset"),
        current_run.get("started_at_epoch"),
        current_run.get("session_log"),
    )
    validation, training_step = _latest_metrics_from_log(current_run.get("session_log"))
    if validation is not None:
        current_run["last_metric_snapshot"] = validation
    if training_step is not None:
        current_run["last_training_step"] = training_step

    queue_entry = _queue_entry(queue, str(current_run.get("preset")))
    if queue_entry is not None:
        queue_entry["status"] = "running"
        queue_entry["run_id"] = current_run.get("run_id")
        queue_entry["updated_at"] = _timestamp()
        queue_entry["session_log"] = current_run.get("session_log")
        queue_entry["last_metric_snapshot"] = current_run.get("last_metric_snapshot")
    return current_run


def _finalize_current_run(queue: dict, current_run: dict) -> dict:
    preset = current_run.get("preset")
    result_path = _result_path_for_preset(preset)
    result_record = _load_result_record(result_path)
    queue_entry = _queue_entry(queue, str(preset)) if preset else None

    final_status = "failed"
    error = current_run.get("last_error")
    if result_record is not None and result_record.get("status") == "success":
        final_status = "finished"
        error = None
    elif result_record is not None and result_record.get("status") == "failed":
        error = result_record.get("error") or error
    elif error is None:
        error = "Training process exited without a result summary."

    ended_at = _timestamp()
    run_record = {
        "timestamp": ended_at,
        "run_id": current_run.get("run_id"),
        "preset": preset,
        "config_path": current_run.get("config_path"),
        "status": final_status,
        "started_at": current_run.get("started_at"),
        "ended_at": ended_at,
        "pid": current_run.get("pid"),
        "session_log": current_run.get("session_log"),
        "result_file": str(result_path) if result_path else None,
        "reason": current_run.get("reason"),
        "last_metric_snapshot": current_run.get("last_metric_snapshot"),
        "best_validation": result_record.get("best_validation") if result_record else None,
        "error": error,
    }

    if queue_entry is not None:
        queue_entry["status"] = final_status
        queue_entry["last_finished_at"] = ended_at
        queue_entry["updated_at"] = ended_at
        queue_entry["result_file"] = run_record["result_file"]
        queue_entry["best_validation"] = run_record["best_validation"]
        queue_entry["last_error"] = error
        _append_run_record_if_needed(queue_entry, run_record)
    else:
        _append_jsonl(RUNS_HISTORY_PATH, run_record)

    return {
        **_default_current_run(),
        "status": final_status,
        "preset": preset,
        "config_path": current_run.get("config_path"),
        "pid": current_run.get("pid"),
        "session_log": current_run.get("session_log"),
        "result_file": run_record["result_file"],
        "last_metric_snapshot": run_record["best_validation"] or current_run.get("last_metric_snapshot"),
        "last_error": error,
    }


def _maybe_queue_next_candidate(queue: dict) -> dict | None:
    pending = _pending_entry(queue)
    if pending is not None:
        return pending

    config_path, reason, source = next_candidate(
        generated_presets=_generated_presets(queue),
        blocked_presets=_blocked_presets(queue),
    )
    if config_path is None:
        return None
    return _ensure_queue_entry(queue, config_path, reason=reason, source=source)


def _mark_orphaned_runs(queue: dict, current_run: dict) -> None:
    if current_run.get("active"):
        return
    for entry in queue.get("entries", []):
        if entry.get("status") == "running":
            entry["status"] = "abandoned"
            entry["updated_at"] = _timestamp()
            if not entry.get("last_error"):
                entry["last_error"] = "Run lost without an active training process."


def _once_complete(args: argparse.Namespace) -> bool:
    return bool(args.once)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reliable autonomous VAD orchestrator.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Perform a single orchestration iteration.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the next action without launching training.")
    parser.add_argument("--max-launches", type=int, default=None, help="Deprecated compatibility flag.")
    args = parser.parse_args(argv)

    _setup_logging()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _mirror_file(ORCHESTRATOR_LOG_PATH)

    log.info("Orchestrator starting in %s", REPO_ROOT)
    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        log.warning("Received signal %s; stopping after current iteration", signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not stop_requested:
        queue = _load_queue()
        current_run = _load_current_run()
        processes = _list_training_processes()

        if len(processes) > 1:
            _write_orchestrator_status(
                queue,
                current_run,
                phase="blocked_multiple_training_processes",
                active=True,
                reason="Multiple train_microwakeword.py processes detected",
                running_training_processes=processes,
            )
            log.warning("Multiple training processes detected; waiting for manual intervention")
            _save_queue(queue)
            _save_current_run(current_run)
            if _once_complete(args):
                return 0
            time.sleep(args.poll_seconds)
            continue

        if processes:
            current_run = _refresh_current_run(queue, current_run, processes[0])
            _save_queue(queue)
            _save_current_run(current_run)
            _write_orchestrator_status(
                queue,
                current_run,
                phase="running",
                active=True,
                reason=current_run.get("reason"),
            )
            if _once_complete(args):
                return 0
            time.sleep(args.poll_seconds)
            continue

        if current_run.get("active"):
            log.info("Finalizing completed run %s", current_run.get("run_id"))
            current_run = _finalize_current_run(queue, current_run)
            _save_queue(queue)
            _save_current_run(current_run)
            _write_orchestrator_status(
                queue,
                current_run,
                phase="idle",
                active=True,
                reason="Run finalized",
            )
            if _once_complete(args):
                return 0
            time.sleep(2)
            continue

        _mark_orphaned_runs(queue, current_run)
        queue_entry = _maybe_queue_next_candidate(queue)
        if queue_entry is None:
            _save_queue(queue)
            _save_current_run(current_run)
            _write_orchestrator_status(
                queue,
                current_run,
                phase="idle",
                active=True,
                reason="No pending or new candidates remain",
            )
            if _once_complete(args):
                return 0
            time.sleep(args.poll_seconds)
            continue

        try:
            if args.dry_run:
                _save_queue(queue)
                _save_current_run(current_run)
                _write_orchestrator_status(
                    queue,
                    current_run,
                    phase="planned",
                    active=True,
                    reason=f"Next preset would be {queue_entry['preset']}",
                )
                log.info("Dry run: next preset would be %s", queue_entry["preset"])
                return 0
            current_run = _launch_training(queue_entry)
        except OSError as exc:
            queue_entry["status"] = "failed"
            queue_entry["updated_at"] = _timestamp()
            queue_entry["last_error"] = str(exc)
            _append_run_record_if_needed(
                queue_entry,
                {
                    "timestamp": _timestamp(),
                    "run_id": queue_entry.get("run_id"),
                    "preset": queue_entry.get("preset"),
                    "config_path": queue_entry.get("config_path"),
                    "status": "failed",
                    "started_at": None,
                    "ended_at": _timestamp(),
                    "pid": None,
                    "session_log": None,
                    "result_file": None,
                    "reason": queue_entry.get("reason"),
                    "last_metric_snapshot": None,
                    "best_validation": None,
                    "error": str(exc),
                },
            )
            _save_queue(queue)
            _save_current_run(_default_current_run())
            _write_orchestrator_status(
                queue,
                _default_current_run(),
                phase="launch_failed",
                active=True,
                reason=str(exc),
            )
            if _once_complete(args):
                return 1
            time.sleep(args.poll_seconds)
            continue

        _save_queue(queue)
        _save_current_run(current_run)
        _write_orchestrator_status(
            queue,
            current_run,
            phase="running",
            active=True,
            reason=queue_entry.get("reason"),
        )
        log.info("Launched %s as run %s (pid=%s)", current_run["preset"], current_run["run_id"], current_run["pid"])

        if _once_complete(args):
            return 0

        time.sleep(min(args.poll_seconds, 10))

    queue = _load_queue()
    current_run = _load_current_run()
    _write_orchestrator_status(
        queue,
        current_run,
        phase="stopped",
        active=False,
        reason="Orchestrator stopped by signal",
    )
    log.info("Orchestrator stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
