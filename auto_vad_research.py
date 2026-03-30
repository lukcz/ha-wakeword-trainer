#!/usr/bin/env python3
"""Compatibility wrapper for the new VAD orchestrator."""

from __future__ import annotations

import argparse
import sys

from vad_orchestrator import main as orchestrator_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deprecated entrypoint kept for compatibility; delegates to vad_orchestrator.py.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-launches", type=int, default=None, help="Ignored compatibility flag.")
    parser.add_argument("--dry-run", action="store_true", help="Run one orchestration iteration without looping.")
    args = parser.parse_args(argv)

    forwarded = ["--poll-seconds", str(args.poll_seconds)]
    if args.dry_run:
        forwarded.extend(["--once", "--dry-run"])
    return orchestrator_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
