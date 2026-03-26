"""Compatibility hooks used by the pipeline.

This repository previously patched several third-party packages in-place.
For now we keep the interface stable but make the step a no-op so the CLI
can run in a clean environment without local monkey patches.
"""

from __future__ import annotations


def apply_all() -> dict[str, str]:
    return {"compat": "OK - no local patches required"}


def verify_all() -> dict[str, bool]:
    return {"compat": True}
