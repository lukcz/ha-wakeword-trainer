#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  "$SCRIPT_DIR/setup_environment.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODE="${1:-wakeword}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  wakeword)
    exec python "$SCRIPT_DIR/train_microwakeword.py" --config "$SCRIPT_DIR/configs/microwakeword_example.yaml" "$@"
    ;;
  vad)
    exec python "$SCRIPT_DIR/train_microwakeword.py" --config "$SCRIPT_DIR/configs/polish_vad.yaml" "$@"
    ;;
  vad-official)
    exec python "$SCRIPT_DIR/fetch_voice_pe_vad.py" "$@"
    ;;
  *)
    printf 'Usage: %s [wakeword|vad|vad-official] [extra args]\n' "$0" >&2
    exit 1
    ;;
esac
