#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  "$SCRIPT_DIR/setup_environment.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODE="${1:-vad}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$MODE" in
  vad)
    exec python "$SCRIPT_DIR/train_wakeword.py" --config "$SCRIPT_DIR/configs/polish_vad.yaml" "$@"
    ;;
  wakeword)
    exec python "$SCRIPT_DIR/train_wakeword.py" --config "$SCRIPT_DIR/configs/wakeword_example.yaml" "$@"
    ;;
  *)
    printf 'Usage: %s [vad|wakeword] [extra train_wakeword.py args]\n' "$0" >&2
    exit 1
    ;;
esac
