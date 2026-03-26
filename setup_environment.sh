#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/lukcz/ha-wakeword-trainer.git}"
REPO_DIR="${REPO_DIR:-$HOME/ha-wakeword-trainer}"

log() {
  printf '[setup] %s\n' "$*" >&2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_system_deps() {
  log "Installing system packages"
  if need_cmd apt-get; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv ffmpeg git wget curl
    return
  fi

  if need_cmd dnf; then
    sudo dnf install -y python3 python3-pip python3-venv ffmpeg git wget curl
    return
  fi

  log "Unsupported package manager. Install manually: python3 python3-pip python3-venv ffmpeg git wget curl"
  exit 1
}

resolve_repo_dir() {
  if [[ -f "./train_wakeword.py" && -f "./requirements.txt" ]]; then
    pwd
    return
  fi

  if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning repository into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
  else
    log "Repository already present at $REPO_DIR"
  fi

  printf '%s\n' "$REPO_DIR"
}

main() {
  if [[ "${OSTYPE:-}" != linux-gnu* ]]; then
    log "This script is intended for Linux or WSL2."
    exit 1
  fi

  install_system_deps

  local repo_dir
  repo_dir="$(resolve_repo_dir)"
  local venv_dir="$repo_dir/.venv"

  chmod +x \
    "$repo_dir/setup_environment.sh" \
    "$repo_dir/train.sh" \
    "$repo_dir/train_vad_full.sh" \
    "$repo_dir/train_wakeword_full.sh"

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating virtual environment in $venv_dir"
    python3 -m venv "$venv_dir"
  fi

  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"

  log "Upgrading pip tooling"
  python -m pip install --upgrade pip wheel "setuptools<81"

  log "Installing Python requirements"
  python -m pip install -r "$repo_dir/requirements.txt"

  log "Environment is ready"
  printf '\n'
  printf 'Next steps:\n'
  printf '  cd %s\n' "$repo_dir"
  printf '  source .venv/bin/activate\n'
  printf '  ./train.sh vad\n'
  printf '\n'
  printf 'Wake word example:\n'
  printf '  ./train.sh wakeword --config configs/wakeword_example.yaml\n'
}

main "$@"
