#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/lukcz/ha-wakeword-trainer.git}"
REPO_DIR="${REPO_DIR:-$HOME/ha-wakeword-trainer}"
MWW_REPO="${MWW_REPO:-https://github.com/OHF-Voice/micro-wake-word.git}"
PIPER_PACKAGE="${PIPER_PACKAGE:-piper-sample-generator==3.2.0}"

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
    sudo apt-get install -y \
      build-essential \
      python3 \
      python3-pip \
      python3-venv \
      ffmpeg \
      git \
      wget \
      curl
    return
  fi

  if need_cmd dnf; then
    sudo dnf install -y \
      gcc gcc-c++ make \
      python3 python3-pip python3-venv \
      ffmpeg git wget curl
    return
  fi

  log "Unsupported package manager. Install manually: build tools, python3, python3-pip, python3-venv, ffmpeg, git, wget, curl"
  exit 1
}

resolve_repo_dir() {
  if [[ -f "./README.md" && -f "./train.sh" ]]; then
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

ensure_git_checkout() {
  local repo_url="$1"
  local dest="$2"
  local label="$3"

  if [[ ! -d "$dest/.git" ]]; then
    log "Cloning $label into $dest"
    git clone --depth 1 "$repo_url" "$dest"
  else
    log "$label already present at $dest"
  fi
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
  local third_party_dir="$repo_dir/third_party"
  local mww_dir="$third_party_dir/micro-wake-word"

  mkdir -p "$third_party_dir"

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

  log "Installing repository helper dependencies"
  python -m pip install -r "$repo_dir/requirements.txt"

  log "Installing audio-metadata compatibility fork"
  python -m pip install "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f"

  ensure_git_checkout "$MWW_REPO" "$mww_dir" "micro-wake-word"
  log "Installing microWakeWord"
  python -m pip install -e "$mww_dir"

  log "Installing Piper sample generator"
  python -m pip install "$PIPER_PACKAGE"

  log "Reinstalling a recent audiomentations for microWakeWord compatibility"
  python -m pip install --upgrade "audiomentations>=0.43.0,<1.0.0"

  log "Environment is ready"
  printf '\n'
  printf 'Next steps:\n'
  printf '  cd %s\n' "$repo_dir"
  printf '  source .venv/bin/activate\n'
  printf '  ./train_wakeword_full.sh\n'
  printf '  ./train_vad_full.sh\n'
}

main "$@"
