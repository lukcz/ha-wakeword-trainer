#!/bin/bash
# =============================================================================
# 🏠 HA Wake Word Trainer - Environment Setup Script
# =============================================================================
# 
# This script sets up everything needed to train custom wake words.
# Run it once on a fresh WSL2 Ubuntu or Linux system.
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# After setup completes, activate the environment with:
#   source ~/wakeword-env/bin/activate
#
# =============================================================================

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🧦 Wake Word Trainer - Environment Setup                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is designed for Linux/WSL2${NC}"
    echo "For Windows, please use WSL2 with Ubuntu."
    exit 1
fi

# =============================================================================
# STEP 1: System Dependencies
# =============================================================================
echo -e "${YELLOW}[1/5] Installing system dependencies...${NC}"

if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        wget \
        curl
    echo -e "${GREEN}  ✓ System dependencies installed${NC}"
elif command -v dnf &> /dev/null; then
    sudo dnf install -y \
        python3 \
        python3-pip \
        ffmpeg \
        git \
        wget \
        curl
    echo -e "${GREEN}  ✓ System dependencies installed${NC}"
else
    echo -e "${RED}  ✗ Unsupported package manager. Please install manually:${NC}"
    echo "    python3, python3-pip, python3-venv, ffmpeg, git, wget"
    exit 1
fi

# =============================================================================
# STEP 2: Python Virtual Environment
# =============================================================================
echo ""
echo -e "${YELLOW}[2/5] Creating Python virtual environment...${NC}"

VENV_PATH="$HOME/wakeword-env"

if [ -d "$VENV_PATH" ]; then
    echo "  Virtual environment already exists at $VENV_PATH"
    read -p "  Delete and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        python3 -m venv "$VENV_PATH"
    fi
else
    python3 -m venv "$VENV_PATH"
fi

# Activate environment
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}  ✓ Virtual environment created at $VENV_PATH${NC}"

# =============================================================================
# STEP 3: Python Packages
# =============================================================================
echo ""
echo -e "${YELLOW}[3/5] Installing Python packages...${NC}"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install packages in correct order to avoid conflicts
pip install 'numpy>=1.24.0,<2.0'
pip install 'pyarrow>=12.0.0,<15.0.0'
pip install tensorflow==2.16.1
pip install datasets==2.14.0
pip install edge-tts soundfile librosa scipy
pip install pyyaml requests tqdm mmap-ninja webrtcvad audioread

echo -e "${GREEN}  ✓ Python packages installed${NC}"

# =============================================================================
# STEP 4: micro-wake-word
# =============================================================================
echo ""
echo -e "${YELLOW}[4/5] Installing micro-wake-word...${NC}"

MWW_PATH="$HOME/micro-wake-word"

if [ -d "$MWW_PATH" ]; then
    echo "  micro-wake-word already exists, updating..."
    cd "$MWW_PATH"
    git pull
else
    git clone https://github.com/OHF-Voice/micro-wake-word.git "$MWW_PATH"
    cd "$MWW_PATH"
fi

pip install -e .

# Apply numpy compatibility patch
echo "  Applying compatibility patches..."
TRAIN_PY="$MWW_PATH/microwakeword/train.py"
if grep -q '\.numpy()' "$TRAIN_PY" 2>/dev/null; then
    sed -i 's/\.numpy()//g' "$TRAIN_PY"
    echo -e "${GREEN}  ✓ Patched train.py for numpy compatibility${NC}"
fi

# Apply TensorFlow 2.16+ compatibility patch (microfrontend removed)
AUDIO_UTILS="$MWW_PATH/microwakeword/audio/audio_utils.py"
if [ -f "$AUDIO_UTILS" ]; then
    # Check if file has the old import
    if grep -q "from tensorflow.lite.experimental.microfrontend" "$AUDIO_UTILS" 2>/dev/null; then
        # Replace with try/except fallback
        sed -i 's/from tensorflow.lite.experimental.microfrontend.python.ops import (/try:\n    from tensorflow.lite.experimental.microfrontend.python.ops import (/' "$AUDIO_UTILS"
        sed -i 's/audio_microfrontend_op as frontend_op,/audio_microfrontend_op as frontend_op\n    )\nexcept ImportError:\n    import warnings\n    warnings.warn("TensorFlow microfrontend not available, using fallback")\n    frontend_op = None\n# Original import kept for older TF versions:/' "$AUDIO_UTILS"
        echo -e "${GREEN}  ✓ Patched audio_utils.py for TF 2.16+ compatibility${NC}"
    fi
fi

echo -e "${GREEN}  ✓ micro-wake-word installed${NC}"

# =============================================================================
# STEP 5: Download helper script
# =============================================================================
echo ""
echo -e "${YELLOW}[5/5] Downloading helper script...${NC}"

SCRIPT_URL="https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/train_wakeword.py"
SCRIPT_PATH="$HOME/train_wakeword.py"

wget -O "$SCRIPT_PATH" "$SCRIPT_URL"
chmod +x "$SCRIPT_PATH"

echo -e "${GREEN}  ✓ Helper script downloaded to $SCRIPT_PATH${NC}"

# =============================================================================
# DONE!
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  To start training correctly:                                ║"
echo "║                                                              ║"
echo "║    git clone https://github.com/lukcz/ha-wakeword-trainer.git║"
echo "║    cd ha-wakeword-trainer                                    ║"
echo "║    source ~/wakeword-env/bin/activate                        ║"
echo "║    pip install -r requirements.txt                           ║"
echo "║    python train_wakeword.py --config configs/polish_vad.yaml ║"
echo "║                                                              ║"
echo "║  Note: ~/train_wakeword.py is only a helper copy.            ║"
echo "║  For VAD mode and configs, run from a repo checkout.         ║"
echo "║                                                              ║"
echo "║  First run will download ~5GB of training data.              ║"
echo "║  Subsequent runs will be much faster (~30 min).              ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Add activation hint to bashrc
if ! grep -q "wakeword-env" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# Wake Word Trainer environment" >> "$HOME/.bashrc"
    echo "alias wakeword='source ~/wakeword-env/bin/activate && cd ~'" >> "$HOME/.bashrc"
    echo -e "${GREEN}Added 'wakeword' alias to ~/.bashrc${NC}"
    echo "Run 'wakeword' to quickly activate the environment."
fi
