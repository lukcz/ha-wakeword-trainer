# 🏠 HA Wake Word Trainer

All-in-one training script for custom [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) models compatible with **Home Assistant Voice PE** and **ESPHome**.

Train your own wake word in any language — just run the script and wait!

[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Compatible-blue?logo=homeassistant)](https://www.home-assistant.io/)
[![ESPHome](https://img.shields.io/badge/ESPHome-Ready-green?logo=esphome)](https://esphome.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎤 **Automatic sample generation** using Edge TTS (Microsoft's neural voices)
- 📥 **Auto-downloads all training data** (~5GB, cached for future runs)
- 🌍 **Multi-language support**: Polish, English, German, Spanish, French
- 🧠 **Smart augmentation**: Room acoustics, background noise, music
- 📦 **Ready-to-use output**: `.tflite` model + `.json` manifest for ESPHome
- 🔧 **Auto-patches** TensorFlow compatibility issues

## Requirements

### System Requirements
- **OS**: WSL2 Ubuntu 22.04+ (recommended) or native Linux
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~10GB free space (for datasets)
- **CPU**: Multi-core recommended (training is CPU-intensive)

### Quick Setup (Automatic)

The easiest way to set up everything:

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/setup_environment.sh
chmod +x setup_environment.sh
./setup_environment.sh
```

This automatically:
- Installs system dependencies (ffmpeg, python3, etc.)
- Creates a Python virtual environment
- Installs all required packages with correct versions
- Clones and installs micro-wake-word
- Downloads the training script
- Patches known compatibility issues

After setup, just run:
```bash
source ~/wakeword-env/bin/activate
python ~/train_wakeword.py "Hey Jarvis" --lang en
```

### Manual Setup

If you prefer to set up manually:

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg git wget

# 2. Create and activate virtual environment
python3 -m venv ~/wakeword-env
source ~/wakeword-env/bin/activate

# 3. Install Python packages (ORDER MATTERS!)
pip install --upgrade pip wheel setuptools
pip install 'numpy>=1.24.0,<2.0'
pip install 'pyarrow>=12.0.0,<15.0.0'
pip install tensorflow==2.16.1
pip install datasets==2.14.0
pip install edge-tts soundfile librosa scipy pyyaml requests tqdm mmap-ninja webrtcvad

# 4. Clone and install micro-wake-word
git clone https://github.com/OHF-Voice/micro-wake-word.git ~/micro-wake-word
cd ~/micro-wake-word && pip install -e .

# 5. Apply compatibility patch
sed -i 's/\.numpy()//g' ~/micro-wake-word/microwakeword/train.py

# 6. Download the trainer script
wget -O ~/train_wakeword.py https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/train_wakeword.py
```

### Using requirements.txt

Alternatively, after creating your virtual environment:

```bash
source ~/wakeword-env/bin/activate
pip install -r https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/requirements.txt
```

### Windows Users (WSL2 Required)

Native Windows is **not supported** due to TensorFlow limitations. Use WSL2:

```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04
```

Then follow the Linux setup instructions inside WSL2.

### Common Issues

#### `ModuleNotFoundError: No module named 'tensorflow.lite.experimental'`
**Cause:** TensorFlow 2.16+ removed the `tensorflow.lite.experimental.microfrontend` module.
**Fix:** If using manual setup, apply the patch to micro-wake-word or use the provided `setup_environment.sh` which handles this automatically.

#### FFmpeg not found (Windows)
**Cause:** FFmpeg is not installed or not in PATH.
**Fix:** Download from https://www.gyan.dev/ffmpeg/builds/ and add `bin/` folder to PATH.

#### MUSAN download rate limited
**Cause:** HuggingFace rate limiting.
**Fix:** Training will continue without MUSAN (optional dataset). For best results, wait 1-2 hours and retry, or create a HuggingFace account and set `HF_TOKEN`.

### About VAD (Voice Activity Detection)

This repository trains a **wake-word model**, not a standalone VAD model.

- **Wake word detection** answers: "did the user say the target phrase?"
- **VAD** answers: "is there speech in this audio at all?"

Why this matters:
- You may still see `webrtcvad` in the dependency list because parts of the broader audio/tooling stack use it.
- The exported files from this repo are a **custom wake-word `.tflite` + `.json` manifest** for `micro_wake_word`.
- If your Home Assistant / ESPHome voice pipeline also uses VAD, that VAD is configured separately from the wake-word model trained here.

Practical rule of thumb:
- If the device wakes too easily on random sounds, tune the **wake-word threshold** first.
- If the device has trouble deciding whether anyone is speaking at all, that is usually a **VAD / voice pipeline** issue, not a problem with the trained wake-word model itself.

### GPU Support (Optional)

GPU training often has compatibility issues with newer cards (RTX 40xx/50xx). 
The script defaults to CPU training, which works reliably.

If you want to try GPU:
1. Install CUDA toolkit in WSL2
2. Run with `--cpu_only=False`

Note: CPU training takes 15-30 minutes, which is acceptable for most use cases.

## Quick Start

### Basic Usage

```bash
# Polish wake word
python train_wakeword.py "Hej Zgredek"

# English wake word  
python train_wakeword.py "Hey Jarvis" --lang en

# With custom variations
python train_wakeword.py "Hey Computer" --lang en --variations "Hey PC,Yo Computer"
```

### Output

After training completes, you'll find:
```
./wakeword_training/
├── hej_zgredek.tflite    # Model file (~100KB)
├── hej_zgredek.json      # ESPHome manifest
├── generated_samples/     # TTS audio files
├── augmented_features/    # Processed spectrograms
└── trained_model/         # Training checkpoints
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wake_word` | (required) | The wake word phrase to train |
| `--variations` | "" | Comma-separated alternative phrasings |
| `--lang` | "pl" | Language: pl, en, de, es, fr |
| `--output_dir` | "./wakeword_training" | Output directory |
| `--steps` | 10000 | Training steps |
| `--batch_size` | 128 | Batch size for training |
| `--author` | "custom" | Author name for manifest |
| `--probability_cutoff` | 0.7 | Detection threshold (0.0-1.0) |
| `--data_dir` | None | Path to existing augmentation data |
| `--skip_generate` | False | Skip TTS sample generation |
| `--skip_download` | False | Skip dataset downloads |

## Examples

### Polish Wake Word (Recommended Settings)
```bash
python train_wakeword.py "Hej Zgredek" \
    --variations "Hej Zgredku,hej zgredek,Heej Zgredek" \
    --steps 15000 \
    --author "myname"
```

### English Wake Word with High Accuracy
```bash
python train_wakeword.py "Hey Assistant" \
    --lang en \
    --steps 20000 \
    --probability_cutoff 0.8
```

### Quick Test Run (Lower Quality)
```bash
python train_wakeword.py "Test Word" \
    --lang en \
    --steps 5000
```

### Reuse Downloaded Data
```bash
# First run downloads everything
python train_wakeword.py "Wake Word One" --output_dir ./training1

# Second run reuses the data (much faster!)
python train_wakeword.py "Wake Word Two" --data_dir ./training1 --output_dir ./training2
```

## Training Time

| Phase | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Sample generation | 5-10 min | 5-10 min |
| Dataset download | 30-60 min | Skipped (cached) |
| Feature generation | 10-20 min | 10-20 min |
| Model training | 10-30 min | 10-30 min |
| **Total** | **1-2 hours** | **25-60 min** |

*Times based on modern CPU. Training uses CPU by default (GPU often has compatibility issues with new hardware).*

## Parameter Trade-offs

### Training Steps (`--steps`)
| Value | Quality | Time | Use Case |
|-------|---------|------|----------|
| 5000 | Low | ~5 min | Quick testing |
| 10000 | Good | ~15 min | Default, most use cases |
| 15000 | Better | ~25 min | Production models |
| 20000+ | Best | ~35+ min | Maximum accuracy |

### Probability Cutoff (`--probability_cutoff`)
| Value | False Positives | False Negatives | Use Case |
|-------|-----------------|-----------------|----------|
| 0.5 | Many | Few | Very responsive, noisy |
| 0.7 | Balanced | Balanced | **Default, recommended** |
| 0.8 | Few | Some | Quiet environments |
| 0.9 | Rare | Many | High-precision needed |

## Choosing Cutoff Based on Training Results

After training completes, the script outputs metrics like:

```
INFO:absl:Cutoff 0.86: frr=0.0818; faph=0.000
INFO:absl:Cutoff 0.85: frr=0.0818; faph=0.187
INFO:absl:Cutoff 0.81: frr=0.0793; faph=0.562
INFO:absl:Cutoff 0.75: frr=0.0691; faph=0.750
INFO:absl:Cutoff 0.67: frr=0.0588; faph=0.937
```

### Understanding the Metrics

| Metric | Full Name | Meaning |
|--------|-----------|---------|
| **frr** | False Rejection Rate | % of wake words NOT detected (lower = better) |
| **faph** | False Accepts Per Hour | How often it triggers incorrectly (lower = better) |
| **Cutoff** | Probability Threshold | The `probability_cutoff` value for ESPHome |

### How to Choose

1. **Find your acceptable faph**: 
   - `0.0` = Never triggers incorrectly (but may miss wake words)
   - `0.5-1.0` = Triggers falsely ~once per hour (reasonable)
   - `2.0+` = May be annoying in quiet environments

2. **Check the frr at that cutoff**:
   - `< 5%` = Excellent (misses 1 in 20 wake words)
   - `5-10%` = Good (misses 1 in 10-20)
   - `> 10%` = May feel unresponsive

3. **Balance based on your use case**:

| Environment | Recommended faph | Typical Cutoff |
|-------------|------------------|----------------|
| Noisy (kitchen, living room) | 1.0-2.0 | 0.60-0.70 |
| Quiet (bedroom, office) | 0.0-0.5 | 0.80-0.90 |
| Mixed/General | 0.5-1.0 | 0.70-0.80 |

### Example Decision

From the training output above:
- At cutoff **0.75**: frr=6.9%, faph=0.75 → **Balanced choice**
- At cutoff **0.85**: frr=8.2%, faph=0.19 → **Fewer false triggers, slightly less responsive**
- At cutoff **0.67**: frr=5.9%, faph=0.94 → **More responsive, ~1 false trigger/hour**

**Recommendation**: Start with the cutoff where `faph < 1.0` and `frr < 10%`, then adjust based on real-world testing.

### Quick Reference Table

| Your Priority | Choose Cutoff Where |
|---------------|---------------------|
| Never miss wake word | faph ~1-2, lowest frr |
| Never false trigger | faph = 0, accept higher frr |
| Balanced | faph < 1.0, frr < 10% |

### Language-Specific Tips

| Language | Recommended Variations | Notes |
|----------|----------------------|-------|
| Polish | Add vocative case (e.g., "Zgredku") | Edge TTS handles Polish well |
| English | Add informal variants | Multiple accents available |
| German | Include formal/informal | Good voice quality |

## Deploying to ESPHome

After training completes, you'll have two files:
- `your_wake_word.tflite` — The model (~100KB)
- `your_wake_word.json` — Manifest for ESPHome

### Step 1: Create a GitHub Repository

If you don't have one yet:

```bash
# Install GitHub CLI (if not installed)
sudo apt install gh

# Login to GitHub
gh auth login

# Create a new repo
gh repo create my-wake-word --public --description "Custom wake word model"

# Clone it
git clone https://github.com/YOUR_USERNAME/my-wake-word
cd my-wake-word
```

### Step 2: Upload Files to GitHub Releases

**Option A: Using GitHub Web Interface (Easiest)**

1. Go to your repo: `https://github.com/YOUR_USERNAME/my-wake-word`
2. Click **"Releases"** (right sidebar)
3. Click **"Create a new release"**
4. Fill in:
   - **Tag**: `v1.0.0`
   - **Title**: `Wake Word Model v1.0.0`
   - **Description**: Your wake word name and settings
5. **Drag and drop** both `.tflite` and `.json` files into the "Attach binaries" area
6. Click **"Publish release"**

**Option B: Using GitHub CLI**

```bash
cd ~/wakeword_training

# Create a release with both files
gh release create v1.0.0 \
    --repo YOUR_USERNAME/my-wake-word \
    --title "Wake Word Model v1.0.0" \
    --notes "Custom wake word: Hey Jarvis" \
    hey_jarvis.tflite \
    hey_jarvis.json
```

**Option C: Using Git + Web Upload**

```bash
cd my-wake-word

# Copy files
cp ~/wakeword_training/hey_jarvis.* .

# Commit to repo (optional, for backup)
git add .
git commit -m "Add wake word model"
git push

# Then create release via web interface and upload files
```

### Step 3: Get the Raw URL

After creating the release, your files will be available at:
```
https://github.com/YOUR_USERNAME/my-wake-word/releases/download/v1.0.0/hey_jarvis.json
https://github.com/YOUR_USERNAME/my-wake-word/releases/download/v1.0.0/hey_jarvis.tflite
```

The JSON manifest should point to the tflite file. Edit `hey_jarvis.json` before uploading:

```json
{
  "version": 2,
  "wake_word": "hey jarvis",
  "author": "your_name",
  "model": "https://github.com/YOUR_USERNAME/my-wake-word/releases/download/v1.0.0/hey_jarvis.tflite",
  "micro": {
    "probability_cutoff": 0.7,
    "sliding_window_size": 5
  }
}
```

**Important**: The `model` field in JSON must contain the **full URL** to the `.tflite` file!

### Step 4: Update ESPHome Config

Edit your Voice PE YAML file (e.g., `voice-pe.yaml`):

```yaml
micro_wake_word:
  models:
    # Your custom wake word
    - model: https://github.com/YOUR_USERNAME/my-wake-word/releases/download/v1.0.0/hey_jarvis.json
      probability_cutoff: 0.7
      sliding_window_size: 5
    
    # Backup wake word (optional but recommended)
    - model: okay_nabu
```

**Configuration Options:**

| Option | Description | Recommended |
|--------|-------------|-------------|
| `probability_cutoff` | Detection threshold (0.0-1.0) | Start with 0.7 |
| `sliding_window_size` | Consecutive detections needed | 5 (default) |

### Step 5: Flash Your Device

```bash
# Using ESPHome CLI
esphome run voice-pe.yaml

# Or via Home Assistant ESPHome add-on
# Go to ESPHome Dashboard → Your Device → Install
```

### Step 6: Test and Tune

1. **Say your wake word** clearly from 1-2 meters
2. **Check logs** for detection events:
   ```bash
   esphome logs voice-pe.yaml
   ```

**Tuning Tips:**

| Problem | Solution |
|---------|----------|
| Not responding | Lower `probability_cutoff` to 0.5-0.6 |
| Too many false triggers | Raise `probability_cutoff` to 0.8-0.9 |
| Inconsistent detection | Increase `sliding_window_size` to 7-10 |
| Works close but not far | Retrain with more variations |

### Complete Example

Here's a complete Voice PE config with custom wake word:

```yaml
esphome:
  name: voice-assistant
  friendly_name: Voice Assistant

esp32:
  board: esp32-s3-devkitc-1
  framework:
    type: esp-idf

# ... (other config like wifi, api, etc.)

micro_wake_word:
  models:
    - model: https://github.com/lukcz/ha-wakeword-trainer/releases/download/v1.0.0/hej_zgredek.json
      probability_cutoff: 0.7
      sliding_window_size: 5
  on_wake_word_detected:
    - light.turn_on:
        id: led_ring
        effect: "Listening"
    - voice_assistant.start:

voice_assistant:
  microphone: mic
  speaker: speaker
  on_end:
    - light.turn_off: led_ring
```

## Troubleshooting

### "Module not found" errors
```bash
pip install edge-tts datasets==2.14.0 soundfile librosa pyyaml requests tqdm
pip install 'numpy<2' 'pyarrow>=12,<15' tensorflow==2.16.1
```

### Training crashes with GPU errors
The script uses CPU by default. If you want to try GPU:
```bash
python train_wakeword.py "Wake Word" --cpu_only=False
```
Note: RTX 40xx/50xx series may have compatibility issues with TensorFlow.

### "numpy.ndarray has no attribute numpy"
The script auto-patches this. If it persists:
```bash
# Manual fix
sed -i 's/\.numpy()//' /path/to/micro-wake-word/microwakeword/train.py
```

### Downloads fail or timeout
```bash
# Use existing data from another training
python train_wakeword.py "New Word" --data_dir /path/to/previous/training
```

### Model not responding to wake word
1. Lower `probability_cutoff` to 0.5-0.6
2. Add more phrase variations
3. Increase training steps to 15000+
4. Check if TTS pronunciation matches your speech

### Too many false activations
1. Raise `probability_cutoff` to 0.8-0.9
2. Increase `negative_class_weight` in code
3. Use a more distinctive wake word (2+ syllables)

### No .tflite file after training
Sometimes the automatic conversion fails. Run manual conversion:

```bash
cd ~/micro-wake-word

CUDA_VISIBLE_DEVICES=-1 python -m microwakeword.model_train_eval \
    --training_config='/path/to/your/training_parameters.yaml' \
    --train 0 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block "1,1,1,1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3
```

Key: `--train 0` skips training and only converts the model.

The .tflite file will be in:
```
trained_model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SAMPLE GENERATION                                        │
│    Edge TTS → Multiple voices, speeds, pitches              │
│    "Hej Zgredek" → 2000+ audio variations                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. AUGMENTATION                                             │
│    + Room acoustics (MIT RIRs)                              │
│    + Background music (FMA)                                 │
│    + Environmental sounds (AudioSet)                        │
│    + Noise, EQ, pitch shifts                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAINING                                                 │
│    MixedNet CNN architecture                                │
│    Positive samples + Negative samples                      │
│    10000 steps → 99%+ accuracy                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EXPORT                                                   │
│    Quantized TFLite model (~100KB)                          │
│    JSON manifest for ESPHome                                │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

The script uses the **MixedNet** architecture optimized for wake word detection:

| Layer | Filters | Description |
|-------|---------|-------------|
| Conv1D | 32 | Initial feature extraction |
| MixConv Block 1 | 64 | Kernel sizes: [5] |
| MixConv Block 2 | 64 | Kernel sizes: [7, 11] |
| MixConv Block 3 | 64 | Kernel sizes: [9, 15] |
| MixConv Block 4 | 64 | Kernel sizes: [23] |
| Output | 1 | Wake word probability |

**Model size**: ~100KB (quantized)  
**Parameters**: ~26,000  
**Inference**: ~10ms per frame on ESP32-S3

## Datasets Used

| Dataset | Size | Purpose |
|---------|------|---------|
| Generated samples | ~500MB | Positive wake word examples |
| MIT RIRs | ~500MB | Room acoustics simulation |
| FMA (music) | ~2GB | Background music augmentation |
| Negative datasets | ~1GB | Speech/noise that's NOT the wake word |

All datasets are automatically downloaded on first run and cached for future use.

## Credits

- [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) by Kevin Ahrendt
- [Edge TTS](https://github.com/rany2/edge-tts) for sample generation
- [ESPHome](https://esphome.io/) for device integration
- Training data from MIT, FMA, AudioSet, and HuggingFace

## License

MIT License - Use freely for personal and commercial projects.

---

Made with ❤️ for the Home Assistant community

Created by [lukcz](https://github.com/lukcz) with help from 🧦 Zgredek
