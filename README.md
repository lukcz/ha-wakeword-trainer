# 🧦 Wake Word Trainer

All-in-one training script for custom [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) models compatible with Home Assistant Voice PE and ESPHome.

**Just run it and wait** — everything is automatic, including downloading all required datasets.

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
wget https://raw.githubusercontent.com/lukcz/zgredek-wakeword/main/setup_environment.sh
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
wget -O ~/train_wakeword.py https://raw.githubusercontent.com/lukcz/zgredek-wakeword/main/train_wakeword.py
```

### Using requirements.txt

Alternatively, after creating your virtual environment:

```bash
source ~/wakeword-env/bin/activate
pip install -r https://raw.githubusercontent.com/lukcz/zgredek-wakeword/main/requirements.txt
```

### Windows Users (WSL2 Required)

Native Windows is **not supported** due to TensorFlow limitations. Use WSL2:

```powershell
# In PowerShell (Admin)
wsl --install -d Ubuntu-22.04
```

Then follow the Linux setup instructions inside WSL2.

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
    - model: https://github.com/lukcz/zgredek-wakeword/releases/download/v1.0.0/hej_zgredek.json
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
| AudioSet | ~2GB | Environmental sounds |
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

Made with 🧦 by [Zgredek](https://github.com/lukcz)
