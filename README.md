# openwakeword-trainer

Train custom wake word models with [openWakeWord](https://github.com/dscripka/openWakeWord). A granular 13-step pipeline with compatibility patches for torchaudio 2.10+, Piper TTS, and speechbrain. Generates tiny ONNX models (~200 KB) for real-time keyword detection — like building your own "Hey Siri" trigger.

## What It Does

This toolkit automates the entire openWakeWord training process. It now also supports a practical **VAD mode** for training a Polish speech-vs-non-speech classifier from datasets selected in the CLI/script.

1. **Synthesizes** thousands of speech clips using Piper TTS with varied voices and accents
2. **Augments** clips with real-world noise, music, and room impulse responses
3. **Trains** a small DNN classifier optimized for always-on, low-latency detection
4. **Exports** a tiny ONNX model you can deploy anywhere

The result is a ~200 KB model that runs on CPU in real-time with negligible resource usage.

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **WSL2 or Linux** | Ubuntu recommended (`wsl --install -d Ubuntu` on Windows) |
| **NVIDIA GPU** | CUDA drivers installed (WSL2 includes CUDA passthrough automatically) |
| **Disk space** | ~15 GB free (temporary downloads; deletable after training) |
| **Python 3.10+** | Inside WSL2/Linux (`python3 --version`) |
| **Time** | ~1–2 hours with GPU, 12–24 hours CPU-only |

### Verify CUDA (WSL2)

```bash
wsl
nvidia-smi
```

You should see your GPU listed. If not, update your NVIDIA Windows driver to the latest version.

## Quick Start

### Option A: One-liner

```bash
# From PowerShell (Windows) — cd to the repo first:
cd path\to\openwakeword-trainer
wsl -- bash train.sh

# Or from within WSL2/Linux:
cd /mnt/c/path/to/openwakeword-trainer
bash train.sh
```

This creates an isolated virtualenv, installs dependencies, downloads datasets, trains the model, and exports the result.

### Option B: Step-by-step

```bash
# Enter WSL2 and navigate to the repo
wsl
cd /mnt/c/path/to/openwakeword-trainer

# Create & activate a training venv (use native filesystem, not /mnt/c/)
python3 -m venv ~/.oww-trainer-venv
source ~/.oww-trainer-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python train_wakeword.py

# Or resume from where you left off
python train_wakeword.py --from augment
```

### Train Your Own Wake Word

1. Copy the example config:
   ```bash
   cp configs/hey_echo.yaml configs/my_word.yaml
   ```

2. Edit `configs/my_word.yaml`:
   ```yaml
   model_name: "my_word"
   target_phrase:
     - "hey computer"
   custom_negative_phrases:
     - "hey commuter"
     - "computer"
     - "hey"
   ```

3. Train:
   ```bash
   python train_wakeword.py --config configs/my_word.yaml
   ```

4. Find your model in `export/my_word.onnx` (and `export/my_word.onnx.data`).

### Train a Polish VAD model from datasets

Use `configs/polish_vad.yaml` when you want **speech vs non-speech** instead of a phrase detector.

## VAD mode: complete workflow

### What VAD mode does

In `--mode vad`, the pipeline does **not** synthesize a target phrase with Piper.
Instead it:

1. reads **real speech clips** from your positive datasets
2. reads **non-speech / ambience / music** from your negative datasets
3. resamples everything to **16 kHz mono WAV**
4. prepares these splits under `output/<model_name>/`:
   - `positive_train`
   - `positive_test`
   - `negative_train`
   - `negative_test`
5. runs augmentation / feature extraction / training as usual
6. exports:
   - `export/<model_name>.onnx`
   - `export/<model_name>.onnx.data` (if external weights are used)
   - `export/<model_name>.json` (ESPHome-style VAD manifest)

This means VAD mode is for **speech-vs-non-speech classification**, not wake phrase detection.

### Supported dataset names

The built-in dataset registry currently recognizes these names:

| Dataset name | Kind | Intended content |
|---|---|---|
| `mc_speech` | positive | Polish speech clips |
| `bigos` | positive | Polish speech clips |
| `pl_speech` | positive | Additional Polish speech clips |
| `no_speech` | negative | Silence / HVAC / room tone |
| `dinner_party` | negative | Crowd / babble / kitchen ambience |
| `musan` | negative | MUSAN noise / music / speech negatives |
| `fma` | negative | FMA music |
| `audioset` | negative | AudioSet background audio |

You can point any of them to local folders with repeated `--dataset-path name=/abs/path` flags.

### Expected input data

**Positive datasets** should contain:
- real human speech
- many speakers
- different microphones / rooms if possible
- audio files in `.wav`, `.flac`, or `.mp3`

**Negative datasets** should contain:
- silence / room tone
- household noise
- music
- ambience
- ideally **not** clean foreground speech

Important:
- The script currently requires at least **2 positive** and **2 negative** audio files to proceed.
- If a selected dataset path does not exist, the script logs a warning.
- In practice you want **thousands** of clips for a useful VAD, not the bare minimum.

### Minimal example

```bash
python train_wakeword.py \
  --config configs/polish_vad.yaml \
  --mode vad \
  --positive-datasets mc_speech \
  --negative-datasets no_speech,musan \
  --dataset-path mc_speech=/data/polish/mc_speech \
  --dataset-path no_speech=/data/no_speech
```

### Recommended full example

```bash
python train_wakeword.py \
  --config configs/polish_vad.yaml \
  --mode vad \
  --positive-datasets mc_speech,bigos,pl_speech \
  --negative-datasets no_speech,dinner_party,musan,fma,audioset \
  --dataset-path mc_speech=/data/polish/mc_speech \
  --dataset-path bigos=/data/polish/bigos \
  --dataset-path pl_speech=/data/polish/extra_speech \
  --dataset-path no_speech=/data/no_speech \
  --dataset-path dinner_party=/data/dinner_party \
  --dataset-path musan=/data/musan \
  --dataset-path fma=/data/fma_small \
  --dataset-path audioset=/data/audioset_16k
```

### Resume a failed VAD run

```bash
python train_wakeword.py --config configs/polish_vad.yaml --mode vad --from augment
```

### Run only one VAD step

```bash
python train_wakeword.py --config configs/polish_vad.yaml --mode vad --step verify-clips
```

### Inspect only, without side effects

```bash
python train_wakeword.py --config configs/polish_vad.yaml --mode vad --verify-only
```

### VAD CLI parameters

These CLI flags are supported directly by `train_wakeword.py` for VAD workflows:

| Flag | Description |
|---|---|
| `--config FILE` | Path to YAML config, e.g. `configs/polish_vad.yaml` |
| `--mode vad` | Forces VAD mode even if config says otherwise |
| `--positive-datasets CSV` | Comma-separated positive dataset names |
| `--negative-datasets CSV` | Comma-separated negative dataset names |
| `--dataset-path NAME=PATH` | Override a dataset location; repeat as needed |
| `--from NAME` | Resume pipeline from a specific step |
| `--step NAME` | Run exactly one step |
| `--verify-only` | Run only verify steps |
| `--list-steps` | Print all pipeline step names |

### VAD config parameters (`configs/polish_vad.yaml`)

These settings matter most in VAD mode:

| Key | Example | Meaning |
|---|---|---|
| `mode` | `"vad"` | Enables speech-vs-non-speech flow |
| `model_name` | `"vad_pl"` | Output model basename |
| `datasets.positive` | `["mc_speech", "bigos"]` | Positive dataset names |
| `datasets.negative` | `["no_speech", "musan", "fma"]` | Negative dataset names |
| `dataset_paths` | `{}` | Optional dataset path overrides in YAML |
| `vad_positive_samples` | `12000` | Number of positive training clips to use |
| `vad_negative_samples` | `16000` | Number of negative training clips to use |
| `vad_validation_samples` | `2000` | Validation/test clips per class |
| `n_samples` | `12000` | Fallback positive count if VAD-specific key is missing |
| `n_samples_val` | `2000` | Fallback validation count |
| `augmentation_batch_size` | `16` | Batch size for augmentation step |
| `augmentation_rounds` | `1` | Number of augmentation passes |
| `model_type` | `"dnn"` | openWakeWord classifier type |
| `layer_size` | `32` | Hidden layer width |
| `steps` | `30000` | Training steps |
| `target_false_positives_per_hour` | `0.2` | Target FPR used during training/tuning |
| `export_manifest.*` | see config | ESPHome VAD manifest metadata |

### How sample limits work in VAD mode

The script slices files in this order:
- first `vad_positive_samples` files → `positive_train`
- next `vad_validation_samples` files → `positive_test`
- first `vad_negative_samples` files → `negative_train`
- next `vad_validation_samples` files → `negative_test`

If there are not enough files for the test split, it falls back to the first available files.
So for good train/test separation, provide **more audio files than the configured limits**.

### Output files

After a successful VAD run, expect:

```text
output/vad_pl/
  positive_train/
  positive_test/
  negative_train/
  negative_test/

export/
  vad_pl.onnx
  vad_pl.onnx.data
  vad_pl.json
```

The JSON manifest is written in ESPHome-style format with fields such as:
- `wake_word: "vad"`
- `trained_languages`
- `micro.probability_cutoff`
- `micro.sliding_window_size`
- `micro.tensor_arena_size`
- `micro.feature_step_size`

### Notes and caveats

- VAD mode skips Piper clip synthesis entirely.
- The repo currently trains and exports **ONNX**; the manifest can point at a future `.tflite`, but an ONNX→TFLite conversion / packaging step is still needed for direct ESPHome deployment.
- `target_phrase` and `custom_negative_phrases` remain in the config only because openWakeWord expects standard knobs; in VAD mode they are effectively placeholders.
- For VAD quality, data diversity matters more than synthetic phrase coverage.

## Pipeline Steps

The pipeline runs **13 granular steps**, each with built-in verification. If any step fails, it stops immediately and tells you exactly how to resume.

| # | Step | Description | Time |
|---|------|-------------|------|
| 1 | `check-env` | Verify Python, CUDA, critical imports | instant |
| 2 | `apply-patches` | Patch torchaudio/speechbrain/piper compat | instant |
| 3 | `download` | Download datasets, Piper TTS model, tools | ~30 min |
| 4 | `verify-data` | Check all data files present & sizes | instant |
| 5 | `resolve-config` | Resolve config paths to absolute | instant |
| 6 | `generate` | Generate clips via Piper TTS | ~10 min (GPU) |
| 7 | `resample-clips` | Spot-check clip sample rates | instant |
| 8 | `verify-clips` | Verify clip counts and directories | instant |
| 9 | `augment` | Augment clips & extract mel features | ~30 min |
| 10 | `verify-features` | Check `.npy` feature files & shapes | instant |
| 11 | `train` | Train DNN model + ONNX export | ~30 min (GPU) |
| 12 | `verify-model` | Load-test with ONNX Runtime | instant |
| 13 | `export` | Copy model to `export/` directory | instant |

If any step fails:
```
Pipeline stopped.  Fix the issue above, then resume:
  python train_wakeword.py --from <failed-step>
```

## CLI Reference

```bash
# Full pipeline (all 13 steps)
python train_wakeword.py

# Use a custom config
python train_wakeword.py --config configs/my_word.yaml

# Resume from a specific step
python train_wakeword.py --from augment

# Run exactly one step
python train_wakeword.py --step verify-clips

# Check current state without side effects
python train_wakeword.py --verify-only

# Show all available steps
python train_wakeword.py --list-steps

# VAD mode with dataset selection overrides
python train_wakeword.py --config configs/polish_vad.yaml --mode vad \
  --positive-datasets mc_speech,bigos \
  --negative-datasets no_speech,dinner_party,musan,fma \
  --dataset-path mc_speech=/abs/path/to/mc_speech
```

## Using Your Model

The export step produces two files that must be kept together:

- `hey_echo.onnx` — the model graph (~14 KB)
- `hey_echo.onnx.data` — external weights (~200 KB)

Copy **both** files to your project. The trained model works with any openWakeWord-compatible runtime:

```python
from openwakeword.model import Model

oww = Model(wakeword_models=["export/hey_echo.onnx"])

# Feed 16 kHz audio frames
prediction = oww.predict(audio_frame)
```

Or with ONNX Runtime directly:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("export/hey_echo.onnx")
# Input shape: [1, 16, 96] (mel spectrogram features)
result = sess.run(None, {"x": features})
```

## Configuration Reference

See [configs/hey_echo.yaml](configs/hey_echo.yaml) for a fully commented example. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `model_name` | — | Name for the model (used for filenames) |
| `target_phrase` | — | List of phrases to detect |
| `custom_negative_phrases` | `[]` | Phrases to explicitly reject |
| `n_samples` | `50000` | Number of positive training clips |
| `tts_batch_size` | `25` | Piper TTS batch size (reduce for low VRAM) |
| `model_type` | `"dnn"` | `"dnn"` or `"rnn"` |
| `layer_size` | `32` | Hidden layer size (32=fast, 64/128=higher capacity) |
| `steps` | `50000` | Training steps |
| `target_false_positives_per_hour` | `0.2` | Target false positive rate |

## Threshold Tuning

After training, tune the detection threshold for your use case:

| Problem | Fix |
|---------|-----|
| False activations (triggers when you didn't say it) | Increase threshold: 0.5 → 0.6 → 0.7 |
| Missed activations (need to over-pronounce) | Decrease threshold: 0.5 → 0.4 → 0.3 |
| False triggers on similar words | Add to `custom_negative_phrases` and retrain |

## Compatibility Patches

This toolkit includes automatic patches for known breaking changes in modern dependency versions:

| Issue | Affected | Patch |
|-------|----------|-------|
| `torchaudio.load()` removed | torchaudio ≥2.10 | Soundfile-based replacement with automatic 22050→16000 Hz resampling |
| `torchaudio.info()` removed | torchaudio ≥2.10 | Soundfile-based metadata reader |
| `torchaudio.list_audio_backends()` removed | torchaudio ≥2.10 | Returns `["soundfile"]` for speechbrain compat |
| `pkg_resources` removed | setuptools ≥82 | Auto-installs setuptools<82 |
| Piper API change | piper-sample-generator v2+ | Auto-resolves `model=` kwarg |

Patches are applied and verified automatically during the `apply-patches` step.

## Cleanup

After training, reclaim disk space:

```bash
rm -rf data/          # ~12 GB of downloaded datasets
rm -rf output/        # intermediate training artifacts
```

Keep only the `export/` directory with your trained model.

## Troubleshooting

### `piper-phonemize` fails to install
This package only has Linux wheels. Make sure you're running inside WSL2, not native Windows.

### `nvidia-smi` not found in WSL2
Update your NVIDIA Windows driver to the latest version. WSL2 CUDA passthrough is included automatically.

### Training is very slow
Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`. If `False`, everything falls back to CPU.

### Out of GPU memory
Reduce `tts_batch_size` in your config (e.g., 25 → 10).

### Download stalls
Re-run the script — all downloads are idempotent and resume where they left off.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Piper](https://github.com/rhasspy/piper) by Rhasspy for synthetic TTS
- Built with PyTorch, ONNX Runtime, and speechbrain
