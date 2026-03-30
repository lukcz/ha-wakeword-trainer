# ha-wakeword-trainer

This repository is now organized around two local training flows for Home Assistant Voice PE:

- `wakeword`: train a custom wake word with the official [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) stack
- `vad`: train a custom speech detector for Polish or other languages, also using a local microWakeWord-based pipeline

The repository still includes a helper to download the official ESPHome VAD model, but that is no longer the default path.

## Quick Start

```bash
wget -O setup_environment.sh https://raw.githubusercontent.com/lukcz/ha-wakeword-trainer/main/setup_environment.sh
chmod +x setup_environment.sh
./setup_environment.sh
cd ~/ha-wakeword-trainer
```

## Train A Wake Word

```bash
./train_wakeword_full.sh
```

This uses `configs/microwakeword_example.yaml` by default.

To train your own wake word config:

```bash
cp configs/microwakeword_example.yaml configs/my_model.yaml
./train.sh wakeword --config configs/my_model.yaml
```

## Train A Custom VAD

```bash
./train_vad_full.sh
```

This uses `configs/polish_vad.yaml` by default.

Alternative Polish VAD presets:

- `configs/polish_vad_balanced.yaml`: less conservative than the default, better balance of recall vs false activations
- `configs/polish_vad_high_recall.yaml`: pushes recall harder, useful if the detector misses too much speech
- `configs/polish_vad_low_fp.yaml`: prioritizes reducing ambient false positives, even if recall drops a bit
- `configs/polish_vad_public_research.yaml`: research preset that adds VoxPopuli PL, MLS Polish, and optional Sounds of Home negatives

Archived experimental presets were moved to `configs/archive/`. Keep them for reference, but start new experiments from the small set above.

Examples:

```bash
python train_microwakeword.py --config configs/polish_vad_balanced.yaml
python train_microwakeword.py --config configs/polish_vad_high_recall.yaml
python train_microwakeword.py --config configs/polish_vad_low_fp.yaml
python train_microwakeword.py --config configs/polish_vad_public_research.yaml
python train_microwakeword.py --config configs/polish_vad_balanced.yaml --step audit-validation
```

You can tune the locally downloaded ambient subsets in that config:

- `asset_subsets.audioset_max_clips`
- `asset_subsets.fma_max_clips`
- `asset_subsets.io_workers`
- `asset_subsets.bootstrap_workers`

For public background datasets with long recordings, the bootstrap can also cut
each source file into shorter augmentation clips:

- `segment_duration_s`
- `segment_overlap_s`
- `min_segment_duration_s`
- `runtime.device`
- `runtime.allow_cpu_fallback`

The VAD flow:

1. downloads augmentation assets
2. downloads official negative feature packs used by microWakeWord
3. bootstraps public Polish speech data and extra ambient/noise corpora
4. prepares positive speech features
5. optionally prepares an extra negative `mmap` pack from public background audio
6. trains a custom model
7. exports `.tflite` and `.json` for ESPHome

For long recordings, the pipeline can now optionally segment source audio in two ways:

- `positive_segmentation`: safely splits long positive speech clips only after assigning the original files to `train` / `validation` / `test`, which avoids leakage between splits
- `background_segmentation`: cuts long background recordings into shorter clips before augmentation or generated background-negative packs use them

The public bootstrap does not use entire remote datasets by default. Most presets intentionally cap downloads with `max_clips` or subset knobs such as:

- `bootstrap_speech_datasets[*].max_clips`
- `bootstrap_background_datasets[*].max_clips`
- `asset_subsets.audioset_max_clips`
- `asset_subsets.fma_max_clips`

If you already created `.venv` before a dependency fix landed, refresh the helper packages with:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The training launcher also auto-installs `tensorboard` if an older `.venv` is missing it.

By default the bootstrap tries:

- `amu-cai/pl-asr-bigos-v2` if you have access
- `Common Voice` for `pl` from Mozilla Data Collective
- `google/fleurs` for `pl_pl`
- `facebook/voxpopuli` for `pl`
- `MLS Polish` from OpenSLR
- `bond005/audioset-nonspeech` for extra nonspeech background clips
- `haydarkadioglu/speech-noise-dataset` filtered to `noise_only`
- `philgzl/wham` as an optional higher-quality urban-noise source
- `MUSAN` from OpenSLR filtered to `music` and `noise`
- `Sounds of Home` as an optional residential speech-removed background source
- any local datasets you place in `data/mc_speech` or `data/pl_speech`

Common Voice is no longer downloaded from Hugging Face. The launcher now uses
Mozilla Data Collective directly and expects:

- `MDC_API_KEY` to be set in the environment
- the Common Voice dataset terms to be accepted in the browser first

The dataset id is resolved automatically from the official Common Voice catalog,
so you usually only need the API key.

`AxonData/speech-free-background-noise` is no longer enabled by default in the
Polish VAD presets because some recordings may still contain intelligible speech
or announcements. You can re-enable it manually if you curate/filter those
clips for your own setup.

For the bundled Polish VAD config, the default runtime is CPU because TensorFlow on WSL2 with very new NVIDIA GPUs can be unstable during the `train` step.

The launcher still supports a more flexible runtime and:

- disables XLA auto-JIT
- relaxes strict XLA GPU conv algorithm picking
- retries once on CPU automatically

You can override that in the YAML:

```yaml
runtime:
  device: "cpu"   # auto, gpu, cpu
  allow_cpu_fallback: true
```

## Fetch The Official VAD Anyway

If you want the stock ESPHome model for comparison:

```bash
./train.sh vad-official
```

Or:

```bash
python fetch_voice_pe_vad.py --model vad
```

## Main Files

- `setup_environment.sh`: Linux and WSL2 bootstrap
- `train.sh`: main launcher
- `train_microwakeword.py`: local training pipeline for wake word and VAD
- `fetch_voice_pe_vad.py`: official model downloader
- `configs/microwakeword_example.yaml`: example wake-word config
- `configs/polish_vad.yaml`: example Polish VAD config
- `configs/archive/`: archived experimental presets kept for reference

## Notes

- For wake words, real recordings are usually much better than pure TTS.
- For Polish VAD, local speech datasets are strongly recommended even if the bootstrap fallback is enabled.
- Generated files are written to `export/`.

## Upstream References

- microWakeWord: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)
- official ESPHome models: [esphome/micro-wake-word-models](https://github.com/esphome/micro-wake-word-models)
- Voice PE firmware: [esphome/home-assistant-voice-pe](https://github.com/esphome/home-assistant-voice-pe)
