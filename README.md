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

The VAD flow:

1. downloads augmentation assets
2. downloads official negative feature packs used by microWakeWord
3. bootstraps positive speech data for Polish
4. prepares positive speech features
5. trains a custom model
6. exports `.tflite` and `.json` for ESPHome

By default the bootstrap tries:

- `amu-cai/pl-asr-bigos-v2` if you have access
- `google/fleurs` for `pl_pl`
- any local datasets you place in `data/mc_speech` or `data/pl_speech`

So the repository can still start without a fully manual dataset setup, but real recordings remain better.

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

## Notes

- For wake words, real recordings are usually much better than pure TTS.
- For Polish VAD, local speech datasets are strongly recommended even if the bootstrap fallback is enabled.
- Generated files are written to `export/`.

## Upstream References

- microWakeWord: [OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)
- official ESPHome models: [esphome/micro-wake-word-models](https://github.com/esphome/micro-wake-word-models)
- Voice PE firmware: [esphome/home-assistant-voice-pe](https://github.com/esphome/home-assistant-voice-pe)
