# Training Notes & Lessons Learned

Technical notes from building and debugging the openWakeWord training pipeline. Useful if you're troubleshooting issues or want to understand why certain design decisions were made.

## torchaudio 2.10+ Breaking Changes

torchaudio 2.10 (shipped with PyTorch 2.10) removed several APIs that openWakeWord, speechbrain, and torch-audiomentations depend on:

- `torchaudio.load()` — replaced with backend-agnostic API
- `torchaudio.info()` — removed entirely
- `torchaudio.list_audio_backends()` — removed

Our `compat.py` patches these with soundfile-based replacements. The `torchaudio.load` patch also handles automatic resampling from 22050 Hz (Piper TTS output) to 16000 Hz (what openWakeWord expects).

### Why not globally patch `soundfile.read`?

We tried patching `soundfile.read` globally to auto-resample, but this broke torchaudio's internal `_soundfile_load` function which passes extra kwargs (`start`, `stop`, `always_2d`) that don't survive wrapping. The fix was to only patch `torchaudio.load` and let it call `soundfile.read` directly.

## Sample Rate Mismatch

Piper TTS generates audio at 22050 Hz, but openWakeWord expects 16000 Hz throughout. We handle this with on-the-fly resampling in the patched `torchaudio.load` using `scipy.signal.resample`.

We initially tried bulk-resampling all 110k+ WAV files, but this was extremely slow (~75 minutes) on WSL2's `/mnt/c/` filesystem due to the 9P protocol overhead. The on-the-fly approach handles it transparently with no I/O penalty during augmentation.

## ONNX Export

PyTorch 2.x's `torch.onnx.export` now requires `onnxscript` (not installed by default). The export may also attempt a TFLite conversion via `onnx_tf` which isn't needed — the ONNX model is the final output.

The export produces two files:
- `model_name.onnx` — the model graph (~14 KB)
- `model_name.onnx.data` — external weights (~200 KB)

Both files must be kept together for the model to load.

## WSL2 Filesystem Considerations

- **Venvs**: Must be created on WSL2's native filesystem (`~/.oww-trainer-venv`), not on `/mnt/c/`. Symlinks don't work across the 9P boundary.
- **Training data**: Works fine on `/mnt/c/` for reads, but bulk writes are slow. The pipeline handles this by minimizing write operations.
- **setuptools**: Pin to `<82` to keep `pkg_resources` available (required by several dependencies).

## Model Architecture

The default config uses:
- **DNN** (not RNN) — simpler, faster inference
- **layer_size: 32** — minimal CPU footprint, good enough for single-phrase detection
- **50k training steps** — typically converges well for simple phrases

For multi-word or phonetically complex phrases, consider `layer_size: 64` or `layer_size: 128`.

## Augmentation Strategy

The pipeline uses three types of augmentation:
1. **Room Impulse Responses (RIR)** — MIT environmental recordings simulate different room acoustics
2. **Background noise** — AudioSet clips add real-world ambient noise
3. **Background music** — FMA clips add music interference

If HuggingFace dataset downloads fail (rate limits, etc.), the pipeline generates synthetic white noise as a fallback. This works but produces a less robust model.
