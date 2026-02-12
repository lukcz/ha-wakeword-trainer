#!/usr/bin/env python3
"""
=============================================================================
🧦 WAKE WORD TRAINER - All-in-one microWakeWord Training Script
=============================================================================

Trains a custom wake word model for Home Assistant Voice PE / ESPHome.
Just run it and be patient - everything is automatic!

USAGE:
    python train_wakeword.py "Hej Zgredek"
    python train_wakeword.py "Hey Computer" --lang en
    python train_wakeword.py "Hej Zgredek" --variations "Hej Zgredku,hej zgredek"

REQUIREMENTS (WSL2 Ubuntu recommended):
    1. Install dependencies:
       pip install edge-tts datasets==2.14.0 soundfile librosa pyyaml requests tqdm
       pip install 'numpy<2' 'pyarrow>=12,<15' tensorflow==2.16.1
       sudo apt install ffmpeg
    
    2. Clone and install micro-wake-word:
       git clone https://github.com/OHF-Voice/micro-wake-word.git
       cd micro-wake-word && pip install -e .

OUTPUTS:
    ./wakeword_training/
    ├── your_wake_word.tflite   <- Model file
    ├── your_wake_word.json     <- Manifest for ESPHome  
    └── ...training data...

TIME ESTIMATE:
    - First run (downloads all data): 1-2 hours
    - Subsequent runs (data cached): 20-40 minutes
    - Training only: 10-20 minutes (CPU)

Author: Zgredek 🧦
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Check for required modules early
MISSING_MODULES = []
for module in ['edge_tts', 'yaml', 'requests', 'tqdm', 'numpy', 'scipy']:
    try:
        __import__(module)
    except ImportError:
        MISSING_MODULES.append(module)

if MISSING_MODULES:
    print("❌ Missing required modules:", ", ".join(MISSING_MODULES))
    print("\nInstall with:")
    print("  pip install edge-tts pyyaml requests tqdm numpy scipy")
    sys.exit(1)

import numpy as np
import requests
import scipy.io.wavfile
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="🧦 Train a custom wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python train_wakeword.py "Hej Zgredek"
  python train_wakeword.py "Hey Jarvis" --lang en --steps 15000
  python train_wakeword.py "Hej Zgredek" --variations "Hej Zgredku,hej zgredek"
  python train_wakeword.py "Hej Zgredek" --data_dir /mnt/d/zgredek_training
        """
    )
    parser.add_argument("wake_word", help="Wake word phrase (e.g., 'Hej Zgredek')")
    parser.add_argument("--variations", default="", 
                        help="Comma-separated phrase variations")
    parser.add_argument("--lang", default="pl", choices=["pl", "en", "de", "es", "fr"],
                        help="Language for TTS voices (default: pl)")
    parser.add_argument("--output_dir", default="./wakeword_training", 
                        help="Output directory")
    parser.add_argument("--data_dir", default=None,
                        help="Use existing augmentation data from this directory")
    parser.add_argument("--steps", type=int, default=10000, 
                        help="Training steps (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size (default: 128)")
    parser.add_argument("--author", default="custom", 
                        help="Author name for manifest")
    parser.add_argument("--probability_cutoff", type=float, default=0.7,
                        help="Detection threshold 0.0-1.0 (default: 0.7)")
    parser.add_argument("--skip_generate", action="store_true", 
                        help="Skip sample generation (use existing)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading datasets (use existing)")
    parser.add_argument("--cpu_only", action="store_true", default=True,
                        help="Force CPU training (default: True, GPU often has issues)")
    return parser.parse_args()


# =============================================================================
# TTS VOICE CONFIGURATIONS
# =============================================================================
VOICES = {
    "pl": ["pl-PL-MarekNeural", "pl-PL-ZofiaNeural"],
    "en": ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural"],
    "de": ["de-DE-ConradNeural", "de-DE-KatjaNeural"],
    "es": ["es-ES-AlvaroNeural", "es-ES-ElviraNeural"],
    "fr": ["fr-FR-HenriNeural", "fr-FR-DeniseNeural"],
}

RATES = ["-20%", "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%", "+20%"]
PITCHES = ["-100Hz", "-50Hz", "-25Hz", "+0Hz", "+25Hz", "+50Hz", "+100Hz"]


# =============================================================================
# SAMPLE GENERATION
# =============================================================================
async def generate_samples(phrases, output_dir, lang="pl"):
    """Generate TTS samples using Edge TTS"""
    import edge_tts
    
    os.makedirs(output_dir, exist_ok=True)
    voices = VOICES.get(lang, VOICES["en"])
    
    sem = asyncio.Semaphore(30)  # Concurrent requests
    generated = [0]
    failed = [0]
    
    async def generate_one(i, phrase, voice, rate, pitch):
        async with sem:
            filename = f"{output_dir}/sample_{i:05d}.mp3"
            try:
                communicate = edge_tts.Communicate(phrase, voice, rate=rate, pitch=pitch)
                await communicate.save(filename)
                generated[0] += 1
                return True
            except Exception:
                failed[0] += 1
                return False
    
    tasks = []
    i = 0
    for phrase in phrases:
        for voice in voices:
            for rate in RATES:
                for pitch in PITCHES:
                    tasks.append(generate_one(i, phrase, voice, rate, pitch))
                    i += 1
    
    total = len(tasks)
    print(f"  Generating {total} samples with {len(voices)} voices...")
    print(f"  (This may take a few minutes)")
    
    # Process in batches with progress
    batch_size = 100
    for batch_start in tqdm(range(0, len(tasks), batch_size), desc="  Progress"):
        batch = tasks[batch_start:batch_start + batch_size]
        await asyncio.gather(*batch)
    
    print(f"  ✓ Generated: {generated[0]}, Failed: {failed[0]}")
    return generated[0]


def convert_to_wav(input_dir, output_dir):
    """Convert MP3 samples to 16kHz mono WAV"""
    os.makedirs(output_dir, exist_ok=True)
    
    mp3_files = list(Path(input_dir).glob("*.mp3"))
    print(f"  Converting {len(mp3_files)} files to 16kHz WAV...")
    
    for mp3_file in tqdm(mp3_files, desc="  Progress"):
        wav_file = Path(output_dir) / mp3_file.with_suffix(".wav").name
        result = subprocess.run([
            "ffmpeg", "-i", str(mp3_file),
            "-ar", "16000", "-ac", "1",
            str(wav_file), "-y", "-loglevel", "error"
        ], capture_output=True)
    
    wav_count = len(list(Path(output_dir).glob("*.wav")))
    print(f"  ✓ Converted {wav_count} files")
    return wav_count


# =============================================================================
# DATASET DOWNLOADS
# =============================================================================
def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc=desc or "Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_negative_datasets(output_dir):
    """Download pre-generated negative datasets from HuggingFace"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    datasets = ['speech.zip', 'dinner_party.zip', 'no_speech.zip', 'dinner_party_eval.zip']
    
    for dataset in datasets:
        extract_name = dataset.replace('.zip', '')
        extract_path = Path(output_dir) / extract_name
        
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"    ✓ {extract_name} (exists)")
            continue
        
        zip_path = Path(output_dir) / dataset
        print(f"    ↓ {dataset}")
        download_file(base_url + dataset, str(zip_path), f"    {dataset}")
        
        print(f"    Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        zip_path.unlink()
        print(f"    ✓ {extract_name}")


def download_mit_rirs(output_dir):
    """Download MIT Room Impulse Responses"""
    rir_dir = Path(output_dir)
    if rir_dir.exists() and len(list(rir_dir.glob("*.wav"))) > 100:
        print(f"    ✓ MIT RIRs (exists)")
        return
    
    rir_dir.mkdir(parents=True, exist_ok=True)
    print(f"    ↓ MIT Room Impulse Responses...")
    
    try:
        import datasets as hf_datasets
        rir_dataset = hf_datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=True
        )
        
        count = 0
        for row in tqdm(rir_dataset, desc="    Downloading RIRs"):
            name = row['audio']['path'].split('/')[-1]
            scipy.io.wavfile.write(
                str(rir_dir / name), 16000,
                (row['audio']['array'] * 32767).astype(np.int16)
            )
            count += 1
        print(f"    ✓ Downloaded {count} RIR files")
    except Exception as e:
        print(f"    ⚠ RIR download failed: {e}")
        print(f"    Model will train without room impulses (still works)")


def download_fma(output_dir):
    """Download FMA (Free Music Archive) background music"""
    fma_dir = Path(output_dir)
    if fma_dir.exists() and len(list(fma_dir.glob("*.wav"))) > 100:
        print(f"    ✓ FMA music (exists)")
        return
    
    fma_raw = fma_dir.parent / "fma_raw"
    fma_raw.mkdir(parents=True, exist_ok=True)
    fma_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = fma_raw / "fma_xs.zip"
    if not zip_path.exists():
        print(f"    ↓ FMA (Free Music Archive)...")
        url = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
        download_file(url, str(zip_path), "    FMA dataset")
    
    # Extract
    print(f"    Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(fma_raw)
    
    # Convert to 16kHz
    print(f"    Converting to 16kHz...")
    mp3_files = list((fma_raw / "fma_small").rglob("*.mp3"))[:500]  # Limit to 500 files
    
    for mp3_file in tqdm(mp3_files, desc="    Converting"):
        wav_file = fma_dir / f"{mp3_file.stem}.wav"
        subprocess.run([
            "ffmpeg", "-i", str(mp3_file),
            "-ar", "16000", "-ac", "1", "-t", "30",  # Max 30 seconds
            str(wav_file), "-y", "-loglevel", "error"
        ], capture_output=True)
    
    print(f"    ✓ FMA ready ({len(list(fma_dir.glob('*.wav')))} files)")


def download_audioset(output_dir):
    """Download AudioSet background sounds"""
    audioset_dir = Path(output_dir)
    if audioset_dir.exists() and len(list(audioset_dir.glob("*.wav"))) > 100:
        print(f"    ✓ AudioSet (exists)")
        return
    
    audioset_raw = audioset_dir.parent / "audioset_raw"
    audioset_raw.mkdir(parents=True, exist_ok=True)
    audioset_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = audioset_raw / "bal_train09.tar"
    
    try:
        import tarfile
        
        if not tar_path.exists() or tar_path.stat().st_size < 1000000:  # < 1MB = probably failed
            if tar_path.exists():
                tar_path.unlink()  # Remove corrupted file
            print(f"    ↓ AudioSet (background sounds)...")
            url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar"
            download_file(url, str(tar_path), "    AudioSet")
        
        # Verify file is valid tar
        if not tarfile.is_tarfile(str(tar_path)):
            print(f"    ⚠ AudioSet download corrupted, skipping (training will still work)")
            if tar_path.exists():
                tar_path.unlink()
            return
        
        # Extract
        print(f"    Extracting...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(audioset_raw)
        
        # Convert to 16kHz
        print(f"    Converting to 16kHz...")
        flac_files = list((audioset_raw / "audio").glob("*.flac"))
        
        for flac_file in tqdm(flac_files, desc="    Converting"):
            wav_file = audioset_dir / f"{flac_file.stem}.wav"
            subprocess.run([
                "ffmpeg", "-i", str(flac_file),
                "-ar", "16000", "-ac", "1",
                str(wav_file), "-y", "-loglevel", "error"
            ], capture_output=True)
        
        print(f"    ✓ AudioSet ready ({len(list(audioset_dir.glob('*.wav')))} files)")
    
    except Exception as e:
        print(f"    ⚠ AudioSet failed: {e}")
        print(f"    Skipping AudioSet (training will still work without it)")


# =============================================================================
# FEATURE GENERATION
# =============================================================================
def generate_augmented_features(config):
    """Generate augmented spectrogram features"""
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap
    
    output_dir = config['augmented_features_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup background paths (only include existing directories)
    background_paths = []
    if config.get('fma_dir') and Path(config['fma_dir']).exists():
        background_paths.append(str(config['fma_dir']))
    if config.get('audioset_dir') and Path(config['audioset_dir']).exists():
        background_paths.append(str(config['audioset_dir']))
    
    impulse_paths = []
    if config.get('rir_dir') and Path(config['rir_dir']).exists():
        impulse_paths.append(str(config['rir_dir']))
    
    clips = Clips(
        input_directory=str(config['samples_16k_dir']),
        file_pattern='*.wav',
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75 if background_paths else 0.0,
            "Gain": 1.0,
            "RIR": 0.5 if impulse_paths else 0.0,
        },
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )
    
    splits = [
        ("training", "train", 2, 10),
        ("validation", "validation", 1, 1),
        ("testing", "test", 1, 1),
    ]
    
    for split_dir, split_name, repetition, slide_frames in splits:
        out_path = Path(output_dir) / split_dir
        mmap_path = out_path / 'wakeword_mmap'
        
        if mmap_path.exists():
            print(f"    ✓ {split_dir} (exists)")
            continue
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )
        
        print(f"    Generating {split_dir}...")
        RaggedMmap.from_generator(
            out_dir=str(mmap_path),
            sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
            batch_size=100,
            verbose=True,
        )


# =============================================================================
# TRAINING
# =============================================================================
def create_training_config(config):
    """Create training_parameters.yaml"""
    training_config = {
        "window_step_ms": 10,
        "train_dir": str(config['model_dir']),
        "features": [
            {
                "features_dir": str(config['augmented_features_dir']),
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(Path(config['negative_datasets_dir']) / "speech"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(Path(config['negative_datasets_dir']) / "dinner_party"),
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(Path(config['negative_datasets_dir']) / "no_speech"),
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(Path(config['negative_datasets_dir']) / "dinner_party_eval"),
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [config['steps']],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": config['batch_size'],
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    
    config_path = config['training_config_path']
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f)
    
    return config_path


def patch_train_py():
    """Patch train.py to fix numpy compatibility issue"""
    try:
        import microwakeword
        train_py = Path(microwakeword.__file__).parent / "train.py"
        
        content = train_py.read_text()
        if '.numpy()' in content:
            patched = content.replace('.numpy()', '')
            train_py.write_text(patched)
            print("  ✓ Patched train.py for numpy compatibility")
    except Exception as e:
        print(f"  ⚠ Could not patch train.py: {e}")


def run_training(config_path, cpu_only=True):
    """Run the training process"""
    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "1",
        "--restore_checkpoint", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]
    
    env = os.environ.copy()
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    
    print("\n  Training started! This will take 10-30 minutes...")
    print("  Progress is shown every 500 steps.\n")
    
    subprocess.run(cmd, env=env, check=True)


def run_conversion(config_path, cpu_only=True):
    """Run model conversion if tflite wasn't generated during training"""
    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train", "0",  # Skip training, only convert
        "--restore_checkpoint", "1",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "3",
    ]
    
    env = os.environ.copy()
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    
    print("\n  Converting model to TFLite format...")
    
    subprocess.run(cmd, env=env, check=True)


def create_manifest(wake_word, author, probability_cutoff, output_path, tflite_filename=None):
    """Create the JSON manifest for ESPHome"""
    manifest = {
        "version": 2,
        "wake_word": wake_word.lower(),
        "author": author,
        "micro": {
            "probability_cutoff": probability_cutoff,
            "sliding_window_size": 5
        },
        "suggested_cutoffs": {
            "conservative": {
                "probability_cutoff": min(0.90, probability_cutoff + 0.15),
                "description": "Fewer false activations, may miss some wake words"
            },
            "balanced": {
                "probability_cutoff": probability_cutoff,
                "description": "Good balance between detection and false positives"
            },
            "sensitive": {
                "probability_cutoff": max(0.50, probability_cutoff - 0.15),
                "description": "Catches more wake words, may have more false activations"
            }
        }
    }
    
    # Add model URL placeholder (user should update this after uploading)
    if tflite_filename:
        manifest["model"] = f"UPDATE_WITH_YOUR_GITHUB_URL/{tflite_filename}"
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing data directory or create new
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        print(f"📁 Using existing data from: {data_dir}")
    else:
        data_dir = output_dir
    
    # Create wake word slug
    wake_word_slug = args.wake_word.lower().replace(" ", "_").replace(",", "")
    
    config = {
        'wake_word': args.wake_word,
        'lang': args.lang,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'samples_dir': output_dir / "generated_samples",
        'samples_16k_dir': output_dir / "generated_samples_16k",
        'augmented_features_dir': output_dir / "augmented_features",
        'negative_datasets_dir': data_dir / "negative_datasets",
        'rir_dir': data_dir / "mit_rirs",
        'fma_dir': data_dir / "fma_16k",
        'audioset_dir': data_dir / "audioset_16k",
        'model_dir': output_dir / "trained_model",
        'training_config_path': output_dir / "training_parameters.yaml",
    }
    
    # Build phrase list with variations
    phrases = [args.wake_word]
    if args.variations:
        phrases.extend([v.strip() for v in args.variations.split(",")])
    
    # Add automatic variations
    base_phrases = phrases.copy()
    for phrase in base_phrases:
        variations = [
            phrase + "!",
            phrase + "?",
            phrase.lower(),
            "".join(c.upper() if i == 0 else c.lower() for i, c in enumerate(phrase)),
        ]
        phrases.extend(variations)
    phrases = list(set(phrases))
    
    # Print banner
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧦 WAKE WORD TRAINER                                        ║
╠══════════════════════════════════════════════════════════════╣
║  Wake Word:    {args.wake_word:<45} ║
║  Language:     {args.lang:<45} ║
║  Variations:   {len(phrases):<45} ║
║  Steps:        {args.steps:<45} ║
║  Output:       {str(output_dir)[:45]:<45} ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # ===================
    # STEP 1: Generate samples
    # ===================
    if not args.skip_generate:
        print("━" * 60)
        print("📢 STEP 1/6: Generating TTS samples")
        print("━" * 60)
        asyncio.run(generate_samples(phrases, str(config['samples_dir']), args.lang))
        
        print("\n━" * 60)
        print("🔊 STEP 2/6: Converting to WAV format")
        print("━" * 60)
        convert_to_wav(str(config['samples_dir']), str(config['samples_16k_dir']))
    else:
        print("⏭️  Skipping sample generation (--skip_generate)")
    
    # ===================
    # STEP 2: Download datasets
    # ===================
    if not args.skip_download:
        print("\n━" * 60)
        print("📥 STEP 3/6: Downloading training datasets")
        print("━" * 60)
        print("  (This is a one-time download, ~5GB total)\n")
        
        print("  [Negative datasets]")
        download_negative_datasets(str(config['negative_datasets_dir']))
        
        print("\n  [Augmentation data]")
        download_mit_rirs(str(config['rir_dir']))
        download_fma(str(config['fma_dir']))
        download_audioset(str(config['audioset_dir']))
    else:
        print("⏭️  Skipping dataset downloads (--skip_download)")
    
    # ===================
    # STEP 3: Generate features
    # ===================
    print("\n━" * 60)
    print("🎛️  STEP 4/6: Generating augmented features")
    print("━" * 60)
    generate_augmented_features(config)
    
    # ===================
    # STEP 4: Prepare training
    # ===================
    print("\n━" * 60)
    print("⚙️  STEP 5/6: Preparing training")
    print("━" * 60)
    patch_train_py()
    create_training_config(config)
    
    if config['model_dir'].exists():
        shutil.rmtree(config['model_dir'])
        print("  ✓ Cleaned previous model directory")
    
    # ===================
    # STEP 5: Train
    # ===================
    print("\n━" * 60)
    print("🚀 STEP 6/6: Training model")
    print("━" * 60)
    run_training(str(config['training_config_path']), cpu_only=args.cpu_only)
    
    # ===================
    # STEP 6: Export
    # ===================
    print("\n━" * 60)
    print("📦 Exporting model")
    print("━" * 60)
    
    tflite_src = config['model_dir'] / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    tflite_dst = output_dir / f"{wake_word_slug}.tflite"
    manifest_dst = output_dir / f"{wake_word_slug}.json"
    
    # If tflite wasn't generated during training, run conversion separately
    if not tflite_src.exists():
        print("  ⚠ TFLite not generated during training, running conversion...")
        run_conversion(str(config['training_config_path']), cpu_only=args.cpu_only)
    
    if tflite_src.exists():
        shutil.copy(tflite_src, tflite_dst)
        create_manifest(args.wake_word, args.author, args.probability_cutoff, str(manifest_dst), tflite_dst.name)
        
        model_size = tflite_dst.stat().st_size / 1024
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ SUCCESS!                                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Model:      {tflite_dst.name:<46} ║
║  Manifest:   {manifest_dst.name:<46} ║
║  Size:       {model_size:.1f} KB{' ' * 42}║
╠══════════════════════════════════════════════════════════════╣
║  📁 Files saved to: {str(output_dir)[:40]:<40} ║
╚══════════════════════════════════════════════════════════════╝

📊 INTERPRETING TRAINING RESULTS:

Look at the training output above for lines like:
  INFO:absl:Cutoff 0.75: frr=0.0691; faph=0.750

  • frr = False Rejection Rate (% of wake words missed)
  • faph = False Accepts Per Hour (false triggers per hour)

CHOOSING YOUR CUTOFF:
┌──────────────────────────────────────────────────────────────┐
│ Environment        │ Target faph │ Typical Cutoff │
├────────────────────┼─────────────┼────────────────┤
│ Noisy (kitchen)    │ 1.0-2.0     │ 0.60-0.70      │
│ Quiet (bedroom)    │ 0.0-0.5     │ 0.80-0.90      │
│ Mixed/General      │ 0.5-1.0     │ 0.70-0.80      │
└──────────────────────────────────────────────────────────────┘

🚀 NEXT STEPS:

1. Upload both files to GitHub releases

2. Update the JSON manifest with your GitHub URL:
   Edit {manifest_dst.name} and replace UPDATE_WITH_YOUR_GITHUB_URL
   with your actual release URL

3. Add to ESPHome Voice PE config:

   micro_wake_word:
     models:
       - model: https://github.com/YOUR_USER/YOUR_REPO/releases/download/v1.0.0/{manifest_dst.name}
         probability_cutoff: {args.probability_cutoff}
         sliding_window_size: 5

4. Flash your device and test!

   Tip: Adjust probability_cutoff based on the training metrics above.
        Start with a balanced cutoff, then tune based on real-world use.

🧦 Happy wake-wording!
""")
    else:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ❌ ERROR: Model file not found                               ║
╠══════════════════════════════════════════════════════════════╣
║  Expected: {str(tflite_src)[:50]:<50} ║
║                                                              ║
║  Check the training logs above for errors.                   ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
