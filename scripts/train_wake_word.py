#!/usr/bin/env python3
"""
End-to-end wake word training script.

This script demonstrates the complete training pipeline:
1. Load a recorded wake word sample
2. Generate augmented positive examples via TTS and audio augmentation
3. Generate negative examples (silence, noise, similar words)
4. Train the model
5. Calibrate detection threshold
6. Export the model with metadata

Usage:
    uv run python scripts/train_wake_word.py --wake-word "hi alexa" --audio hi-alexa.wav
    uv run python scripts/train_wake_word.py --wake-word "hey jarvis" --audio hey-jarvis.wav --model tc_resnet
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from wakebuilder.audio import (
    AudioPreprocessor,
    DataAugmenter,
    NegativeExampleGenerator,
    load_audio,
)
from wakebuilder.models import (
    Trainer,
    TrainingConfig,
    calibrate_threshold,
    create_model,
    get_model_info,
    print_threshold_report,
)


def print_banner(text: str) -> None:
    """Print a banner with the given text."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def load_wake_word_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load the wake word audio file."""
    print(f"Loading audio from: {audio_path}")
    audio, sr = load_audio(str(audio_path), sample_rate=16000)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s, Sample rate: {sr}Hz, Samples: {len(audio)}")
    return audio, sr


def save_augmented_samples(
    samples: list[tuple[np.ndarray, int]],
    output_dir: Path,
    prefix: str,
    max_save: int = 20,
) -> None:
    """Save a subset of augmented samples for manual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (audio, sr) in enumerate(samples[:max_save]):
        filepath = output_dir / f"{prefix}_{i:03d}.wav"
        sf.write(filepath, audio, sr)
    
    print(f"  Saved {min(len(samples), max_save)} samples to {output_dir}")


def generate_positive_samples(
    audio: np.ndarray,
    sr: int,
    wake_word: str,
    target_samples: int = 500,
    tts_output_dir: Path = None,
) -> list[tuple[np.ndarray, int]]:
    """Generate positive samples from the recorded audio."""
    print_banner("Generating Positive Samples")

    samples = []
    tts_samples = []  # Track TTS samples separately for saving

    # Original recording
    samples.append((audio, sr))
    print("  Original recording: 1 sample")

    # Augmented versions of the recording
    print("  Augmenting original recording...")
    augmenter = DataAugmenter(target_sample_rate=16000)

    augmented_count = 0
    for aug_sample in augmenter.augment_audio(audio, sr):
        samples.append((aug_sample.audio, 16000))
        augmented_count += 1
        # Generate more augmentations to help balance
        if augmented_count >= target_samples // 3:
            break

    print(f"  Augmented from recording: {augmented_count} samples")

    # TTS-generated samples (if available)
    try:
        from wakebuilder.tts import TTSGenerator, list_available_voices

        voices = list_available_voices()
        if voices:
            num_voices = min(len(voices), 8)  # Use more voices
            print(f"  Generating TTS samples with {num_voices} voices...")
            tts = TTSGenerator()

            tts_count = 0
            remaining = target_samples - len(samples)

            for voice in voices[:num_voices]:
                try:
                    # Generate with different length scales (speed variations)
                    for length_scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
                        tts_audio, _ = tts.synthesize(
                            wake_word, voice.name, length_scale=length_scale
                        )
                        if tts_audio is not None and len(tts_audio) > 0:
                            samples.append((tts_audio, 16000))
                            tts_samples.append((tts_audio, 16000, voice.name, length_scale))
                            tts_count += 1

                            # Augment TTS samples more aggressively
                            aug_count = 0
                            for aug in augmenter.augment_audio(tts_audio, 16000):
                                samples.append((aug.audio, 16000))
                                tts_count += 1
                                aug_count += 1
                                if aug_count >= 5:  # More augmentations per TTS
                                    break

                            if tts_count >= remaining:
                                break
                except Exception as e:
                    print(f"    Warning: TTS failed for {voice.name}: {e}")
                    continue

                if tts_count >= remaining:
                    break

            print(f"  TTS-generated: {tts_count} samples")

            # Save raw TTS samples for listening
            if tts_output_dir and tts_samples:
                tts_output_dir.mkdir(parents=True, exist_ok=True)
                for i, (audio_data, sr_val, voice_name, scale) in enumerate(tts_samples[:20]):
                    filename = f"tts_{i:02d}_{voice_name}_scale{scale:.2f}.wav"
                    sf.write(tts_output_dir / filename, audio_data, sr_val)
                print(f"  Saved {min(len(tts_samples), 20)} TTS samples to {tts_output_dir}")
        else:
            print("  No TTS voices available, skipping TTS generation")
    except ImportError:
        print("  TTS not available, skipping TTS generation")

    print(f"\n  Total positive samples: {len(samples)}")
    return samples


def generate_negative_samples(
    wake_word: str,
    target_samples: int = 500,
) -> list[tuple[np.ndarray, int]]:
    """Generate negative samples to match positive count."""
    print_banner("Generating Negative Samples")

    samples = []
    neg_gen = NegativeExampleGenerator(target_sample_rate=16000)

    # Calculate proportions based on target
    silence_count = max(20, target_samples // 10)
    noise_count = max(50, target_samples // 5)
    similar_count = max(100, target_samples // 3)
    random_count = target_samples - silence_count - noise_count - similar_count

    # Silence
    print("  Generating silence samples...")
    for sample in neg_gen.generate_silence(num_samples=silence_count):
        samples.append((sample.audio, 16000))
    print(f"    Silence: {silence_count} samples")

    # Pure noise
    print("  Generating noise samples...")
    for sample in neg_gen.generate_pure_noise(num_samples=noise_count):
        samples.append((sample.audio, 16000))
    print(f"    Noise: {noise_count} samples")

    # TTS-based negatives
    if neg_gen.tts_available:
        # Phonetically similar words
        print("  Generating phonetically similar words...")
        actual_similar = 0
        for sample in neg_gen.generate_phonetically_similar(
            wake_word, num_voices=8, add_noise=True
        ):
            samples.append((sample.audio, 16000))
            actual_similar += 1
            if actual_similar >= similar_count:
                break
        print(f"    Similar words: {actual_similar} samples")

        # Random speech
        print("  Generating random speech...")
        actual_random = 0
        for sample in neg_gen.generate_random_speech(
            num_samples=random_count, num_voices=8, add_noise=True
        ):
            samples.append((sample.audio, 16000))
            actual_random += 1
            if actual_random >= random_count:
                break
        print(f"    Random speech: {actual_random} samples")
    else:
        print("  TTS not available, using only silence and noise")

    print(f"\n  Total negative samples: {len(samples)}")
    return samples


def train_model(
    positive_samples: list[tuple[np.ndarray, int]],
    negative_samples: list[tuple[np.ndarray, int]],
    wake_word: str,
    model_type: str = "tc_resnet",
    num_epochs: int = 50,
    output_dir: Path = None,
) -> tuple:
    """Train the wake word model."""
    print_banner(f"Training {model_type.upper()} Model")

    # Configure training
    config = TrainingConfig(
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=1e-3,
        patience=10,
        label_smoothing=0.1,
        mixup_alpha=0.2,
    )

    print(f"  Model type: {config.model_type}")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")

    # Create trainer
    trainer = Trainer(config=config, output_dir=output_dir)

    # Prepare data (this will do additional augmentation)
    train_loader, val_loader = trainer.prepare_data(
        positive_audio=positive_samples,
        negative_audio=negative_samples,
        wake_word=wake_word,
        augment_positive=False,  # Already augmented
        generate_negatives=False,  # Already generated
    )

    # Create model for info display
    temp_model = trainer.create_model()
    info = get_model_info(temp_model)
    print(f"\n  Model parameters: {info['total_parameters']:,}")
    print(f"  Model size: {info['size_mb']:.2f} MB")
    del temp_model  # Will be recreated in train()

    print("\n" + "-" * 60)
    trained_model = trainer.train(train_loader, val_loader, wake_word)

    return trainer, trained_model, val_loader


def calibrate_and_export(
    trainer: Trainer,
    model: torch.nn.Module,
    val_loader,
    wake_word: str,
    model_type: str = "tc_resnet",
) -> Path:
    """Calibrate threshold and export the model."""
    print_banner("Threshold Calibration")

    device = torch.device(trainer.config.device)

    # Calibrate threshold
    optimal_threshold, metrics = calibrate_threshold(
        model, val_loader, device, num_thresholds=100
    )

    print_threshold_report(optimal_threshold, metrics)

    # Export model with model type in name (e.g., "hi_alexa_tc_resnet")
    print_banner("Exporting Model")

    # Include model type in the wake word name for the export path
    wake_word_with_model = f"{wake_word}_{model_type}"

    model_dir = trainer.save_model(
        wake_word=wake_word_with_model,
        threshold=optimal_threshold,
        metadata={
            "version": "1.0.0",
            "framework": "pytorch",
            "original_wake_word": wake_word,
        },
    )

    # Verify export
    print("\nVerifying exported files:")
    for file in model_dir.iterdir():
        size = file.stat().st_size
        print(f"  {file.name}: {size / 1024:.1f} KB")

    return model_dir


def test_inference(model_dir: Path, test_audio: np.ndarray, sr: int) -> None:
    """Test inference with the exported model."""
    print_banner("Testing Inference")

    import json

    # Load metadata
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  Wake word: {metadata['wake_word']}")
    print(f"  Threshold: {metadata['threshold']:.3f}")
    print(f"  Model type: {metadata['model_type']}")

    # Load model
    checkpoint = torch.load(model_dir / "model.pt", weights_only=True)
    model = create_model(
        model_type=checkpoint["model_type"],
        n_mels=checkpoint["n_mels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Process test audio
    preprocessor = AudioPreprocessor(n_mels=checkpoint["n_mels"])
    spec = preprocessor.process_audio(test_audio, sr)
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()

    # Run inference
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(spec_tensor)
        inference_time = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1)
    wake_word_prob = probs[0, 1].item()

    print(f"\n  Inference time: {inference_time:.2f} ms")
    print(f"  Wake word probability: {wake_word_prob:.3f}")
    print(f"  Threshold: {metadata['threshold']:.3f}")
    detected = wake_word_prob >= metadata["threshold"]
    print(f"  Detection: {'YES (DETECTED)' if detected else 'NO (not detected)'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a wake word detection model end-to-end"
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default="hi alexa",
        help="The wake word to train (default: 'hi alexa')",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="hi-alexa.wav",
        help="Path to recorded wake word audio file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tc_resnet", "bc_resnet"],
        default="tc_resnet",
        help="Model architecture (default: tc_resnet)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer samples and epochs for testing",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = project_root / audio_path

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models"

    # Check audio file exists
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Create augmented data directory
    wake_word_slug = args.wake_word.lower().replace(" ", "_")
    augmented_dir = project_root / "data" / "augmented" / wake_word_slug

    # TTS samples directory for listening
    tts_dir = augmented_dir / "tts_raw"

    print_banner("WakeBuilder Training Pipeline")
    print(f"  Wake word: '{args.wake_word}'")
    print(f"  Audio file: {audio_path}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {output_dir}")
    print(f"  Augmented data: {augmented_dir}")
    print(f"  TTS samples: {tts_dir}")

    # Quick mode for testing (fewer samples, fewer epochs)
    # For 95%+ accuracy, use full mode with more data and epochs
    # BALANCED: same number of positive and negative samples
    target_samples = 100 if args.quick else 500
    epochs = 5 if args.quick else args.epochs

    start_time = time.time()

    # Step 1: Load audio
    audio, sr = load_wake_word_audio(audio_path)

    # Step 2: Generate positive samples (with TTS saved for listening)
    positive_samples = generate_positive_samples(
        audio, sr, args.wake_word,
        target_samples=target_samples,
        tts_output_dir=tts_dir,
    )

    # Step 3: Generate negative samples (BALANCED with positive)
    # Use actual positive count to ensure balance
    negative_samples = generate_negative_samples(
        args.wake_word, target_samples=len(positive_samples)
    )

    print_banner("Data Balance Check")
    print(f"  Positive samples: {len(positive_samples)}")
    print(f"  Negative samples: {len(negative_samples)}")
    ratio = len(positive_samples) / max(len(negative_samples), 1)
    print(f"  Balance ratio: {ratio:.2f} (ideal: 1.0)")

    # Step 4: Save augmented samples for manual inspection
    print_banner("Saving Augmented Samples for Inspection")
    save_augmented_samples(
        positive_samples, augmented_dir / "positive", "positive", max_save=30
    )
    save_augmented_samples(
        negative_samples, augmented_dir / "negative", "negative", max_save=30
    )

    # Step 5: Train model
    trainer, model, val_loader = train_model(
        positive_samples,
        negative_samples,
        args.wake_word,
        model_type=args.model,
        num_epochs=epochs,
        output_dir=output_dir,
    )

    # Step 6: Calibrate and export (include model type in path)
    model_dir = calibrate_and_export(
        trainer, model, val_loader, args.wake_word, args.model
    )

    # Step 7: Test inference
    test_inference(model_dir, audio, sr)

    # Summary
    total_time = time.time() - start_time
    print_banner("Training Complete!")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Model saved to: {model_dir}")
    print("\n  To use this model:")
    print("    from wakebuilder.models import create_model")
    print(f"    model = create_model('{args.model}')")
    print(f"    # Load weights from {model_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
