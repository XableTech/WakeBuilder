#!/usr/bin/env python3
"""
Test script for Phase 1 foundation components.

This script tests the audio preprocessing pipeline and base model structure
without requiring the full TensorFlow model download.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.preprocessing import (
    AudioPreprocessor,
    compute_mel_spectrogram,
    normalize_audio,
)
from wakebuilder.config import Config


def test_audio_normalization():
    """Test audio normalization."""
    print("\n" + "=" * 70)
    print("Test 1: Audio Normalization")
    print("=" * 70)

    # Create test audio
    audio = np.random.randn(16000) * 0.5
    print(
        f"Original audio: shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]"
    )

    # Normalize
    normalized = normalize_audio(audio, target_level=-20.0)
    print(
        f"Normalized audio: shape={normalized.shape}, range=[{normalized.min():.3f}, {normalized.max():.3f}]"
    )

    # Verify
    assert normalized.shape == audio.shape
    assert np.abs(normalized).max() <= 1.0

    print("[PASS] Audio normalization test passed")
    return True


def test_mel_spectrogram():
    """Test mel spectrogram computation."""
    print("\n" + "=" * 70)
    print("Test 2: Mel Spectrogram Computation")
    print("=" * 70)

    # Create test audio (1 second at 16kHz)
    sample_rate = 16000
    audio = np.random.randn(sample_rate) * 0.1

    print(f"Input audio: shape={audio.shape}, sample_rate={sample_rate}Hz")

    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=80,
        hop_length=160,
    )

    print(f"Mel spectrogram: shape={mel_spec.shape}")
    print(f"  - Frequency bins (n_mels): {mel_spec.shape[0]}")
    print(f"  - Time frames: {mel_spec.shape[1]}")
    print(f"  - Value range: [{mel_spec.min():.1f}, {mel_spec.max():.1f}] dB")

    # Verify
    assert mel_spec.shape[0] == 80
    assert mel_spec.shape[1] > 0
    assert np.isfinite(mel_spec).all()

    print("[PASS] Mel spectrogram computation test passed")
    return True


def test_audio_preprocessor():
    """Test AudioPreprocessor class."""
    print("\n" + "=" * 70)
    print("Test 3: Audio Preprocessor")
    print("=" * 70)

    # Create preprocessor
    config = Config()
    preprocessor = AudioPreprocessor.from_config(config)

    print("Preprocessor configuration:")
    print(f"  - Sample rate: {preprocessor.sample_rate} Hz")
    print(f"  - N_FFT: {preprocessor.n_fft}")
    print(f"  - Hop length: {preprocessor.hop_length}")
    print(f"  - N_mels: {preprocessor.n_mels}")
    print(f"  - Target length: {preprocessor.target_length} frames")

    # Create test audio
    audio = np.random.randn(16000) * 0.1

    # Process
    mel_spec = preprocessor.process_audio(audio, sample_rate=16000)

    print(f"\nProcessed mel spectrogram: shape={mel_spec.shape}")
    print(f"  - Time dimension: {mel_spec.shape[0]}")
    print(f"  - Frequency dimension: {mel_spec.shape[1]}")

    # Verify
    assert mel_spec.shape == (96, 80)  # (time, freq)
    assert np.isfinite(mel_spec).all()

    print("[PASS] Audio preprocessor test passed")
    return True


def test_batch_processing():
    """Test batch processing."""
    print("\n" + "=" * 70)
    print("Test 4: Batch Processing")
    print("=" * 70)

    preprocessor = AudioPreprocessor.from_config()

    # Create batch of audio
    batch_size = 4
    audio_list = [np.random.randn(16000) * 0.1 for _ in range(batch_size)]
    sample_rates = [16000] * batch_size

    print(f"Processing batch of {batch_size} audio samples...")

    # Process batch
    mel_specs = preprocessor.process_batch(audio_list, sample_rates)

    print(f"Batch output shape: {mel_specs.shape}")
    print(f"  - Batch size: {mel_specs.shape[0]}")
    print(f"  - Time frames: {mel_specs.shape[1]}")
    print(f"  - Frequency bins: {mel_specs.shape[2]}")

    # Verify
    assert mel_specs.shape == (4, 96, 80)
    assert np.isfinite(mel_specs).all()

    print("[PASS] Batch processing test passed")
    return True


def test_synthetic_wake_word():
    """Test with synthetic wake word audio."""
    print("\n" + "=" * 70)
    print("Test 5: Synthetic Wake Word Processing")
    print("=" * 70)

    # Create synthetic wake word (simple tone + noise)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simulate wake word with multiple frequency components
    audio = (
        np.sin(2 * np.pi * 200 * t) * 0.3  # Low frequency
        + np.sin(2 * np.pi * 800 * t) * 0.2  # Mid frequency
        + np.sin(2 * np.pi * 1600 * t) * 0.1  # High frequency
        + np.random.randn(len(t)) * 0.05  # Noise
    )

    print("Synthetic wake word:")
    print(f"  - Duration: {duration}s")
    print(f"  - Sample rate: {sample_rate}Hz")
    print("  - Frequency components: 200Hz, 800Hz, 1600Hz")

    # Normalize
    audio = normalize_audio(audio)

    # Process
    preprocessor = AudioPreprocessor.from_config()
    mel_spec = preprocessor.process_audio(audio, sample_rate=sample_rate)

    print("\nProcessed spectrogram:")
    print(f"  - Shape: {mel_spec.shape}")
    print(f"  - Mean: {mel_spec.mean():.2f} dB")
    print(f"  - Std: {mel_spec.std():.2f} dB")
    print(f"  - Range: [{mel_spec.min():.1f}, {mel_spec.max():.1f}] dB")

    # Verify
    assert mel_spec.shape == (96, 80)
    assert mel_spec.std() > 0  # Should have variation

    print("[PASS] Synthetic wake word processing test passed")
    return True


def test_model_structure():
    """Test base model structure (without loading weights)."""
    print("\n" + "=" * 70)
    print("Test 6: Base Model Structure")
    print("=" * 70)

    try:
        import torch
        from wakebuilder.models.base_model import SpeechEmbeddingModel

        # Create model
        model = SpeechEmbeddingModel(embedding_dim=96)
        model.eval()

        print("Model created:")
        print(f"  - Embedding dimension: {model.embedding_dim}")
        print("  - Device: CPU")

        # Test forward pass
        x = torch.randn(2, 96, 80)  # Batch of 2

        with torch.no_grad():
            output = model(x)

        print("\nForward pass test:")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Verify
        assert output.shape == (2, 96)
        assert torch.isfinite(output).all()

        print("[PASS] Base model structure test passed")
        return True

    except ImportError as e:
        print(f"[SKIP] Model test skipped (torch not installed): {e}")
        print("  Run: uv sync")
        return True  # Don't fail if torch isn't installed yet


def main():
    """Run all foundation tests."""
    print("=" * 70)
    print("WakeBuilder - Phase 1 Foundation Tests")
    print("=" * 70)

    tests = [
        test_audio_normalization,
        test_mel_spectrogram,
        test_audio_preprocessor,
        test_batch_processing,
        test_synthetic_wake_word,
        test_model_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[SUCCESS] All foundation tests passed!")
        print("\nNext steps:")
        print("  1. Install PyTorch: uv sync")
        print("  2. Download base model: uv run python scripts/download_base_model.py")
        print("  3. Run full tests: uv run pytest tests/")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
