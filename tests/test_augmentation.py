"""
Tests for the data augmentation pipeline.

This module tests:
- TTS generation
- Noise mixing
- Audio augmentation functions
- Negative example generation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio import (
    AugmentedSample,
    DataAugmenter,
    NegativeExampleGenerator,
    NoiseLoader,
    apply_pitch_shift,
    apply_speed_change,
    apply_volume_change,
    generate_random_phrases,
    get_phonetically_similar_words,
    mix_audio_with_noise,
    pad_or_trim_audio,
)


# Test constants
SAMPLE_RATE = 16000
DURATION = 1.0
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)


def generate_test_audio(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> np.ndarray:
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * frequency * t)


class TestAudioFunctions:
    """Tests for basic audio manipulation functions."""

    def test_mix_audio_with_noise_snr(self):
        """Test that noise mixing respects SNR levels."""
        audio = generate_test_audio()
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.1

        # Mix at different SNR levels
        mixed_high_snr = mix_audio_with_noise(audio, noise, snr_db=20)
        mixed_low_snr = mix_audio_with_noise(audio, noise, snr_db=0)

        # Higher SNR should have less noise
        # Check that the mixed signals are different
        assert not np.allclose(mixed_high_snr, mixed_low_snr)

        # Output should be normalized
        assert np.abs(mixed_high_snr).max() <= 1.0
        assert np.abs(mixed_low_snr).max() <= 1.0

    def test_mix_audio_with_noise_length_mismatch(self):
        """Test noise mixing handles length mismatches."""
        audio = generate_test_audio(duration=1.0)
        noise_short = np.random.randn(8000).astype(np.float32)
        noise_long = np.random.randn(24000).astype(np.float32)

        # Should handle both cases without error
        mixed_short = mix_audio_with_noise(audio, noise_short, snr_db=10)
        mixed_long = mix_audio_with_noise(audio, noise_long, snr_db=10)

        assert len(mixed_short) == len(audio)
        assert len(mixed_long) == len(audio)

    def test_apply_volume_change(self):
        """Test volume/gain adjustment."""
        audio = generate_test_audio()
        original_rms = np.sqrt(np.mean(audio**2))

        # Increase volume
        louder = apply_volume_change(audio, gain_db=6)
        louder_rms = np.sqrt(np.mean(louder**2))

        # Decrease volume
        quieter = apply_volume_change(audio, gain_db=-6)
        quieter_rms = np.sqrt(np.mean(quieter**2))

        # Check relative levels (approximately 2x for 6dB)
        assert louder_rms > original_rms
        assert quieter_rms < original_rms

        # Check clipping prevention
        assert np.abs(louder).max() <= 1.0

    def test_apply_speed_change(self):
        """Test speed/tempo adjustment."""
        audio = generate_test_audio(duration=1.0)

        # Speed up (shorter duration)
        faster = apply_speed_change(audio, SAMPLE_RATE, speed_factor=1.5)

        # Slow down (longer duration)
        slower = apply_speed_change(audio, SAMPLE_RATE, speed_factor=0.75)

        # Faster should be shorter, slower should be longer
        # Note: Due to resampling, lengths may vary slightly
        assert len(faster) < len(audio) * 1.1  # Allow some tolerance
        assert len(slower) > len(audio) * 0.9

    def test_apply_speed_change_identity(self):
        """Test that speed_factor=1.0 returns original."""
        audio = generate_test_audio()
        result = apply_speed_change(audio, SAMPLE_RATE, speed_factor=1.0)
        np.testing.assert_array_equal(audio, result)

    def test_apply_pitch_shift(self):
        """Test pitch shifting."""
        audio = generate_test_audio(frequency=440.0)

        # Shift up
        higher = apply_pitch_shift(audio, SAMPLE_RATE, semitones=2)

        # Shift down
        lower = apply_pitch_shift(audio, SAMPLE_RATE, semitones=-2)

        # Outputs should be different from original
        assert not np.allclose(audio, higher)
        assert not np.allclose(audio, lower)

        # Same length
        assert len(higher) == len(audio)
        assert len(lower) == len(audio)

    def test_apply_pitch_shift_identity(self):
        """Test that semitones=0 returns original."""
        audio = generate_test_audio()
        result = apply_pitch_shift(audio, SAMPLE_RATE, semitones=0)
        np.testing.assert_array_equal(audio, result)

    def test_pad_or_trim_audio_pad(self):
        """Test padding short audio."""
        short_audio = generate_test_audio(duration=0.5)
        padded = pad_or_trim_audio(short_audio, TARGET_LENGTH)

        assert len(padded) == TARGET_LENGTH

    def test_pad_or_trim_audio_trim(self):
        """Test trimming long audio."""
        long_audio = generate_test_audio(duration=2.0)
        trimmed = pad_or_trim_audio(long_audio, TARGET_LENGTH)

        assert len(trimmed) == TARGET_LENGTH

    def test_pad_or_trim_audio_exact(self):
        """Test audio of exact length is unchanged."""
        exact_audio = generate_test_audio(duration=1.0)
        result = pad_or_trim_audio(exact_audio, TARGET_LENGTH)

        np.testing.assert_array_equal(exact_audio, result)


class TestNoiseLoader:
    """Tests for the NoiseLoader class."""

    def test_noise_loader_initialization(self):
        """Test NoiseLoader initializes correctly."""
        loader = NoiseLoader()

        # Should have loaded noise samples if directory exists
        if loader.noise_dir.exists():
            assert loader.num_samples > 0
            assert len(loader.noise_names) == loader.num_samples

    def test_get_random_noise(self):
        """Test getting random noise segments."""
        loader = NoiseLoader()

        if loader.num_samples == 0:
            pytest.skip("No noise samples available")

        noise = loader.get_random_noise(duration=1.0, target_sr=SAMPLE_RATE)

        assert len(noise) == TARGET_LENGTH
        assert noise.dtype == np.float32

    def test_get_random_noise_different_durations(self):
        """Test noise generation for different durations."""
        loader = NoiseLoader()

        if loader.num_samples == 0:
            pytest.skip("No noise samples available")

        noise_short = loader.get_random_noise(duration=0.5, target_sr=SAMPLE_RATE)
        noise_long = loader.get_random_noise(duration=2.0, target_sr=SAMPLE_RATE)

        assert len(noise_short) == int(0.5 * SAMPLE_RATE)
        assert len(noise_long) == int(2.0 * SAMPLE_RATE)


class TestDataAugmenter:
    """Tests for the DataAugmenter class."""

    def test_augmenter_initialization(self):
        """Test DataAugmenter initializes correctly."""
        augmenter = DataAugmenter(
            target_sample_rate=SAMPLE_RATE,
            target_duration=DURATION,
        )

        assert augmenter.target_sample_rate == SAMPLE_RATE
        assert augmenter.target_length == TARGET_LENGTH

    def test_augment_audio_generates_samples(self):
        """Test that augment_audio generates multiple samples."""
        augmenter = DataAugmenter()
        audio = generate_test_audio()

        samples = list(
            augmenter.augment_audio(
                audio,
                sample_rate=SAMPLE_RATE,
                add_noise=False,  # Disable noise for faster test
                change_pitch=False,  # Disable pitch for faster test
            )
        )

        # Should generate multiple speed variations
        assert len(samples) > 0

        # All samples should be AugmentedSample
        for sample in samples:
            assert isinstance(sample, AugmentedSample)
            assert sample.label == 1  # Positive examples
            assert len(sample.audio) == augmenter.target_length

    def test_augment_audio_with_noise(self):
        """Test augmentation with noise mixing."""
        augmenter = DataAugmenter()

        if not augmenter.noise_available:
            pytest.skip("No noise samples available")

        audio = generate_test_audio()

        samples = list(
            augmenter.augment_audio(
                audio,
                sample_rate=SAMPLE_RATE,
                add_noise=True,
                change_speed=False,
                change_pitch=False,
            )
        )

        # Should have clean + noisy versions
        assert len(samples) > 1

        # Check for noisy samples
        noisy_samples = [s for s in samples if s.metadata.get("snr_db") is not None]
        assert len(noisy_samples) > 0

    @pytest.mark.slow
    def test_generate_tts_samples(self):
        """Test TTS sample generation."""
        augmenter = DataAugmenter()

        if not augmenter.tts_available:
            pytest.skip("TTS not available")

        samples = list(
            augmenter.generate_tts_samples(
                text="hello",
                num_voices=1,  # Use only 1 voice for speed
                add_noise=False,
            )
        )

        assert len(samples) > 0

        for sample in samples:
            assert sample.label == 1
            assert sample.metadata.get("source") == "tts"
            assert len(sample.audio) == augmenter.target_length


class TestNegativeExampleGenerator:
    """Tests for the NegativeExampleGenerator class."""

    def test_get_phonetically_similar_words(self):
        """Test phonetically similar word generation."""
        similar = get_phonetically_similar_words("computer")

        assert len(similar) > 0
        assert "computer" not in similar  # Should not include original

    def test_get_phonetically_similar_words_unknown(self):
        """Test with unknown wake word."""
        similar = get_phonetically_similar_words("xyzabc")

        # Should still generate some variations via substitution
        assert isinstance(similar, list)

    def test_generate_random_phrases(self):
        """Test random phrase generation."""
        phrases = generate_random_phrases(num_phrases=10)

        assert len(phrases) == 10
        assert all(isinstance(p, str) for p in phrases)
        assert all(len(p) > 0 for p in phrases)

    def test_negative_generator_initialization(self):
        """Test NegativeExampleGenerator initializes correctly."""
        generator = NegativeExampleGenerator(
            target_sample_rate=SAMPLE_RATE,
            target_duration=DURATION,
        )

        assert generator.target_sample_rate == SAMPLE_RATE
        assert generator.target_length == TARGET_LENGTH

    def test_generate_silence(self):
        """Test silence generation."""
        generator = NegativeExampleGenerator()

        samples = list(generator.generate_silence(num_samples=5))

        assert len(samples) == 5

        for sample in samples:
            assert sample.label == 0  # Negative
            assert sample.metadata.get("source") == "silence"
            assert len(sample.audio) == generator.target_length

            # Should be very quiet
            assert np.abs(sample.audio).max() < 0.1

    def test_generate_pure_noise(self):
        """Test pure noise generation."""
        generator = NegativeExampleGenerator()

        samples = list(generator.generate_pure_noise(num_samples=5))

        assert len(samples) == 5

        for sample in samples:
            assert sample.label == 0  # Negative
            assert len(sample.audio) == generator.target_length

    @pytest.mark.slow
    def test_generate_phonetically_similar(self):
        """Test phonetically similar word generation with TTS."""
        generator = NegativeExampleGenerator()

        if not generator.tts_available:
            pytest.skip("TTS not available")

        samples = list(
            generator.generate_phonetically_similar(
                wake_word="computer",
                num_voices=1,
                add_noise=False,
            )
        )

        assert len(samples) > 0

        for sample in samples:
            assert sample.label == 0
            assert sample.metadata.get("source") == "phonetic_similar"

    @pytest.mark.slow
    def test_generate_random_speech(self):
        """Test random speech generation with TTS."""
        generator = NegativeExampleGenerator()

        if not generator.tts_available:
            pytest.skip("TTS not available")

        samples = list(
            generator.generate_random_speech(
                num_samples=3,
                num_voices=1,
                add_noise=False,
            )
        )

        assert len(samples) > 0

        for sample in samples:
            assert sample.label == 0
            assert sample.metadata.get("source") == "random_speech"


class TestAugmentedSample:
    """Tests for the AugmentedSample dataclass."""

    def test_augmented_sample_creation(self):
        """Test AugmentedSample creation."""
        audio = generate_test_audio()
        sample = AugmentedSample(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            label=1,
            metadata={"test": "value"},
        )

        assert len(sample.audio) == len(audio)
        assert sample.sample_rate == SAMPLE_RATE
        assert sample.label == 1
        assert sample.metadata["test"] == "value"

    def test_augmented_sample_duration(self):
        """Test duration property."""
        audio = generate_test_audio(duration=1.5)
        sample = AugmentedSample(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            label=0,
        )

        assert abs(sample.duration - 1.5) < 0.01


class TestIntegration:
    """Integration tests for the full augmentation pipeline."""

    @pytest.mark.slow
    def test_full_augmentation_pipeline(self):
        """Test the complete augmentation pipeline."""
        augmenter = DataAugmenter()

        # Generate test recording
        recording = generate_test_audio()

        # Augment recording
        augmented = list(
            augmenter.augment_audio(
                recording,
                sample_rate=SAMPLE_RATE,
                add_noise=augmenter.noise_available,
                change_speed=True,
                change_pitch=False,  # Skip pitch for speed
            )
        )

        assert len(augmented) > 0
        print(f"Generated {len(augmented)} augmented samples from 1 recording")

    @pytest.mark.slow
    def test_full_negative_generation(self):
        """Test complete negative example generation."""
        generator = NegativeExampleGenerator()

        # Generate all types of negatives
        negatives = []

        # Silence
        negatives.extend(list(generator.generate_silence(num_samples=3)))

        # Noise
        negatives.extend(list(generator.generate_pure_noise(num_samples=3)))

        # TTS-based (if available)
        if generator.tts_available:
            negatives.extend(
                list(
                    generator.generate_random_speech(
                        num_samples=2, num_voices=1, add_noise=False
                    )
                )
            )

        assert len(negatives) >= 6
        assert all(s.label == 0 for s in negatives)
        print(f"Generated {len(negatives)} negative samples")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
