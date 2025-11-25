"""
Tests for audio preprocessing module.
"""

import numpy as np
import pytest

from wakebuilder.audio.preprocessing import (
    AudioPreprocessor,
    compute_mel_spectrogram,
    normalize_audio,
)
from wakebuilder.config import Config


class TestNormalizeAudio:
    """Tests for audio normalization."""

    def test_normalize_audio_basic(self):
        """Test basic audio normalization."""
        # Create test audio
        audio = np.random.randn(16000) * 0.1
        
        # Normalize
        normalized = normalize_audio(audio, target_level=-20.0)
        
        # Check output
        assert normalized.shape == audio.shape
        assert np.abs(normalized).max() <= 1.0
        
    def test_normalize_audio_silent(self):
        """Test normalization with silent audio."""
        audio = np.zeros(16000)
        normalized = normalize_audio(audio)
        
        assert np.all(normalized == 0)
        
    def test_normalize_audio_clipping(self):
        """Test that normalization clips properly."""
        audio = np.random.randn(16000) * 10.0
        normalized = normalize_audio(audio)
        
        assert np.abs(normalized).max() <= 1.0


class TestComputeMelSpectrogram:
    """Tests for mel spectrogram computation."""

    def test_compute_mel_spectrogram_shape(self):
        """Test mel spectrogram output shape."""
        # Create test audio (1 second at 16kHz)
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        # Compute mel spectrogram
        mel_spec = compute_mel_spectrogram(
            audio=audio,
            sample_rate=sample_rate,
            n_mels=80,
        )
        
        # Check shape
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_spec.shape[1] > 0  # time dimension
        
    def test_compute_mel_spectrogram_values(self):
        """Test mel spectrogram value range."""
        audio = np.random.randn(16000) * 0.1
        sample_rate = 16000
        
        mel_spec = compute_mel_spectrogram(audio, sample_rate)
        
        # Check that values are in dB scale (typically negative)
        assert np.isfinite(mel_spec).all()
        assert mel_spec.max() <= 0  # dB scale relative to max
        
    def test_compute_mel_spectrogram_parameters(self):
        """Test mel spectrogram with different parameters."""
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        mel_spec = compute_mel_spectrogram(
            audio=audio,
            sample_rate=sample_rate,
            n_mels=40,
            n_fft=1024,
            hop_length=256,
        )
        
        assert mel_spec.shape[0] == 40


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=80,
            target_length=96,
        )
        
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 80
        assert preprocessor.target_length == 96
        
    def test_preprocessor_from_config(self):
        """Test creating preprocessor from config."""
        config = Config()
        preprocessor = AudioPreprocessor.from_config(config)
        
        assert preprocessor.sample_rate == config.SAMPLE_RATE
        assert preprocessor.n_mels == config.N_MELS
        
    def test_process_audio_array(self):
        """Test processing audio array."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=80,
            target_length=96,
        )
        
        # Create test audio (1 second)
        audio = np.random.randn(16000) * 0.1
        
        # Process
        mel_spec = preprocessor.process_audio(audio, sample_rate=16000)
        
        # Check output shape
        assert mel_spec.shape == (96, 80)  # (time, freq)
        assert np.isfinite(mel_spec).all()
        
    def test_process_audio_padding(self):
        """Test that short audio is padded correctly."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            target_length=96,
        )
        
        # Create short audio (0.5 seconds)
        audio = np.random.randn(8000) * 0.1
        
        mel_spec = preprocessor.process_audio(audio, sample_rate=16000)
        
        # Should be padded to target length
        assert mel_spec.shape[0] == 96
        
    def test_process_audio_trimming(self):
        """Test that long audio is trimmed correctly."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            target_length=96,
        )
        
        # Create long audio (3 seconds)
        audio = np.random.randn(48000) * 0.1
        
        mel_spec = preprocessor.process_audio(audio, sample_rate=16000)
        
        # Should be trimmed to target length
        assert mel_spec.shape[0] == 96
        
    def test_process_audio_no_target_length(self):
        """Test processing without target length constraint."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            target_length=None,
        )
        
        audio = np.random.randn(16000) * 0.1
        
        mel_spec = preprocessor.process_audio(audio, sample_rate=16000)
        
        # Should have variable length
        assert mel_spec.shape[0] > 0
        assert mel_spec.shape[1] == 80  # n_mels
        
    def test_process_batch(self):
        """Test batch processing."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            target_length=96,
        )
        
        # Create batch of audio
        audio_list = [
            np.random.randn(16000) * 0.1,
            np.random.randn(16000) * 0.1,
            np.random.randn(16000) * 0.1,
        ]
        sample_rates = [16000, 16000, 16000]
        
        # Process batch
        mel_specs = preprocessor.process_batch(audio_list, sample_rates)
        
        # Check output shape
        assert mel_specs.shape == (3, 96, 80)  # (batch, time, freq)
        assert np.isfinite(mel_specs).all()
        
    def test_process_audio_resampling(self):
        """Test that audio is resampled correctly."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            target_length=96,
        )
        
        # Create audio at different sample rate (8kHz)
        audio = np.random.randn(8000) * 0.1
        
        mel_spec = preprocessor.process_audio(audio, sample_rate=8000)
        
        # Should still produce correct output shape
        assert mel_spec.shape == (96, 80)


class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing pipeline."""
        # Create synthetic audio (wake word simulation)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simple tone + noise
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        audio += np.random.randn(len(audio)) * 0.05
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Create preprocessor
        preprocessor = AudioPreprocessor.from_config()
        
        # Process
        mel_spec = preprocessor.process_audio(audio, sample_rate=sample_rate)
        
        # Verify output
        assert mel_spec.shape[0] == 96  # target length
        assert mel_spec.shape[1] == 80  # n_mels
        assert np.isfinite(mel_spec).all()
        assert mel_spec.std() > 0  # Should have variation
        
    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent."""
        audio = np.random.randn(16000) * 0.1
        preprocessor = AudioPreprocessor.from_config()
        
        # Process same audio twice
        mel_spec1 = preprocessor.process_audio(audio.copy(), sample_rate=16000)
        mel_spec2 = preprocessor.process_audio(audio.copy(), sample_rate=16000)
        
        # Should be identical
        np.testing.assert_array_almost_equal(mel_spec1, mel_spec2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
