"""
Pytest configuration and shared fixtures for WakeBuilder tests.

This module provides common fixtures and configuration used across
all test modules.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a sample audio waveform for testing.
    
    Returns:
        A 1-second audio waveform at 16kHz sample rate.
    """
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    return audio.astype(np.float32)


@pytest.fixture
def temp_audio_file(tmp_path: Path, sample_audio: np.ndarray) -> Path:
    """Create a temporary audio file for testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_audio: Sample audio waveform fixture.
    
    Returns:
        Path to the temporary audio file.
    """
    import soundfile as sf
    
    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), sample_audio, 16000)
    
    return audio_path


@pytest.fixture
def mock_wake_word() -> str:
    """Provide a mock wake word for testing.
    
    Returns:
        A test wake word string.
    """
    return "Phoenix"


@pytest.fixture
def mock_recordings(sample_audio: np.ndarray) -> list[np.ndarray]:
    """Generate multiple mock recordings for testing.
    
    Args:
        sample_audio: Sample audio waveform fixture.
    
    Returns:
        List of audio waveforms representing multiple recordings.
    """
    # Create 3 slightly different versions
    recordings = []
    for i in range(3):
        # Add small variations
        noise = np.random.randn(*sample_audio.shape) * 0.01
        recording = sample_audio + noise
        recordings.append(recording.astype(np.float32))
    
    return recordings
