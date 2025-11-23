"""
Tests for configuration module.

This module tests the configuration settings and directory creation
functionality of WakeBuilder.
"""

import pytest
from pathlib import Path
from src.wakebuilder import config


def test_audio_config_has_required_keys():
    """Test that audio configuration contains all required parameters."""
    required_keys = [
        "sample_rate",
        "n_mels",
        "n_fft",
        "hop_length",
        "win_length",
        "fmin",
        "fmax",
        "duration",
    ]
    
    for key in required_keys:
        assert key in config.AUDIO_CONFIG, f"Missing required key: {key}"


def test_sample_rate_is_valid():
    """Test that sample rate is set to standard 16kHz for speech."""
    assert config.AUDIO_CONFIG["sample_rate"] == 16000


def test_training_config_has_valid_values():
    """Test that training configuration has sensible values."""
    assert config.TRAINING_CONFIG["learning_rate"] > 0
    assert config.TRAINING_CONFIG["batch_size"] > 0
    assert config.TRAINING_CONFIG["max_epochs"] > 0
    assert 0 <= config.TRAINING_CONFIG["dropout_rate"] <= 1
    assert 0 < config.TRAINING_CONFIG["validation_split"] < 1


def test_augmentation_config_has_variations():
    """Test that augmentation config includes speed and pitch variations."""
    assert len(config.AUGMENTATION_CONFIG["speed_variations"]) > 0
    assert len(config.AUGMENTATION_CONFIG["pitch_shifts"]) > 0
    assert len(config.AUGMENTATION_CONFIG["noise_levels"]) > 0


def test_wakeword_validation_parameters():
    """Test that wake word validation parameters are reasonable."""
    assert config.WAKEWORD_CONFIG["min_length"] > 0
    assert config.WAKEWORD_CONFIG["max_length"] > config.WAKEWORD_CONFIG["min_length"]
    assert config.WAKEWORD_CONFIG["max_words"] in [1, 2]


def test_recording_config_constraints():
    """Test that recording configuration has valid constraints."""
    assert config.RECORDING_CONFIG["min_recordings"] >= 3
    assert config.RECORDING_CONFIG["max_recordings"] >= config.RECORDING_CONFIG["min_recordings"]
    assert config.RECORDING_CONFIG["min_duration"] > 0
    assert config.RECORDING_CONFIG["max_duration"] > config.RECORDING_CONFIG["min_duration"]


def test_project_paths_are_absolute():
    """Test that all configured paths are absolute paths."""
    assert config.PROJECT_ROOT.is_absolute()
    assert config.MODELS_DIR.is_absolute()
    assert config.DATA_DIR.is_absolute()


def test_ensure_directories_creates_structure(tmp_path, monkeypatch):
    """Test that ensure_directories creates all required directories."""
    # Temporarily change PROJECT_ROOT to tmp_path
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "DEFAULT_MODELS_DIR", tmp_path / "models" / "default")
    monkeypatch.setattr(config, "CUSTOM_MODELS_DIR", tmp_path / "models" / "custom")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "data" / "temp")
    monkeypatch.setattr(config, "TTS_VOICES_DIR", tmp_path / "tts_voices")
    
    # Call ensure_directories
    config.ensure_directories()
    
    # Verify directories were created
    assert (tmp_path / "models").exists()
    assert (tmp_path / "models" / "default").exists()
    assert (tmp_path / "models" / "custom").exists()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "data" / "temp").exists()
    assert (tmp_path / "tts_voices").exists()


def test_negative_ratio_is_reasonable():
    """Test that negative to positive example ratio is reasonable."""
    ratio = config.AUGMENTATION_CONFIG["negative_ratio"]
    assert 2 <= ratio <= 10, "Negative ratio should be between 2 and 10"


def test_threshold_range_is_valid():
    """Test that evaluation threshold range is valid."""
    min_threshold, max_threshold = config.EVALUATION_CONFIG["threshold_range"]
    assert 0 < min_threshold < max_threshold < 1
    assert config.EVALUATION_CONFIG["default_threshold"] >= min_threshold
    assert config.EVALUATION_CONFIG["default_threshold"] <= max_threshold
