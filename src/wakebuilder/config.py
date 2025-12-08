"""
Configuration settings for WakeBuilder.

This module centralizes all configuration parameters for the training platform,
including paths, model parameters, and training hyperparameters.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODELS_DIR = MODELS_DIR / "default"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
TTS_VOICES_DIR = PROJECT_ROOT / "tts_voices"

# Audio processing parameters
AUDIO_CONFIG: Dict[str, Any] = {
    "sample_rate": 16000,  # 16 kHz for speech processing
    "n_mels": 80,  # Number of mel frequency bins
    "n_fft": 512,  # FFT window size
    "hop_length": 160,  # Hop length for STFT (10ms at 16kHz)
    "win_length": 400,  # Window length (25ms at 16kHz)
    "fmin": 0,  # Minimum frequency
    "fmax": 8000,  # Maximum frequency (Nyquist for 16kHz)
    "duration": 1.0,  # Expected wake word duration in seconds
}

# Data augmentation parameters
AUGMENTATION_CONFIG: Dict[str, Any] = {
    "speed_variations": [0.8, 0.9, 1.0, 1.1, 1.2],  # Speed multipliers
    "pitch_shifts": [-2, -1, 0, 1, 2],  # Semitones
    "noise_levels": [-20, -15, -10, -5],  # SNR in dB
    "volume_range": (0.7, 1.3),  # Volume multiplier range
    "min_synthetic_samples": 500,  # Minimum synthetic samples to generate
    "negative_ratio": 4,  # Ratio of negative to positive examples
}

# AST Model configuration
AST_CONFIG: Dict[str, Any] = {
    "model_checkpoint": "MIT/ast-finetuned-speech-commands-v2",
    "embedding_dim": 768,  # AST hidden size
    "freeze_base": True,  # Freeze AST base model during training
    "max_length": 1024,  # Maximum sequence length for AST
}

# Training parameters
TRAINING_CONFIG: Dict[str, Any] = {
    "embedding_dim": 768,  # AST embedding dimension
    "hidden_dims": [256, 128],  # Hidden layer dimensions for classifier
    "dropout_rate": 0.3,  # Dropout for regularization
    "learning_rate": 0.001,  # Initial learning rate
    "batch_size": 32,  # Training batch size
    "max_epochs": 50,  # Maximum training epochs
    "early_stopping_patience": 5,  # Epochs to wait before early stopping
    "validation_split": 0.2,  # Fraction of data for validation
    "weight_decay": 1e-5,  # L2 regularization
}

# Model evaluation parameters
EVALUATION_CONFIG: Dict[str, Any] = {
    "threshold_range": (0.3, 0.9),  # Range for threshold search
    "threshold_step": 0.05,  # Step size for threshold search
    "default_threshold": 0.5,  # Default detection threshold
}

# Wake word validation parameters
WAKEWORD_CONFIG: Dict[str, Any] = {
    "min_length": 2,  # Minimum characters
    "max_length": 30,  # Maximum characters
    "max_words": 2,  # Maximum number of words
    "allowed_chars": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ",
}

# Recording parameters
RECORDING_CONFIG: Dict[str, Any] = {
    "min_recordings": 1,  # Minimum user recordings (TTS generates the rest)
    "max_recordings": 5,  # Maximum user recordings accepted
    "min_duration": 0.5,  # Minimum recording duration in seconds
    "max_duration": 3.0,  # Maximum recording duration in seconds
}

# API configuration
API_CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,  # Set to True for development
    "log_level": "info",
}

# WebSocket configuration
WEBSOCKET_CONFIG: Dict[str, Any] = {
    "chunk_duration": 0.5,  # Duration of audio chunks in seconds
    "detection_cooldown": 1.0,  # Minimum time between detections in seconds
}


class Config:
    """Configuration class for WakeBuilder."""

    # Paths
    PROJECT_ROOT = PROJECT_ROOT
    SRC_ROOT = SRC_ROOT
    MODELS_DIR = MODELS_DIR
    DEFAULT_MODELS_DIR = DEFAULT_MODELS_DIR
    CUSTOM_MODELS_DIR = CUSTOM_MODELS_DIR
    DATA_DIR = DATA_DIR
    TEMP_DIR = TEMP_DIR
    TTS_VOICES_DIR = TTS_VOICES_DIR

    # Audio processing
    SAMPLE_RATE = AUDIO_CONFIG["sample_rate"]
    N_MELS = AUDIO_CONFIG["n_mels"]
    N_FFT = AUDIO_CONFIG["n_fft"]
    HOP_LENGTH = AUDIO_CONFIG["hop_length"]
    WIN_LENGTH = AUDIO_CONFIG["win_length"]
    FMIN = AUDIO_CONFIG["fmin"]
    FMAX = AUDIO_CONFIG["fmax"]
    DURATION = AUDIO_CONFIG["duration"]
    MEL_SPEC_TIME_DIM = 96  # Target time dimension for mel spectrograms

    # AST Model
    AST_MODEL_CHECKPOINT = AST_CONFIG["model_checkpoint"]
    AST_EMBEDDING_DIM = AST_CONFIG["embedding_dim"]
    AST_FREEZE_BASE = AST_CONFIG["freeze_base"]
    AST_MAX_LENGTH = AST_CONFIG["max_length"]

    # Training
    EMBEDDING_DIM = TRAINING_CONFIG["embedding_dim"]
    HIDDEN_DIMS = TRAINING_CONFIG["hidden_dims"]
    DROPOUT_RATE = TRAINING_CONFIG["dropout_rate"]
    LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
    BATCH_SIZE = TRAINING_CONFIG["batch_size"]
    MAX_EPOCHS = TRAINING_CONFIG["max_epochs"]
    EARLY_STOPPING_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]
    VALIDATION_SPLIT = TRAINING_CONFIG["validation_split"]
    WEIGHT_DECAY = TRAINING_CONFIG["weight_decay"]

    # Augmentation
    SPEED_VARIATIONS = AUGMENTATION_CONFIG["speed_variations"]
    PITCH_SHIFTS = AUGMENTATION_CONFIG["pitch_shifts"]
    NOISE_LEVELS = AUGMENTATION_CONFIG["noise_levels"]
    VOLUME_RANGE = AUGMENTATION_CONFIG["volume_range"]
    MIN_SYNTHETIC_SAMPLES = AUGMENTATION_CONFIG["min_synthetic_samples"]
    NEGATIVE_RATIO = AUGMENTATION_CONFIG["negative_ratio"]

    # Evaluation
    THRESHOLD_RANGE = EVALUATION_CONFIG["threshold_range"]
    THRESHOLD_STEP = EVALUATION_CONFIG["threshold_step"]
    DEFAULT_THRESHOLD = EVALUATION_CONFIG["default_threshold"]

    # Wake word validation
    WAKEWORD_MIN_LENGTH = WAKEWORD_CONFIG["min_length"]
    WAKEWORD_MAX_LENGTH = WAKEWORD_CONFIG["max_length"]
    WAKEWORD_MAX_WORDS = WAKEWORD_CONFIG["max_words"]
    WAKEWORD_ALLOWED_CHARS = WAKEWORD_CONFIG["allowed_chars"]

    # Recording
    MIN_RECORDINGS = RECORDING_CONFIG["min_recordings"]
    MAX_RECORDINGS = RECORDING_CONFIG["max_recordings"]
    MIN_DURATION = RECORDING_CONFIG["min_duration"]
    MAX_DURATION = RECORDING_CONFIG["max_duration"]

    # API
    API_HOST = API_CONFIG["host"]
    API_PORT = API_CONFIG["port"]
    API_RELOAD = API_CONFIG["reload"]
    API_LOG_LEVEL = API_CONFIG["log_level"]

    # WebSocket
    WS_CHUNK_DURATION = WEBSOCKET_CONFIG["chunk_duration"]
    WS_DETECTION_COOLDOWN = WEBSOCKET_CONFIG["detection_cooldown"]


def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        MODELS_DIR,
        MODELS_DIR / "base",
        DEFAULT_MODELS_DIR,
        CUSTOM_MODELS_DIR,
        DATA_DIR,
        TEMP_DIR,
        TTS_VOICES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    print("WakeBuilder Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Audio Sample Rate: {AUDIO_CONFIG['sample_rate']} Hz")
    print(f"Embedding Dimension: {TRAINING_CONFIG['embedding_dim']}")
    print("\nEnsuring directories exist...")
    ensure_directories()
    print("[OK] All directories created successfully")
