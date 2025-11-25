"""
Audio processing module for WakeBuilder.

This module handles audio preprocessing, including mel spectrogram computation
and audio augmentation for training.
"""

from .preprocessing import (
    AudioPreprocessor,
    compute_mel_spectrogram,
    load_audio,
    normalize_audio,
)

__all__ = [
    "AudioPreprocessor",
    "compute_mel_spectrogram",
    "load_audio",
    "normalize_audio",
]
