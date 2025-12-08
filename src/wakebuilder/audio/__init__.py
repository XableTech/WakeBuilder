"""
Audio processing module for WakeBuilder.

This module handles audio preprocessing, including mel spectrogram computation
and audio augmentation for training.
"""

from .augmentation import (
    AugmentedSample,
    DataAugmenter,
    NoiseLoader,
    apply_pitch_shift,
    apply_speed_change,
    apply_volume_change,
    mix_audio_with_noise,
    pad_or_trim_audio,
)
from .negative_generator import (
    NegativeExampleGenerator,
    generate_random_phrases,
    get_phonetically_similar_words,
)
from .preprocessing import (
    AudioPreprocessor,
    compute_mel_spectrogram,
    load_audio,
    normalize_audio,
)
from .real_data_loader import (
    MassivePositiveAugmenter,
    RealNegativeDataLoader,
    chunk_audio,
    load_audio_file,
)

__all__ = [
    # Preprocessing
    "AudioPreprocessor",
    "compute_mel_spectrogram",
    "load_audio",
    "normalize_audio",
    # Augmentation
    "AugmentedSample",
    "DataAugmenter",
    "NoiseLoader",
    "apply_pitch_shift",
    "apply_speed_change",
    "apply_volume_change",
    "mix_audio_with_noise",
    "pad_or_trim_audio",
    # Negative generation
    "NegativeExampleGenerator",
    "generate_random_phrases",
    "get_phonetically_similar_words",
    # Real data loading
    "RealNegativeDataLoader",
    "MassivePositiveAugmenter",
    "load_audio_file",
    "chunk_audio",
]
