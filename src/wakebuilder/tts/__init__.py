"""
TTS (Text-to-Speech) module for WakeBuilder.

This module provides text-to-speech functionality using:
- Piper TTS: Local TTS with multiple voices
- Kokoro TTS: High-quality 82M parameter TTS with 28 English voices

Both TTS engines are used for generating synthetic voice samples
during wake word training to improve model robustness.
"""

from .generator import TTSGenerator, VoiceInfo, list_available_voices

# Kokoro TTS imports - may not be available
try:
    from .kokoro_generator import (
        KokoroTTSGenerator,
        KokoroVoiceInfo,
        list_kokoro_voices,
        KOKORO_VOICES,
        KOKORO_ENGLISH_VOICES,
        KOKORO_SPEED_VARIATIONS,
        KOKORO_VOLUME_VARIATIONS,
        get_kokoro_sample_count,
        KOKORO_AVAILABLE,
    )
except ImportError:
    KOKORO_AVAILABLE = False
    KokoroTTSGenerator = None  # type: ignore
    KokoroVoiceInfo = None  # type: ignore
    list_kokoro_voices = None  # type: ignore
    KOKORO_VOICES = {}  # type: ignore
    KOKORO_ENGLISH_VOICES = {}  # type: ignore
    KOKORO_SPEED_VARIATIONS = []  # type: ignore
    KOKORO_VOLUME_VARIATIONS = []  # type: ignore
    get_kokoro_sample_count = None  # type: ignore

__all__ = [
    # Piper TTS
    "TTSGenerator",
    "VoiceInfo",
    "list_available_voices",
    # Kokoro TTS
    "KokoroTTSGenerator",
    "KokoroVoiceInfo",
    "list_kokoro_voices",
    "KOKORO_VOICES",
    "KOKORO_ENGLISH_VOICES",
    "KOKORO_SPEED_VARIATIONS",
    "KOKORO_VOLUME_VARIATIONS",
    "get_kokoro_sample_count",
    "KOKORO_AVAILABLE",
]
