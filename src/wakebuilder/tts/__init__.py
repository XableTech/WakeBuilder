"""
TTS (Text-to-Speech) module for WakeBuilder.

This module provides text-to-speech functionality using:
- Piper TTS: Local TTS with multiple voices
- Kokoro TTS: High-quality 82M parameter TTS with 28 English voices
- Coqui TTS: Multi-speaker models (VCTK 109 voices, YourTTS multi-lingual)
- Edge TTS: Microsoft neural TTS with 400+ voices

All TTS engines are used for generating synthetic voice samples
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

# Coqui TTS imports - may not be available
try:
    from .coqui_generator import (
        CoquiTTSGenerator,
        COQUI_MODELS,
        COQUI_SPEED_VARIATIONS,
        COQUI_VOLUME_VARIATIONS,
        list_coqui_voices,
        get_coqui_sample_count,
        COQUI_AVAILABLE,
    )
except ImportError:
    COQUI_AVAILABLE = False
    CoquiTTSGenerator = None  # type: ignore
    COQUI_MODELS = {}  # type: ignore
    COQUI_SPEED_VARIATIONS = []  # type: ignore
    COQUI_VOLUME_VARIATIONS = []  # type: ignore
    list_coqui_voices = None  # type: ignore
    get_coqui_sample_count = None  # type: ignore

# Edge TTS imports - may not be available
try:
    from .edge_generator import (
        EdgeTTSGenerator,
        EDGE_ENGLISH_VOICES,
        EDGE_EUROPEAN_VOICES,
        EDGE_DEFAULT_VOICES,
        EDGE_SPEED_VARIATIONS,
        EDGE_VOLUME_VARIATIONS,
        get_edge_sample_count,
        EDGE_TTS_AVAILABLE,
    )
except ImportError:
    EDGE_TTS_AVAILABLE = False
    EdgeTTSGenerator = None  # type: ignore
    EDGE_ENGLISH_VOICES = []  # type: ignore
    EDGE_EUROPEAN_VOICES = []  # type: ignore
    EDGE_DEFAULT_VOICES = []  # type: ignore
    EDGE_SPEED_VARIATIONS = []  # type: ignore
    EDGE_VOLUME_VARIATIONS = []  # type: ignore
    get_edge_sample_count = None  # type: ignore

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
    # Coqui TTS
    "CoquiTTSGenerator",
    "COQUI_MODELS",
    "COQUI_SPEED_VARIATIONS",
    "COQUI_VOLUME_VARIATIONS",
    "list_coqui_voices",
    "get_coqui_sample_count",
    "COQUI_AVAILABLE",
    # Edge TTS
    "EdgeTTSGenerator",
    "EDGE_ENGLISH_VOICES",
    "EDGE_EUROPEAN_VOICES",
    "EDGE_DEFAULT_VOICES",
    "EDGE_SPEED_VARIATIONS",
    "EDGE_VOLUME_VARIATIONS",
    "get_edge_sample_count",
    "EDGE_TTS_AVAILABLE",
]
