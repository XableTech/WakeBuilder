"""
Coqui TTS Generator for WakeBuilder.

This module provides text-to-speech functionality using Coqui TTS models
with support for multi-speaker models (VCTK, YourTTS) and single-speaker
European language models.

Package: coqui-tts (supports Python 3.10 - 3.13)
"""

import gc
import os
import re
import sys
from typing import Iterator, Optional

import numpy as np

# Add espeak-ng to PATH on Windows if installed
if sys.platform == "win32":
    espeak_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
    ]
    for esp_path in espeak_paths:
        if os.path.exists(esp_path) and esp_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = esp_path + os.pathsep + os.environ.get("PATH", "")
            break

# Check if Coqui TTS is available
try:
    from TTS.api import TTS
    import torch

    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    TTS = None
    torch = None


# Model configurations - curated list of good models
COQUI_MODELS = {
    "vctk": {
        "model_name": "tts_models/en/vctk/vits",
        "description": "109 English speakers, high quality VITS",
        "multi_speaker": True,
        "languages": ["en"],
    },
    "your_tts": {
        "model_name": "tts_models/multilingual/multi-dataset/your_tts",
        "description": "Multi-lingual model (EN, FR, PT)",
        "multi_speaker": True,
        "languages": ["en", "fr-fr", "pt-br"],
    },
    # NOTE: Tortoise-v2 removed - requires 8GB+ RAM just to load, not suitable for containers
    # If you need Tortoise, run it separately on a high-memory machine
    "ljspeech": {
        "model_name": "tts_models/en/ljspeech/vits",
        "description": "Single speaker, very natural",
        "multi_speaker": False,
        "languages": ["en"],
    },
    "german": {
        "model_name": "tts_models/de/thorsten/vits",
        "description": "German single speaker VITS",
        "multi_speaker": False,
        "languages": ["de"],
    },
    "czech": {
        "model_name": "tts_models/cs/cv/vits",
        "description": "Czech VITS",
        "multi_speaker": False,
        "languages": ["cs"],
    },
    "slovak": {
        "model_name": "tts_models/sk/cv/vits",
        "description": "Slovak VITS",
        "multi_speaker": False,
        "languages": ["sk"],
    },
    "slovenian": {
        "model_name": "tts_models/sl/cv/vits",
        "description": "Slovenian VITS",
        "multi_speaker": False,
        "languages": ["sl"],
    },
    "catalan": {
        "model_name": "tts_models/ca/custom/vits",
        "description": "Catalan VITS",
        "multi_speaker": False,
        "languages": ["ca"],
    },
    "portuguese": {
        "model_name": "tts_models/pt/cv/vits",
        "description": "Portuguese VITS",
        "multi_speaker": False,
        "languages": ["pt"],
    },
}

# Speed variations for augmentation
COQUI_SPEED_VARIATIONS = [1.0, 1.5]

# Volume variations in dB
COQUI_VOLUME_VARIATIONS = [-6, -3, 0, 3]


def list_coqui_voices() -> dict[str, dict]:
    """
    List all available Coqui TTS voices.

    Returns:
        Dictionary mapping voice_id to voice info dict.
    """
    return COQUI_MODELS.copy()


def get_coqui_sample_count() -> int:
    """
    Calculate expected number of samples from Coqui TTS.

    Based on:
    - VCTK: 109 speakers × 2 speeds × 4 volumes = 872
    - YourTTS: ~6 speakers × 3 languages × 2 speeds × 4 volumes = 144
    - Single speaker models: 8 models × 2 speeds × 4 volumes = 64

    Returns:
        Estimated sample count.
    """
    count = 0
    for model_key, info in COQUI_MODELS.items():
        if info["multi_speaker"]:
            if model_key == "vctk":
                speakers = 109
            elif model_key == "your_tts":
                speakers = 6 * len(info["languages"])  # ~6 speakers per language
            else:
                speakers = 10  # Estimate
        else:
            speakers = 1

        count += speakers * len(COQUI_SPEED_VARIATIONS) * len(COQUI_VOLUME_VARIATIONS)

    return count


class CoquiTTSGenerator:
    """
    Text-to-Speech generator using Coqui TTS.

    This class provides methods to generate high-quality synthetic speech
    using multiple Coqui TTS models including multi-speaker models like
    VCTK (109 speakers) and YourTTS (multi-lingual).

    Key features:
    - 109+ English voices from VCTK
    - Multi-lingual support (EN, FR, PT, DE, etc.)
    - Speed variations
    - GPU acceleration when available
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        use_gpu: bool = True,
        models: Optional[list[str]] = None,
    ):
        """
        Initialize the Coqui TTS generator.

        Args:
            target_sample_rate: Target sample rate for output audio (default: 16000 Hz)
            use_gpu: Whether to use GPU if available (default: True)
            models: List of model keys to use. If None, uses all available.
        """
        if not COQUI_AVAILABLE:
            raise ImportError(
                "Coqui TTS is not installed. Install with: uv add coqui-tts"
            )

        self.target_sample_rate = target_sample_rate
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self._device = "cuda" if self._use_gpu else "cpu"

        # Models to use
        if models is None:
            self._model_keys = list(COQUI_MODELS.keys())
        else:
            self._model_keys = [k for k in models if k in COQUI_MODELS]

        # Lazy-loaded TTS instances
        self._tts_instances: dict[str, TTS] = {}

        # Cache for speakers per model
        self._speakers_cache: dict[str, list[str]] = {}

    @property
    def is_available(self) -> bool:
        """Check if Coqui TTS is available."""
        return COQUI_AVAILABLE

    @property
    def model_keys(self) -> list[str]:
        """Get list of model keys."""
        return self._model_keys

    @property
    def using_gpu(self) -> bool:
        """Check if GPU is being used."""
        return self._use_gpu

    def _get_tts(self, model_key: str) -> Optional[TTS]:
        """Get or create TTS instance for a model."""
        if model_key not in self._tts_instances:
            if model_key not in COQUI_MODELS:
                return None

            model_info = COQUI_MODELS[model_key]
            model_name = model_info["model_name"]

            try:
                print(f"  Loading Coqui model: {model_key} ({model_name})...")
                tts = TTS(model_name).to(self._device)
                self._tts_instances[model_key] = tts

                # Cache speakers
                if hasattr(tts, "speakers") and tts.speakers:
                    self._speakers_cache[model_key] = tts.speakers
                else:
                    self._speakers_cache[model_key] = [None]  # Single speaker

                print(
                    f"    Loaded with {len(self._speakers_cache[model_key])} speaker(s)"
                )

            except Exception as e:
                print(f"  [WARNING] Failed to load {model_key}: {e}")
                return None

        return self._tts_instances.get(model_key)

    def get_speakers(self, model_key: str) -> list[Optional[str]]:
        """Get list of speakers for a model."""
        if model_key in self._speakers_cache:
            return self._speakers_cache[model_key]

        # Try to load model to get speakers
        tts = self._get_tts(model_key)
        if tts is None:
            return [None]

        return self._speakers_cache.get(model_key, [None])

    def synthesize(
        self,
        text: str,
        model_key: str = "vctk",
        speaker: Optional[str] = None,
        language: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            model_key: Model key (e.g., 'vctk', 'your_tts').
            speaker: Speaker name for multi-speaker models.
            language: Language code for multi-lingual models.

        Returns:
            Tuple of (audio_samples, sample_rate).
            Audio samples are float32 normalized to [-1, 1].
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        tts = self._get_tts(model_key)
        if tts is None:
            raise ValueError(f"Failed to load model: {model_key}")

        # Build kwargs
        kwargs = {"text": text}
        if speaker:
            kwargs["speaker"] = speaker
        if language and hasattr(tts, "languages") and tts.languages:
            kwargs["language"] = language

        # Generate audio
        audio = tts.tts(**kwargs)
        audio = np.array(audio, dtype=np.float32)

        # Get sample rate from model
        if hasattr(tts, "synthesizer") and hasattr(
            tts.synthesizer, "output_sample_rate"
        ):
            native_sr = tts.synthesizer.output_sample_rate
        else:
            native_sr = 22050  # Default

        # Resample if needed
        if native_sr != self.target_sample_rate:
            audio = self._resample(audio, native_sr, self.target_sample_rate)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0.01:
            audio = audio / max_val * 0.9

        return audio.astype(np.float32), self.target_sample_rate

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def synthesize_all_voices(
        self,
        text: str,
        models: Optional[list[str]] = None,
        max_speakers_per_model: int = -1,
    ) -> Iterator[tuple[np.ndarray, int, dict]]:
        """
        Generate audio for all voices across all models.

        Args:
            text: Text to synthesize.
            models: List of model keys to use. If None, uses all.
            max_speakers_per_model: Max speakers per model (-1 for all).

        Yields:
            Tuples of (audio_samples, sample_rate, metadata).
        """
        if models is None:
            models = self._model_keys

        for model_key in models:
            model_info = COQUI_MODELS.get(model_key)
            if model_info is None:
                continue

            speakers = self.get_speakers(model_key)
            if max_speakers_per_model > 0:
                speakers = speakers[:max_speakers_per_model]

            # Get languages for multi-lingual models
            languages = model_info.get("languages", [None])
            if not model_info.get("multi_speaker"):
                languages = [None]  # Single speaker models don't need language

            for speaker in speakers:
                for lang in languages:
                    try:
                        audio, sr = self.synthesize(
                            text,
                            model_key=model_key,
                            speaker=speaker,
                            language=lang,
                        )

                        # Sanitize speaker name for metadata
                        speaker_str = speaker if speaker else "default"
                        speaker_str = re.sub(r"[\n\r\t]", "_", speaker_str).strip()

                        metadata = {
                            "model_key": model_key,
                            "model_name": model_info["model_name"],
                            "speaker": speaker_str,
                            "language": lang,
                            "tts_engine": "coqui",
                        }

                        yield audio, sr, metadata

                    except Exception:
                        # Skip failed syntheses
                        continue

    def cleanup(self) -> None:
        """
        Clean up GPU memory and resources.

        Call this after generating all samples to free GPU memory
        before training.
        """
        for tts in self._tts_instances.values():
            del tts

        self._tts_instances.clear()
        self._speakers_cache.clear()

        if self._use_gpu and torch is not None:
            torch.cuda.empty_cache()
            gc.collect()
            print("Coqui TTS: GPU memory cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
