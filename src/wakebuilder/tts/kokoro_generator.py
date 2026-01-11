"""
Kokoro TTS Generator for WakeBuilder.

This module provides a wrapper around Kokoro TTS (hexgrad/Kokoro-82M) for generating
high-quality synthetic voice samples with multiple voices and speed variations.

Kokoro TTS is an 82M parameter open-weight TTS model that delivers high quality
speech synthesis with support for multiple languages and voices.

References:
- Model: https://huggingface.co/hexgrad/Kokoro-82M
- GitHub: https://github.com/hexgrad/kokoro
"""

import gc
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# Suppress torch RNN dropout warning (Kokoro uses num_layers=1 with dropout)
warnings.filterwarnings(
    "ignore", message="dropout option adds dropout after all but last recurrent layer"
)
# Suppress Kokoro repo_id default warning (we explicitly set it where possible)
warnings.filterwarnings("ignore", message="Defaulting repo_id to hexgrad/Kokoro-82M")

# Kokoro imports - wrapped in try/except for graceful degradation
try:
    import torch
    from kokoro import KModel, KPipeline

    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KModel = None  # type: ignore
    KPipeline = None  # type: ignore
    torch = None  # type: ignore


# All available Kokoro voices organized by language
# Format: voice_id -> (display_name, gender, language/accent, lang_code)
KOKORO_VOICES = {
    # American English Female (11 voices) - lang_code='a'
    "af_heart": ("Heart", "female", "american", "a"),
    "af_bella": ("Bella", "female", "american", "a"),
    "af_nicole": ("Nicole", "female", "american", "a"),
    "af_aoede": ("Aoede", "female", "american", "a"),
    "af_kore": ("Kore", "female", "american", "a"),
    "af_sarah": ("Sarah", "female", "american", "a"),
    "af_nova": ("Nova", "female", "american", "a"),
    "af_sky": ("Sky", "female", "american", "a"),
    "af_alloy": ("Alloy", "female", "american", "a"),
    "af_jessica": ("Jessica", "female", "american", "a"),
    "af_river": ("River", "female", "american", "a"),
    # American English Male (9 voices) - lang_code='a'
    "am_michael": ("Michael", "male", "american", "a"),
    "am_fenrir": ("Fenrir", "male", "american", "a"),
    "am_puck": ("Puck", "male", "american", "a"),
    "am_echo": ("Echo", "male", "american", "a"),
    "am_eric": ("Eric", "male", "american", "a"),
    "am_liam": ("Liam", "male", "american", "a"),
    "am_onyx": ("Onyx", "male", "american", "a"),
    "am_santa": ("Santa", "male", "american", "a"),
    "am_adam": ("Adam", "male", "american", "a"),
    # British English Female (4 voices) - lang_code='b'
    "bf_emma": ("Emma", "female", "british", "b"),
    "bf_isabella": ("Isabella", "female", "british", "b"),
    "bf_alice": ("Alice", "female", "british", "b"),
    "bf_lily": ("Lily", "female", "british", "b"),
    # British English Male (4 voices) - lang_code='b'
    "bm_george": ("George", "male", "british", "b"),
    "bm_fable": ("Fable", "male", "british", "b"),
    "bm_lewis": ("Lewis", "male", "british", "b"),
    "bm_daniel": ("Daniel", "male", "british", "b"),
    # Spanish Female (1 voice) - lang_code='e'
    "ef_dora": ("Dora", "female", "spanish", "e"),
    # Spanish Male (2 voices) - lang_code='e'
    "em_alex": ("Alex", "male", "spanish", "e"),
    "em_santa": ("Santa", "male", "spanish", "e"),
    # French Female (1 voice) - lang_code='f'
    "ff_siwis": ("Siwis", "female", "french", "f"),
    # Hindi Female (2 voices) - lang_code='h'
    "hf_alpha": ("Alpha", "female", "hindi", "h"),
    "hf_beta": ("Beta", "female", "hindi", "h"),
    # Hindi Male (2 voices) - lang_code='h'
    "hm_omega": ("Omega", "male", "hindi", "h"),
    "hm_psi": ("Psi", "male", "hindi", "h"),
    # Italian Female (1 voice) - lang_code='i'
    "if_sara": ("Sara", "female", "italian", "i"),
    # Italian Male (1 voice) - lang_code='i'
    "im_nicola": ("Nicola", "male", "italian", "i"),
    # Brazilian Portuguese Female (1 voice) - lang_code='p'
    "pf_dora": ("Dora", "female", "portuguese", "p"),
    # Brazilian Portuguese Male (2 voices) - lang_code='p'
    "pm_alex": ("Alex", "male", "portuguese", "p"),
    "pm_santa": ("Santa", "male", "portuguese", "p"),
    # Mandarin Chinese Female (4 voices) - lang_code='z'
    "zf_xiaobei": ("Xiaobei", "female", "chinese", "z"),
    "zf_xiaoni": ("Xiaoni", "female", "chinese", "z"),
    "zf_xiaoxiao": ("Xiaoxiao", "female", "chinese", "z"),
    "zf_xiaoyi": ("Xiaoyi", "female", "chinese", "z"),
    # Mandarin Chinese Male (4 voices) - lang_code='z'
    "zm_yunjian": ("Yunjian", "male", "chinese", "z"),
    "zm_yunxi": ("Yunxi", "male", "chinese", "z"),
    "zm_yunxia": ("Yunxia", "male", "chinese", "z"),
    "zm_yunyang": ("Yunyang", "male", "chinese", "z"),
}

# English-only voices for wake word training (most relevant for English wake words)
KOKORO_ENGLISH_VOICES = {k: v for k, v in KOKORO_VOICES.items() if v[3] in ("a", "b")}

# Speed variations (1.0 = normal, 1.5 = faster)
KOKORO_SPEED_VARIATIONS = [1.0, 1.5]

# Volume variations in dB for augmentation
KOKORO_VOLUME_VARIATIONS = [-6, -3, 0, 3]

# Kokoro native sample rate
KOKORO_SAMPLE_RATE = 24000


@dataclass
class KokoroVoiceInfo:
    """Information about a Kokoro voice."""

    voice_id: str
    display_name: str
    gender: str
    accent: str

    @property
    def lang_code(self) -> str:
        """Get the language code for this voice."""
        return self.voice_id[0]  # 'a' for American, 'b' for British


def list_kokoro_voices() -> list[KokoroVoiceInfo]:
    """
    List all available Kokoro voices.

    Returns:
        List of KokoroVoiceInfo objects for all voices.
    """
    voices = []
    for voice_id, info in KOKORO_VOICES.items():
        name, gender, accent, lang_code = info
        voices.append(
            KokoroVoiceInfo(
                voice_id=voice_id,
                display_name=name,
                gender=gender,
                accent=accent,
            )
        )
    return voices


class KokoroTTSGenerator:
    """
    Text-to-Speech generator using Kokoro TTS.

    This class provides methods to generate high-quality synthetic speech
    with support for 28 English voices (20 American + 8 British) and
    speed variations.

    Key features:
    - 28 diverse English voices (male/female, American/British)
    - Speed variations (0.5x and 1.0x)
    - GPU acceleration when available
    - Automatic GPU memory cleanup after generation
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        use_gpu: bool = True,
    ):
        """
        Initialize the Kokoro TTS generator.

        Args:
            target_sample_rate: Target sample rate for output audio (default: 16000 Hz)
            use_gpu: Whether to use GPU if available (default: True)
        """
        if not KOKORO_AVAILABLE:
            raise ImportError(
                "Kokoro TTS is not installed. Install with: uv add kokoro soundfile"
            )

        self.target_sample_rate = target_sample_rate
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self._device = "cuda" if self._use_gpu else "cpu"

        # Lazy-loaded model and pipelines
        self._model: Optional[KModel] = None
        self._pipelines: dict[str, KPipeline] = {}
        self._voice_packs: dict[str, any] = {}

        # Available voices
        self._voices = list_kokoro_voices()

    @property
    def is_available(self) -> bool:
        """Check if Kokoro TTS is available."""
        return KOKORO_AVAILABLE

    @property
    def voice_ids(self) -> list[str]:
        """Get list of all voice IDs."""
        return list(KOKORO_VOICES.keys())

    @property
    def num_voices(self) -> int:
        """Get number of available voices."""
        return len(KOKORO_VOICES)

    @property
    def using_gpu(self) -> bool:
        """Check if GPU is being used."""
        return self._use_gpu

    def _ensure_model_loaded(self) -> None:
        """Ensure the Kokoro model is loaded."""
        if self._model is None:
            print(f"Loading Kokoro model on {self._device}...")
            # Explicitly pass repo_id to suppress warning
            self._model = KModel(repo_id="hexgrad/Kokoro-82M").to(self._device).eval()
            print(f"Kokoro model loaded successfully on {self._device}")

    def _get_pipeline(self, voice_id: str) -> KPipeline:
        """Get the appropriate pipeline for a voice, creating if needed."""
        # Get lang_code from voice info or infer from voice_id prefix
        voice_info = KOKORO_VOICES.get(voice_id)
        if voice_info:
            lang_code = voice_info[3]  # 4th element is lang_code
        else:
            lang_code = voice_id[0]  # Fallback to first character

        # Create pipeline if not cached
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False)

        return self._pipelines[lang_code]

    def _load_voice_pack(self, voice_id: str) -> any:
        """Load and cache a voice pack."""
        if voice_id not in self._voice_packs:
            pipeline = self._get_pipeline(voice_id)
            # Suppress the "Defaulting repo_id" warning printed by Kokoro
            # Kokoro uses print() which goes to stdout
            import sys
            import io

            # Suppress stdout (where Kokoro prints the warning)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            # Also suppress stderr just in case
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                self._voice_packs[voice_id] = pipeline.load_voice(voice_id)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        return self._voice_packs[voice_id]

    def synthesize(
        self,
        text: str,
        voice_id: str = "af_heart",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_id: Kokoro voice ID (e.g., 'af_heart', 'am_michael').
            speed: Speed multiplier (0.5 = slower, 1.0 = normal, 2.0 = faster).

        Returns:
            Tuple of (audio_samples, sample_rate).
            Audio samples are float32 normalized to [-1, 1].

        Raises:
            ValueError: If voice_id is invalid or text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if voice_id not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown voice: {voice_id}. Available: {list(KOKORO_VOICES.keys())}"
            )

        # Ensure model is loaded
        self._ensure_model_loaded()

        # Get pipeline and voice pack
        pipeline = self._get_pipeline(voice_id)
        pack = self._load_voice_pack(voice_id)

        # Generate audio
        audio_segments = []
        for _, ps, _ in pipeline(text, voice_id, speed):
            ref_s = pack[len(ps) - 1]
            audio = self._model(ps, ref_s, speed)
            audio_segments.append(audio.cpu().numpy())

        if not audio_segments:
            raise ValueError(f"Failed to generate audio for text: {text}")

        # Concatenate all segments
        audio = np.concatenate(audio_segments)

        # Resample if needed
        if KOKORO_SAMPLE_RATE != self.target_sample_rate:
            audio = self._resample(audio, KOKORO_SAMPLE_RATE, self.target_sample_rate)

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
        speeds: Optional[list[float]] = None,
        voices: Optional[list[str]] = None,
    ) -> Iterator[tuple[np.ndarray, int, dict]]:
        """
        Generate audio for all voices with speed variations.

        Args:
            text: Text to synthesize.
            speeds: List of speed multipliers (default: [0.5, 1.0]).
            voices: List of voice IDs to use (default: all voices).

        Yields:
            Tuples of (audio_samples, sample_rate, metadata).
            Metadata includes voice_id, voice_name, gender, accent, speed.
        """
        if speeds is None:
            speeds = KOKORO_SPEED_VARIATIONS

        if voices is None:
            voices = self.voice_ids

        for voice_id in voices:
            voice_info = KOKORO_VOICES.get(voice_id)
            if voice_info is None:
                continue

            display_name, gender, accent = voice_info

            for speed in speeds:
                try:
                    audio, sr = self.synthesize(text, voice_id=voice_id, speed=speed)

                    metadata = {
                        "voice_id": voice_id,
                        "voice_name": display_name,
                        "gender": gender,
                        "accent": accent,
                        "speed": speed,
                        "text": text,
                        "tts_engine": "kokoro",
                    }

                    yield audio, sr, metadata

                except Exception as e:
                    print(
                        f"Warning: Failed to synthesize with {voice_id} at speed {speed}: {e}"
                    )
                    continue

    def cleanup(self) -> None:
        """
        Clean up GPU memory and resources.

        Call this after generating all samples to free GPU memory
        before training.
        """
        if self._model is not None:
            del self._model
            self._model = None

        self._pipelines.clear()
        self._voice_packs.clear()

        if self._use_gpu and torch is not None:
            torch.cuda.empty_cache()
            gc.collect()
            print("Kokoro TTS: GPU memory cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass


def get_kokoro_sample_count(num_voices: Optional[int] = None) -> int:
    """
    Calculate how many samples will be generated.

    Args:
        num_voices: Number of voices to use (default: all 28).

    Returns:
        Total number of samples (voices × speed variations).
    """
    if num_voices is None:
        num_voices = len(KOKORO_VOICES)
    return num_voices * len(KOKORO_SPEED_VARIATIONS)
