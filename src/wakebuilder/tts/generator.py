"""
TTS Generator for WakeBuilder.

This module provides a wrapper around Piper TTS for generating synthetic
voice samples with various speed and pitch variations for data augmentation.
"""

import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..config import Config

# Piper imports - wrapped in try/except for graceful degradation
try:
    from piper.config import SynthesisConfig
    from piper.voice import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None  # type: ignore
    SynthesisConfig = None  # type: ignore


@dataclass
class VoiceInfo:
    """Information about a Piper voice model."""

    name: str
    onnx_path: Path
    json_path: Path
    sample_rate: int
    language: str

    @property
    def is_valid(self) -> bool:
        """Check if voice files exist and are valid."""
        return self.onnx_path.exists() and self.json_path.exists()


def list_available_voices(voices_dir: Optional[Path] = None) -> list[VoiceInfo]:
    """
    List all available Piper voice models.

    Args:
        voices_dir: Directory containing voice models. If None, uses config default.

    Returns:
        List of VoiceInfo objects for available voices.
    """
    if voices_dir is None:
        voices_dir = Path(Config.TTS_VOICES_DIR)

    voices = []

    # Look for .onnx files
    for onnx_path in sorted(voices_dir.glob("*.onnx")):
        json_path = onnx_path.with_suffix(".onnx.json")

        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            voice = VoiceInfo(
                name=onnx_path.stem,
                onnx_path=onnx_path,
                json_path=json_path,
                sample_rate=config.get("audio", {}).get("sample_rate", 22050),
                language=config.get("espeak", {}).get("voice", "en-us"),
            )
            voices.append(voice)

        except (json.JSONDecodeError, KeyError):
            continue

    return voices


class TTSGenerator:
    """
    Text-to-Speech generator using Piper TTS.

    This class provides methods to generate synthetic speech from text
    with support for multiple voices and speed/pitch variations.
    """

    def __init__(
        self,
        voices_dir: Optional[Path] = None,
        target_sample_rate: int = 16000,
    ):
        """
        Initialize the TTS generator.

        Args:
            voices_dir: Directory containing Piper voice models.
                       If None, uses config default.
            target_sample_rate: Target sample rate for output audio.
                               Audio will be resampled if needed.

        Raises:
            ImportError: If piper-tts is not installed.
            FileNotFoundError: If voices directory doesn't exist.
        """
        if not PIPER_AVAILABLE:
            raise ImportError(
                "piper-tts is not installed. " "Install with: uv add piper-tts"
            )

        self.voices_dir = (
            Path(voices_dir) if voices_dir else Path(Config.TTS_VOICES_DIR)
        )
        self.target_sample_rate = target_sample_rate

        if not self.voices_dir.exists():
            raise FileNotFoundError(
                f"Voices directory not found: {self.voices_dir}\n"
                f"Run: uv run python scripts/download_voices.py"
            )

        # Cache for loaded voice models
        self._voice_cache: dict[str, PiperVoice] = {}

        # Get available voices
        self._available_voices = list_available_voices(self.voices_dir)

        if not self._available_voices:
            raise FileNotFoundError(
                f"No voice models found in: {self.voices_dir}\n"
                f"Run: uv run python scripts/download_voices.py"
            )

    @property
    def available_voices(self) -> list[VoiceInfo]:
        """Get list of available voice models."""
        return self._available_voices

    @property
    def voice_names(self) -> list[str]:
        """Get list of available voice names."""
        return [v.name for v in self._available_voices]

    def _load_voice(self, voice_name: str) -> PiperVoice:
        """
        Load a voice model, using cache if available.

        Args:
            voice_name: Name of the voice to load.

        Returns:
            Loaded PiperVoice instance.

        Raises:
            ValueError: If voice is not found.
        """
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        # Find voice info
        voice_info = None
        for v in self._available_voices:
            if v.name == voice_name:
                voice_info = v
                break

        if voice_info is None:
            available = ", ".join(self.voice_names)
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")

        # Load voice
        voice = PiperVoice.load(str(voice_info.onnx_path))
        self._voice_cache[voice_name] = voice

        return voice

    def _get_voice_sample_rate(self, voice_name: str) -> int:
        """Get the native sample rate of a voice."""
        for v in self._available_voices:
            if v.name == voice_name:
                return v.sample_rate
        return 22050  # Default Piper sample rate

    def synthesize(
        self,
        text: str,
        voice_name: Optional[str] = None,
        length_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_name: Name of voice to use. If None, uses first available.
            length_scale: Speed multiplier (< 1.0 = faster, > 1.0 = slower).
                         Default 1.0 = normal speed.

        Returns:
            Tuple of (audio_samples, sample_rate).
            Audio samples are float32 normalized to [-1, 1].

        Raises:
            ValueError: If voice is not found or text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Use first voice if none specified
        if voice_name is None:
            voice_name = self.voice_names[0]

        # Load voice
        voice = self._load_voice(voice_name)
        native_sr = self._get_voice_sample_rate(voice_name)

        # Create synthesis config with length_scale
        syn_config = SynthesisConfig(length_scale=length_scale)

        # Synthesize to WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config,
            )

        # Read WAV data
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
            sample_width = wav_file.getsampwidth()

        # Convert to numpy array
        if sample_width == 2:
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int.astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_int = np.frombuffer(audio_bytes, dtype=np.int32)
            audio = audio_int.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Resample if needed
        if native_sr != self.target_sample_rate:
            audio = self._resample(audio, native_sr, self.target_sample_rate)

        return audio, self.target_sample_rate

    def synthesize_variations(
        self,
        text: str,
        voice_names: Optional[list[str]] = None,
        speed_variations: Optional[list[float]] = None,
    ) -> Iterator[tuple[np.ndarray, int, dict]]:
        """
        Generate multiple variations of synthesized speech.

        Args:
            text: Text to synthesize.
            voice_names: List of voices to use. If None, uses all available.
            speed_variations: List of speed multipliers. If None, uses config defaults.

        Yields:
            Tuples of (audio_samples, sample_rate, metadata).
            Metadata includes voice_name and speed.
        """
        if voice_names is None:
            voice_names = self.voice_names

        if speed_variations is None:
            speed_variations = Config.SPEED_VARIATIONS

        for voice_name in voice_names:
            for speed in speed_variations:
                try:
                    audio, sr = self.synthesize(
                        text,
                        voice_name=voice_name,
                        length_scale=1.0 / speed,  # length_scale is inverse of speed
                    )

                    metadata = {
                        "voice": voice_name,
                        "speed": speed,
                        "text": text,
                    }

                    yield audio, sr, metadata

                except Exception as e:
                    # Log error but continue with other variations
                    print(
                        f"Warning: Failed to synthesize with {voice_name} at {speed}x: {e}"
                    )
                    continue

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Uses librosa for high-quality resampling.

        Args:
            audio: Input audio samples.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio.
        """
        import librosa

        return librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Path,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio to a WAV file.

        Args:
            audio: Audio samples (float32, normalized to [-1, 1]).
            output_path: Path to save the WAV file.
            sample_rate: Sample rate. If None, uses target_sample_rate.
        """
        import soundfile as sf

        if sample_rate is None:
            sample_rate = self.target_sample_rate

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, audio, sample_rate)

    def generate_wake_word_samples(
        self,
        wake_word: str,
        output_dir: Optional[Path] = None,
        num_voices: Optional[int] = None,
        speed_variations: Optional[list[float]] = None,
    ) -> list[Path]:
        """
        Generate synthetic wake word samples for training.

        Args:
            wake_word: The wake word text to synthesize.
            output_dir: Directory to save samples. If None, uses temp directory.
            num_voices: Number of voices to use. If None, uses all available.
            speed_variations: Speed variations to apply. If None, uses config defaults.

        Returns:
            List of paths to generated audio files.
        """
        if output_dir is None:
            output_dir = Path(Config.TEMP_DIR) / "tts_samples"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select voices
        voices = self.voice_names
        if num_voices is not None and num_voices < len(voices):
            voices = voices[:num_voices]

        # Generate samples
        generated_files = []
        sample_idx = 0

        for audio, sr, metadata in self.synthesize_variations(
            wake_word,
            voice_names=voices,
            speed_variations=speed_variations,
        ):
            # Create filename
            voice = metadata["voice"].replace("-", "_")
            speed = f"{metadata['speed']:.1f}".replace(".", "p")
            filename = f"tts_{sample_idx:04d}_{voice}_{speed}x.wav"
            output_path = output_dir / filename

            # Save audio
            self.save_audio(audio, output_path, sr)
            generated_files.append(output_path)
            sample_idx += 1

        return generated_files

    def clear_cache(self) -> None:
        """Clear the voice model cache to free memory."""
        self._voice_cache.clear()

    def __repr__(self) -> str:
        return (
            f"TTSGenerator("
            f"voices_dir={self.voices_dir}, "
            f"num_voices={len(self._available_voices)}, "
            f"target_sr={self.target_sample_rate})"
        )
