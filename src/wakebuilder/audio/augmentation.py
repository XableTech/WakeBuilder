"""
Data augmentation for WakeBuilder.

This module provides audio augmentation functions for creating diverse
training data from a small set of wake word recordings.

Augmentation techniques include:
- TTS-based synthetic sample generation
- Background noise mixing at various SNR levels
- Volume/gain variations
- Speed/tempo variations (via resampling)
- Pitch shifting
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import librosa
import numpy as np
import soundfile as sf

from ..config import Config
from ..tts import TTSGenerator


@dataclass
class AugmentedSample:
    """Container for an augmented audio sample with metadata."""

    audio: np.ndarray
    sample_rate: int
    label: int  # 1 = positive (wake word), 0 = negative
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate


class NoiseLoader:
    """Loader for background noise samples."""

    def __init__(self, noise_dir: Optional[Path] = None):
        """Initialize the noise loader."""
        self.noise_dir = (
            Path(noise_dir) if noise_dir else Path(Config.DATA_DIR) / "noise"
        )
        self._noise_samples: list[tuple[np.ndarray, int]] = []
        self._noise_names: list[str] = []

        if self.noise_dir.exists():
            self._load_noise_samples()

    def _load_noise_samples(self) -> None:
        """Load all noise samples from the noise directory."""
        for wav_file in sorted(self.noise_dir.glob("*.wav")):
            try:
                audio, sr = sf.read(wav_file)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                self._noise_samples.append((audio.astype(np.float32), sr))
                self._noise_names.append(wav_file.stem)
            except Exception:
                continue

    @property
    def num_samples(self) -> int:
        """Number of loaded noise samples."""
        return len(self._noise_samples)

    @property
    def noise_names(self) -> list[str]:
        """Names of loaded noise samples."""
        return self._noise_names

    def get_random_noise(self, duration: float, target_sr: int = 16000) -> np.ndarray:
        """Get a random noise segment of specified duration."""
        if not self._noise_samples:
            raise RuntimeError(
                f"No noise samples loaded from {self.noise_dir}. "
                "Run: uv run python scripts/prepare_noise.py"
            )

        audio, sr = random.choice(self._noise_samples)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        required_length = int(duration * target_sr)

        if len(audio) >= required_length:
            start = random.randint(0, len(audio) - required_length)
            return audio[start : start + required_length]
        else:
            repeats = (required_length // len(audio)) + 1
            return np.tile(audio, repeats)[:required_length]


def mix_audio_with_noise(
    audio: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix audio with noise at a specified SNR level."""
    if len(noise) != len(audio):
        if len(noise) > len(audio):
            noise = noise[: len(audio)]
        else:
            noise = np.pad(noise, (0, len(audio) - len(noise)))

    signal_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return audio
    if signal_power == 0:
        return noise

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    mixed = audio + noise * noise_scale
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed


def apply_volume_change(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply volume/gain change to audio."""
    gain_linear = 10 ** (gain_db / 20)
    return np.clip(audio * gain_linear, -1.0, 1.0)


def apply_speed_change(
    audio: np.ndarray, sample_rate: int, speed_factor: float
) -> np.ndarray:
    """Apply speed change to audio via resampling."""
    if speed_factor == 1.0:
        return audio

    intermediate_sr = int(sample_rate / speed_factor)
    resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=intermediate_sr)
    return librosa.resample(resampled, orig_sr=intermediate_sr, target_sr=sample_rate)


def apply_pitch_shift(
    audio: np.ndarray, sample_rate: int, semitones: float
) -> np.ndarray:
    """Apply pitch shift to audio."""
    if semitones == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)


def pad_or_trim_audio(
    audio: np.ndarray,
    target_length: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Pad or trim audio to target length."""
    if len(audio) == target_length:
        return audio
    elif len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        return audio[start : start + target_length]
    else:
        pad_total = target_length - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Use keyword argument for mode to satisfy type checker
        return np.pad(audio, (pad_left, pad_right), mode=pad_mode)  # type: ignore[call-overload]


class DataAugmenter:
    """Data augmentation pipeline for wake word training."""

    def __init__(
        self,
        tts_generator: Optional[TTSGenerator] = None,
        noise_loader: Optional[NoiseLoader] = None,
        target_sample_rate: int = 16000,
        target_duration: float = 1.0,
    ):
        """Initialize the data augmenter."""
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_sample_rate * target_duration)

        self._tts: Optional[TTSGenerator] = None
        if tts_generator is not None:
            self._tts = tts_generator
        else:
            try:
                self._tts = TTSGenerator(target_sample_rate=target_sample_rate)
            except (ImportError, FileNotFoundError):
                self._tts = None

        self._noise = noise_loader if noise_loader else NoiseLoader()

        self.speed_variations = Config.SPEED_VARIATIONS
        self.pitch_shifts = Config.PITCH_SHIFTS
        self.noise_levels = Config.NOISE_LEVELS

    @property
    def tts_available(self) -> bool:
        """Check if TTS is available."""
        return self._tts is not None

    @property
    def noise_available(self) -> bool:
        """Check if noise samples are available."""
        return self._noise.num_samples > 0

    def augment_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        add_noise: bool = True,
        change_speed: bool = True,
        change_pitch: bool = True,
    ) -> Iterator[AugmentedSample]:
        """Generate augmented versions of a single audio sample."""
        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.target_sample_rate
            )

        speed_factors = self.speed_variations if change_speed else [1.0]
        pitch_semitones = self.pitch_shifts if change_pitch else [0]

        for speed in speed_factors:
            for pitch in pitch_semitones:
                aug_audio = apply_speed_change(audio, self.target_sample_rate, speed)

                if pitch != 0:
                    aug_audio = apply_pitch_shift(
                        aug_audio, self.target_sample_rate, pitch
                    )

                aug_audio = pad_or_trim_audio(aug_audio, self.target_length)

                # Normalize to consistent level before applying gain
                max_val = np.abs(aug_audio).max()
                if max_val > 0.01:
                    aug_audio = aug_audio / max_val * 0.9

                gain_db = random.uniform(-6, 6)
                aug_audio = apply_volume_change(aug_audio, gain_db)

                yield AugmentedSample(
                    audio=aug_audio,
                    sample_rate=self.target_sample_rate,
                    label=1,
                    metadata={"speed": speed, "pitch": pitch, "gain_db": gain_db},
                )

                if add_noise and self.noise_available:
                    for snr_db in self.noise_levels:
                        noise = self._noise.get_random_noise(
                            self.target_duration, self.target_sample_rate
                        )
                        noisy = mix_audio_with_noise(aug_audio, noise, snr_db)

                        yield AugmentedSample(
                            audio=noisy,
                            sample_rate=self.target_sample_rate,
                            label=1,
                            metadata={
                                "speed": speed,
                                "pitch": pitch,
                                "snr_db": snr_db,
                            },
                        )

    def generate_tts_samples(
        self, text: str, num_voices: Optional[int] = None, add_noise: bool = True
    ) -> Iterator[AugmentedSample]:
        """Generate synthetic samples using TTS."""
        if not self.tts_available or self._tts is None:
            return

        voices = self._tts.voice_names
        if num_voices is not None and num_voices < len(voices):
            voices = voices[:num_voices]

        for voice_name in voices:
            for speed in self.speed_variations:
                try:
                    audio, _ = self._tts.synthesize(
                        text, voice_name=voice_name, length_scale=1.0 / speed
                    )
                    audio = pad_or_trim_audio(audio, self.target_length)

                    yield AugmentedSample(
                        audio=audio,
                        sample_rate=self.target_sample_rate,
                        label=1,
                        metadata={"source": "tts", "voice": voice_name, "speed": speed},
                    )

                    if add_noise and self.noise_available:
                        for snr_db in self.noise_levels:
                            noise = self._noise.get_random_noise(
                                self.target_duration, self.target_sample_rate
                            )
                            noisy = mix_audio_with_noise(audio, noise, snr_db)

                            yield AugmentedSample(
                                audio=noisy,
                                sample_rate=self.target_sample_rate,
                                label=1,
                                metadata={
                                    "source": "tts",
                                    "voice": voice_name,
                                    "snr_db": snr_db,
                                },
                            )
                except Exception as e:
                    print(f"Warning: TTS failed for {voice_name}: {e}")
                    continue
