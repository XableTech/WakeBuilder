"""
Audio preprocessing for WakeBuilder.

This module provides functions to convert raw audio into mel spectrograms
that can be processed by the base speech embedding model.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np

from ..config import Config


def load_audio(
    audio_path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file from disk.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate. If None, uses original sample rate.
        mono: If True, convert to mono
        duration: Duration to load in seconds. If None, loads entire file.
        offset: Start reading after this time (in seconds)

    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: Audio samples as numpy array
        - sample_rate: Sample rate of the audio

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is invalid
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Load audio using librosa
        audio, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            mono=mono,
            duration=duration,
            offset=offset,
        )

        return audio, sr

    except Exception as e:
        raise ValueError(f"Failed to load audio file {audio_path}: {e}") from e


def normalize_audio(
    audio: np.ndarray,
    target_level: float = -20.0,
    method: str = "peak",
) -> np.ndarray:
    """
    Normalize audio for consistent input to the model.
    
    CRITICAL: This normalization MUST match between training and inference!
    We use peak normalization (scale to max=0.9) for consistency.

    Args:
        audio: Input audio samples
        target_level: Target dB level (only used if method='rms')
        method: Normalization method:
            - 'peak': Scale so max absolute value = 0.9 (RECOMMENDED)
            - 'rms': Scale to target RMS level in dB

    Returns:
        Normalized audio
    """
    if method == "peak":
        # Peak normalization - consistent and simple
        # This is what we use during data augmentation, so inference must match
        max_val = np.abs(audio).max()
        if max_val > 0.01:  # Avoid division by near-zero
            normalized = audio / max_val * 0.9
        else:
            normalized = audio
        return normalized.astype(np.float32)
    
    # RMS normalization (legacy, not recommended)
    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return audio

    # Calculate scaling factor
    current_db = 20 * np.log10(rms)
    scale = 10 ** ((target_level - current_db) / 20)

    # Apply scaling
    normalized = audio * scale

    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized.astype(np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    window: str = "hann",
    center: bool = True,
    power: float = 2.0,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio.

    Args:
        audio: Input audio samples
        sample_rate: Sample rate of audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (None = sample_rate / 2)
        window: Window function
        center: If True, center frames
        power: Exponent for magnitude spectrogram (1.0 = energy, 2.0 = power)

    Returns:
        Mel spectrogram of shape (n_mels, time)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=window,
        center=center,
        power=power,
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


class AudioPreprocessor:
    """
    Audio preprocessor for WakeBuilder.

    This class handles all audio preprocessing steps required to convert
    raw audio into mel spectrograms suitable for the base embedding model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        target_length: Optional[int] = 96,
        normalize: bool = True,
    ):
        """
        Initialize the audio preprocessor.

        Args:
            sample_rate: Target sample rate (default: 16000 Hz)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length in samples (default: 160 = 10ms at 16kHz)
            n_mels: Number of mel bands (default: 80)
            fmin: Minimum frequency (default: 0 Hz)
            fmax: Maximum frequency (default: 8000 Hz)
            target_length: Target time dimension (default: 96 frames)
            normalize: Whether to normalize audio (default: True)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.target_length = target_length
        self.normalize = normalize

    def process_audio(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Process audio into mel spectrogram.
        
        CRITICAL: This preprocessing MUST be identical for training and inference!
        The pipeline is:
        1. Resample to target sample rate (16kHz)
        2. Peak normalize audio to max=0.9
        3. Compute mel spectrogram
        4. Convert to log scale (dB)
        5. Transpose to (time, freq)
        6. Pad/trim to target length
        7. Standardize spectrogram (zero mean, unit variance)

        Args:
            audio: Either audio samples (numpy array) or path to audio file
            sample_rate: Sample rate of input audio (required if audio is array)

        Returns:
            Mel spectrogram of shape (time, freq) or (n_mels, time) depending on target_length

        Raises:
            ValueError: If inputs are invalid
        """
        # Load audio if path is provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = load_audio(audio, sample_rate=self.sample_rate)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when audio is array")

            # Resample if needed
            if sample_rate != self.sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate,
                )
                sample_rate = self.sample_rate

        # Normalize audio using PEAK normalization
        # CRITICAL: This must match the normalization used during data augmentation
        if self.normalize:
            audio = normalize_audio(audio, method="peak")

        # Compute mel spectrogram
        mel_spec = compute_mel_spectrogram(
            audio=audio,
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Transpose to (time, freq) for model input
        mel_spec = mel_spec.T

        # Pad or trim to target length if specified
        if self.target_length is not None:
            mel_spec = self._adjust_length(mel_spec, self.target_length)

        # CRITICAL: Standardize spectrogram (zero mean, unit variance)
        # This makes the model invariant to overall volume and recording conditions
        # Using both mean and std normalization is more robust than mean-only
        mel_mean = mel_spec.mean()
        mel_std = mel_spec.std()
        if mel_std > 1e-6:  # Avoid division by zero
            mel_spec = (mel_spec - mel_mean) / mel_std
        else:
            mel_spec = mel_spec - mel_mean

        return mel_spec.astype(np.float32)

    def _adjust_length(
        self,
        mel_spec: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Adjust mel spectrogram to target length by padding or trimming.

        Args:
            mel_spec: Input mel spectrogram of shape (time, freq)
            target_length: Target time dimension

        Returns:
            Adjusted mel spectrogram of shape (target_length, freq)
        """
        current_length = mel_spec.shape[0]

        if current_length < target_length:
            # Pad with zeros
            pad_width = target_length - current_length
            mel_spec = np.pad(
                mel_spec,
                ((0, pad_width), (0, 0)),
                mode="constant",
                constant_values=mel_spec.min(),
            )
        elif current_length > target_length:
            # Trim from center
            start = (current_length - target_length) // 2
            mel_spec = mel_spec[start : start + target_length]

        return mel_spec

    def process_batch(
        self,
        audio_list: list,
        sample_rates: Optional[list] = None,
    ) -> np.ndarray:
        """
        Process a batch of audio samples.

        Args:
            audio_list: List of audio samples or paths
            sample_rates: List of sample rates (if audio_list contains arrays)

        Returns:
            Batch of mel spectrograms of shape (batch, time, freq)
        """
        if sample_rates is None:
            sample_rates = [None] * len(audio_list)

        mel_specs = []
        for audio, sr in zip(audio_list, sample_rates):
            mel_spec = self.process_audio(audio, sample_rate=sr)
            mel_specs.append(mel_spec)

        return np.stack(mel_specs, axis=0)

    @classmethod
    def from_config(cls, config: Optional[Config] = None) -> "AudioPreprocessor":
        """
        Create preprocessor from configuration.

        Args:
            config: Configuration object. If None, uses default config.

        Returns:
            AudioPreprocessor instance
        """
        if config is None:
            config = Config()

        return cls(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
            target_length=config.MEL_SPEC_TIME_DIM,
            normalize=True,
        )
