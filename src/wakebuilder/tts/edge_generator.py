"""
Edge TTS Generator for WakeBuilder.

This module provides text-to-speech functionality using Microsoft Edge TTS
with support for 400+ voices across 100+ languages.

Edge TTS is free and provides high-quality neural voices.

Package: edge-tts
"""

import asyncio
import gc
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Check if Edge TTS is available
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None


# Speed variations for augmentation
EDGE_SPEED_VARIATIONS = [1.0, 1.5]

# Volume variations in dB
EDGE_VOLUME_VARIATIONS = [-6, -3, 0, 3]

# Path to voice list file
EDGE_VOICE_LIST_FILE = Path(__file__).parent.parent.parent.parent / "scripts" / "list of voices available in Edge TTS.txt"


def parse_edge_voice_file(filepath: Path) -> list[dict]:
    """
    Parse the voice list file to extract voice information.
    
    Args:
        filepath: Path to the voice list text file.
    
    Returns:
        List of voice dictionaries with Name, ShortName, Gender, Locale.
    """
    voices = []
    current_voice = {}
    
    if not filepath.exists():
        return voices
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_voice:
                    voices.append(current_voice)
                    current_voice = {}
                continue
            
            if line.startswith("Name:"):
                current_voice["Name"] = line[5:].strip()
            elif line.startswith("ShortName:"):
                current_voice["ShortName"] = line[10:].strip()
            elif line.startswith("Gender:"):
                current_voice["Gender"] = line[7:].strip()
            elif line.startswith("Locale:"):
                current_voice["Locale"] = line[7:].strip()
    
    # Don't forget the last voice
    if current_voice:
        voices.append(current_voice)
    
    return voices


def load_all_edge_voices(voice_file: Optional[Path] = None) -> list[str]:
    """
    Load all voice ShortNames from the voice list file.
    
    Args:
        voice_file: Path to voice list file. If None, uses default.
    
    Returns:
        List of voice ShortNames (e.g., 'en-US-JennyNeural').
    """
    if voice_file is None:
        voice_file = EDGE_VOICE_LIST_FILE
    
    voices = parse_edge_voice_file(voice_file)
    return [v["ShortName"] for v in voices if "ShortName" in v]


# Load all voices from file at module load time
_ALL_EDGE_VOICES = load_all_edge_voices()

# Fallback curated list if file not found
EDGE_ENGLISH_VOICES = [
    "en-US-AriaNeural", "en-US-JennyNeural", "en-US-GuyNeural",
    "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural",
]

EDGE_EUROPEAN_VOICES = [
    "de-DE-KatjaNeural", "fr-FR-DeniseNeural", "es-ES-ElviraNeural",
    "it-IT-ElsaNeural", "nl-NL-ColetteNeural", "pt-PT-RaquelNeural",
]

# Use all voices from file, or fallback to curated list
EDGE_DEFAULT_VOICES = _ALL_EDGE_VOICES if _ALL_EDGE_VOICES else (EDGE_ENGLISH_VOICES + EDGE_EUROPEAN_VOICES)


def get_edge_sample_count(voices: Optional[list[str]] = None) -> int:
    """
    Calculate expected number of samples from Edge TTS.
    
    Args:
        voices: List of voice IDs. If None, uses default voices.
    
    Returns:
        Estimated sample count.
    """
    if voices is None:
        voices = EDGE_DEFAULT_VOICES
    
    return len(voices) * len(EDGE_SPEED_VARIATIONS) * len(EDGE_VOLUME_VARIATIONS)


class EdgeTTSGenerator:
    """
    Text-to-Speech generator using Microsoft Edge TTS.
    
    This class provides methods to generate high-quality synthetic speech
    using Microsoft's neural TTS voices. Edge TTS is free and provides
    400+ voices across 100+ languages.
    
    Key features:
    - 400+ neural voices
    - Multi-lingual support
    - Speed variations via rate parameter
    - No API key required
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        voices: Optional[list[str]] = None,
    ):
        """
        Initialize the Edge TTS generator.
        
        Args:
            target_sample_rate: Target sample rate for output audio (default: 16000 Hz)
            voices: List of voice IDs to use. If None, uses curated English + European voices.
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError(
                "Edge TTS is not installed. Install with: uv add edge-tts"
            )
        
        self.target_sample_rate = target_sample_rate
        
        # Voices to use
        if voices is None:
            self._voices = EDGE_DEFAULT_VOICES.copy()
        else:
            self._voices = voices
        
        # Temp directory for audio files
        self._temp_dir = Path(tempfile.gettempdir()) / "edge_tts_cache"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_available(self) -> bool:
        """Check if Edge TTS is available."""
        return EDGE_TTS_AVAILABLE
    
    @property
    def voice_ids(self) -> list[str]:
        """Get list of voice IDs."""
        return self._voices
    
    @property
    def num_voices(self) -> int:
        """Get number of available voices."""
        return len(self._voices)
    
    async def _synthesize_async(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
    ) -> np.ndarray:
        """
        Async synthesis using Edge TTS.
        
        Args:
            text: Text to synthesize.
            voice: Voice ID.
            rate: Rate adjustment (e.g., "+50%" for 1.5x speed).
        
        Returns:
            Audio samples as numpy array.
        """
        import soundfile as sf
        
        # Generate unique temp file
        temp_file = self._temp_dir / f"edge_{hash(text + voice + rate)}.mp3"
        
        try:
            # Create communicate object with rate
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(temp_file))
            
            # Load audio
            audio, sr = sf.read(temp_file)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != self.target_sample_rate:
                audio = self._resample(audio, sr, self.target_sample_rate)
            
            return audio.astype(np.float32)
            
        finally:
            # Clean up temp file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
    
    def synthesize(
        self,
        text: str,
        voice: str = "en-US-JennyNeural",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            voice: Voice ID (e.g., 'en-US-JennyNeural').
            speed: Speed multiplier (1.0 = normal, 1.5 = 50% faster).
        
        Returns:
            Tuple of (audio_samples, sample_rate).
            Audio samples are float32 normalized to [-1, 1].
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Convert speed to rate string
        rate_percent = int((speed - 1.0) * 100)
        rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
        
        # Run async synthesis
        audio = asyncio.run(self._synthesize_async(text, voice, rate))
        
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
        voices: Optional[list[str]] = None,
        speeds: Optional[list[float]] = None,
    ) -> Iterator[tuple[np.ndarray, int, dict]]:
        """
        Generate audio for all voices with speed variations.
        
        Args:
            text: Text to synthesize.
            voices: List of voice IDs to use. If None, uses configured voices.
            speeds: List of speed multipliers. If None, uses defaults.
        
        Yields:
            Tuples of (audio_samples, sample_rate, metadata).
        """
        if voices is None:
            voices = self._voices
        
        if speeds is None:
            speeds = EDGE_SPEED_VARIATIONS
        
        for voice in voices:
            # Extract locale and name from voice ID
            parts = voice.split("-")
            if len(parts) >= 3:
                locale = f"{parts[0]}-{parts[1]}"
                name = parts[2].replace("Neural", "")
            else:
                locale = "unknown"
                name = voice
            
            for speed in speeds:
                try:
                    audio, sr = self.synthesize(text, voice=voice, speed=speed)
                    
                    metadata = {
                        "voice_id": voice,
                        "voice_name": name,
                        "locale": locale,
                        "speed": speed,
                        "tts_engine": "edge",
                    }
                    
                    yield audio, sr, metadata
                    
                except Exception as e:
                    # Skip failed syntheses
                    continue
    
    def cleanup(self) -> None:
        """
        Clean up temporary files.
        """
        # Clean up temp directory
        if self._temp_dir.exists():
            for f in self._temp_dir.glob("edge_*.mp3"):
                try:
                    f.unlink()
                except Exception:
                    pass
        
        gc.collect()
        print("Edge TTS: Cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass
