"""
Real data loader for WakeBuilder.

This module provides functions to load real audio data from directories
for training wake word models with actual recordings instead of synthetic data.

Supports:
- Loading negative samples from data/negative/ directory
- Chunking long audio files into 1-second segments
- Converting MP3/FLAC/OGG to WAV format on-the-fly
- Deduplication to avoid training on similar audio
- Caching processed chunks for fast loading
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Iterator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import soundfile as sf

from ..config import Config


# Cache directories
CACHE_DIR = Path(Config.DATA_DIR) / "cache" / "negative_chunks"
SPEC_CACHE_DIR = Path(Config.DATA_DIR) / "cache" / "negative_spectrograms"


def compute_audio_hash(audio: np.ndarray, precision: int = 3) -> str:
    """
    Compute a hash of audio content for deduplication.
    
    Uses a combination of audio statistics to create a fingerprint.
    """
    # Round to reduce floating point noise
    rounded = np.round(audio, precision)
    
    # Compute statistics-based fingerprint
    stats = np.array([
        rounded.mean(),
        rounded.std(),
        rounded.min(),
        rounded.max(),
        np.percentile(rounded, 25),
        np.percentile(rounded, 75),
    ])
    
    return hashlib.md5(stats.tobytes()).hexdigest()[:16]


def load_audio_file(
    file_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file, converting to target sample rate and mono.
    
    Supports WAV, MP3, FLAC, OGG formats.
    """
    try:
        # Use librosa for format conversion
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio.astype(np.float32), sr
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}") from e


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float = 1.0,
    overlap: float = 0.5,
    min_energy: float = 0.001,
) -> Iterator[np.ndarray]:
    """
    Split audio into overlapping chunks.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks (0-1)
        min_energy: Minimum RMS energy to include chunk (filters silence)
    
    Yields:
        Audio chunks of specified duration
    """
    chunk_samples = int(chunk_duration * sample_rate)
    hop_samples = int(chunk_samples * (1 - overlap))
    
    # Ensure we have at least one chunk
    if len(audio) < chunk_samples:
        # Pad short audio
        padded = np.zeros(chunk_samples, dtype=audio.dtype)
        padded[:len(audio)] = audio
        yield padded
        return
    
    # Generate overlapping chunks
    for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
        chunk = audio[start:start + chunk_samples]
        
        # Filter out silent chunks
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms >= min_energy:
            yield chunk


class RealNegativeDataLoader:
    """
    Loader for real negative training data.
    
    Loads audio files from the data/negative/ directory structure:
    - data/negative/music/     - Music files (MP3/WAV)
    - data/negative/speech/    - Speech recordings (MP3/WAV)
    - data/negative/noise/     - Noise samples (WAV)
    """
    
    def __init__(
        self,
        negative_dir: Optional[Path] = None,
        target_sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        chunk_overlap: float = 0.5,
        max_samples_per_file: int = 10,
    ):
        """
        Initialize the loader.
        
        Args:
            negative_dir: Directory containing negative samples
            target_sample_rate: Target sample rate for all audio
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks (0-1)
            max_samples_per_file: Maximum chunks to extract per file
        """
        self.negative_dir = negative_dir or Path(Config.DATA_DIR) / "negative"
        self.target_sr = target_sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.max_samples_per_file = max_samples_per_file
        
        self._seen_hashes: set[str] = set()
    
    @property
    def available(self) -> bool:
        """Check if negative data directory exists and has files."""
        if not self.negative_dir.exists():
            return False
        
        # Check for any audio files
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            if list(self.negative_dir.rglob(ext)):
                return True
        return False
    
    def get_file_count(self) -> dict[str, int]:
        """Get count of audio files by type."""
        counts = {"wav": 0, "mp3": 0, "flac": 0, "ogg": 0, "total": 0}
        
        if not self.negative_dir.exists():
            return counts
        
        for ext in ["wav", "mp3", "flac", "ogg"]:
            # Use set to avoid double-counting on Windows
            files = set(self.negative_dir.rglob(f"*.{ext}"))
            files.update(self.negative_dir.rglob(f"*.{ext.upper()}"))
            counts[ext] = len(files)
            counts["total"] += len(files)
        
        return counts
    
    def get_cache_info(self) -> dict:
        """Get information about cached chunks."""
        cache_dir = CACHE_DIR
        metadata_file = cache_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {"cached": False, "chunk_count": 0, "source_files": 0}
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Count actual chunk files
            chunk_files = list(cache_dir.glob("chunk_*.npy"))
            
            return {
                "cached": True,
                "chunk_count": len(chunk_files),
                "source_files": metadata.get("source_files", 0),
                "created_at": metadata.get("created_at", "unknown"),
                "sample_rate": metadata.get("sample_rate", 16000),
            }
        except Exception:
            return {"cached": False, "chunk_count": 0, "source_files": 0}
    
    def build_cache(
        self,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Pre-process all audio files and cache chunks to disk.
        
        Args:
            max_workers: Number of parallel workers
            progress_callback: Optional callback(processed, total) for progress
        
        Returns:
            Total number of chunks cached
        """
        import time
        
        cache_dir = CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg"]:
            audio_files.extend(self.negative_dir.rglob(ext))
        
        # Deduplicate (Windows case-insensitivity)
        audio_files = list(set(audio_files))
        
        if not audio_files:
            print("No audio files found to cache")
            return 0
        
        print(f"Building cache for {len(audio_files)} audio files...")
        print(f"Cache directory: {cache_dir}")
        
        chunk_index = 0
        processed_files = 0
        seen_hashes: set[str] = set()
        
        def process_file(file_path: Path) -> list[tuple[np.ndarray, str]]:
            """Process a single file and return chunks."""
            chunks = []
            try:
                audio, sr = load_audio_file(file_path, self.target_sr)
                
                # Normalize
                max_val = np.abs(audio).max()
                if max_val > 0.01:
                    audio = audio / max_val * 0.9
                
                # Chunk the audio
                chunk_count = 0
                for chunk in chunk_audio(
                    audio, sr, self.chunk_duration, self.chunk_overlap
                ):
                    if chunk_count >= self.max_samples_per_file:
                        break
                    
                    chunk_hash = compute_audio_hash(chunk)
                    chunks.append((chunk, chunk_hash))
                    chunk_count += 1
                    
            except Exception as e:
                pass  # Skip failed files silently
            
            return chunks
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in audio_files}
            
            for future in as_completed(futures):
                processed_files += 1
                
                if progress_callback:
                    progress_callback(processed_files, len(audio_files))
                
                if processed_files % 500 == 0:
                    print(f"  Processed {processed_files}/{len(audio_files)} files, {chunk_index} chunks...")
                
                try:
                    chunks = future.result()
                    for chunk, chunk_hash in chunks:
                        # Deduplicate
                        if chunk_hash in seen_hashes:
                            continue
                        seen_hashes.add(chunk_hash)
                        
                        # Save chunk
                        chunk_file = cache_dir / f"chunk_{chunk_index:06d}.npy"
                        np.save(chunk_file, chunk)
                        chunk_index += 1
                        
                except Exception:
                    pass
        
        # Save metadata
        metadata = {
            "source_files": len(audio_files),
            "chunk_count": chunk_index,
            "sample_rate": self.target_sr,
            "chunk_duration": self.chunk_duration,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Cached {chunk_index} chunks from {len(audio_files)} files")
        return chunk_index
    
    def clear_cache(self) -> None:
        """Delete all cached chunks."""
        cache_dir = CACHE_DIR
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")
    
    def load_from_cache(
        self,
        max_samples: int = 0,
        shuffle: bool = True,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """
        Load chunks from cache (fast).
        
        Args:
            max_samples: Maximum samples to load (0 = no limit)
            shuffle: Shuffle chunks before loading
        
        Yields:
            Tuple of (audio_chunk, metadata)
        """
        cache_dir = CACHE_DIR
        
        if not cache_dir.exists():
            return
        
        chunk_files = list(cache_dir.glob("chunk_*.npy"))
        
        if not chunk_files:
            return
        
        if shuffle:
            random.shuffle(chunk_files)
        
        for i, chunk_file in enumerate(chunk_files):
            if max_samples > 0 and i >= max_samples:
                break
            
            try:
                chunk = np.load(chunk_file)
                metadata = {
                    "source": "cached_negative",
                    "file": chunk_file.name,
                    "category": "cached",
                }
                yield chunk, metadata
            except Exception:
                continue
    
    # ========================================================================
    # Spectrogram Cache (fastest - skips mel computation)
    # ========================================================================
    
    def get_spectrogram_cache_info(self) -> dict:
        """Get information about cached spectrograms."""
        cache_dir = SPEC_CACHE_DIR
        metadata_file = cache_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {"cached": False, "count": 0}
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return {
                "cached": True,
                "count": metadata.get("count", 0),
                "created_at": metadata.get("created_at", "unknown"),
            }
        except Exception:
            return {"cached": False, "count": 0}
    
    def build_spectrogram_cache(
        self,
        preprocessor,
        max_workers: int = 4,
        progress_callback: callable = None,
    ) -> int:
        """
        Build spectrogram cache from audio cache.
        
        Args:
            preprocessor: MelSpectrogramPreprocessor instance
            max_workers: Number of parallel workers
            progress_callback: Optional callback(processed, total) for progress updates
        
        Returns:
            Number of spectrograms cached
        """
        import time
        
        # First check if audio cache exists
        cache_info = self.get_cache_info()
        if not cache_info["cached"] or cache_info["chunk_count"] == 0:
            print("Audio cache not found. Building audio cache first...")
            self.build_cache(max_workers=max_workers)
            cache_info = self.get_cache_info()
        
        spec_cache_dir = SPEC_CACHE_DIR
        spec_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all audio chunks and compute spectrograms
        chunk_files = list(CACHE_DIR.glob("chunk_*.npy"))
        total_chunks = len(chunk_files)
        print(f"Building spectrogram cache for {total_chunks:,} audio chunks...")
        
        specs = []
        for i, chunk_file in enumerate(chunk_files):
            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i:,}/{total_chunks:,} chunks...")
            
            # Call progress callback every 100 chunks for responsive updates
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_chunks)
            
            try:
                audio = np.load(chunk_file)
                spec = preprocessor.process_audio(audio, 16000)
                specs.append(spec)
            except Exception:
                continue
        
        # Final progress update
        if progress_callback:
            progress_callback(total_chunks, total_chunks)
        
        # Save as single numpy file (much faster to load)
        specs_array = np.stack(specs, axis=0)
        np.save(spec_cache_dir / "spectrograms.npy", specs_array)
        
        # Save metadata
        metadata = {
            "count": len(specs),
            "shape": list(specs_array.shape),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(spec_cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Cached {len(specs):,} spectrograms")
        return len(specs)
    
    def load_spectrograms_from_cache(
        self,
        max_samples: int = 0,
        shuffle: bool = True,
    ) -> list[np.ndarray]:
        """
        Load spectrograms from cache (instant).
        
        Args:
            max_samples: Maximum samples to load (0 = no limit)
            shuffle: Shuffle before returning
        
        Returns:
            List of spectrogram arrays
        """
        spec_file = SPEC_CACHE_DIR / "spectrograms.npy"
        
        if not spec_file.exists():
            return []
        
        # Load all spectrograms at once (fast!)
        specs_array = np.load(spec_file)
        
        # Convert to list
        indices = list(range(len(specs_array)))
        
        if shuffle:
            random.shuffle(indices)
        
        if max_samples > 0:
            indices = indices[:max_samples]
        
        return [specs_array[i] for i in indices]
    
    def clear_spectrogram_cache(self) -> None:
        """Delete cached spectrograms."""
        if SPEC_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(SPEC_CACHE_DIR)
            print(f"Cleared spectrogram cache: {SPEC_CACHE_DIR}")
    
    def load_samples(
        self,
        max_samples: int = 0,
        categories: Optional[list[str]] = None,
        shuffle: bool = True,
        deduplicate: bool = True,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """
        Load negative samples from the directory.
        
        Args:
            max_samples: Maximum number of samples to load (0 = no limit)
            categories: Specific subdirectories to load from (e.g., ["music", "speech"])
            shuffle: Shuffle files before loading
            deduplicate: Skip similar audio chunks
        
        Yields:
            Tuple of (audio_chunk, metadata)
        """
        if not self.negative_dir.exists():
            print(f"Warning: Negative data directory not found: {self.negative_dir}")
            return
        
        # Find all audio files
        audio_files = []
        search_dirs = [self.negative_dir]
        
        if categories:
            search_dirs = [self.negative_dir / cat for cat in categories if (self.negative_dir / cat).exists()]
        
        for search_dir in search_dirs:
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.WAV", "*.MP3"]:
                audio_files.extend(search_dir.rglob(ext))
        
        if not audio_files:
            print(f"Warning: No audio files found in {self.negative_dir}")
            return
        
        if shuffle:
            random.shuffle(audio_files)
        
        sample_count = 0
        self._seen_hashes.clear()
        
        for file_path in audio_files:
            # 0 = no limit
            if max_samples > 0 and sample_count >= max_samples:
                break
            
            try:
                # Load audio file
                audio, sr = load_audio_file(file_path, self.target_sr)
                
                # Normalize
                max_val = np.abs(audio).max()
                if max_val > 0.01:
                    audio = audio / max_val * 0.9
                
                # Get category from directory structure
                try:
                    category = file_path.parent.relative_to(self.negative_dir).parts[0]
                except (ValueError, IndexError):
                    category = "unknown"
                
                # Chunk the audio
                chunks_from_file = 0
                for chunk in chunk_audio(
                    audio, sr, self.chunk_duration, self.chunk_overlap
                ):
                    if chunks_from_file >= self.max_samples_per_file:
                        break
                    
                    if sample_count >= max_samples:
                        break
                    
                    # Deduplicate
                    if deduplicate:
                        chunk_hash = compute_audio_hash(chunk)
                        if chunk_hash in self._seen_hashes:
                            continue
                        self._seen_hashes.add(chunk_hash)
                    
                    metadata = {
                        "source": "real_negative",
                        "file": file_path.name,
                        "category": category,
                    }
                    
                    yield chunk, metadata
                    sample_count += 1
                    chunks_from_file += 1
                    
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        print(f"  Loaded {sample_count} real negative samples from {len(audio_files)} files")


class MassivePositiveAugmenter:
    """
    Generate a large number of positive samples from user recordings.
    
    Uses aggressive augmentation to create 6000+ unique samples from
    just 3-5 user recordings, while avoiding duplicates and ensuring diversity.
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_duration: float = 1.0,
    ):
        self.target_sr = target_sample_rate
        self.target_duration = target_duration
        self.target_length = int(target_sample_rate * target_duration)
        
        # Augmentation parameters for massive generation
        # More variations = more unique samples
        self.speed_variations = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        self.pitch_shifts = [-3, -2, -1, 0, 1, 2, 3]
        self.volume_variations = [-9, -6, -3, 0, 3, 6]
        self.noise_snr_levels = [5, 10, 15, 20, 25, 30]  # dB
        
        # Time shift variations (in samples)
        self.time_shifts = [-800, -400, -200, 0, 200, 400, 800]
        
        self._seen_hashes: set[str] = set()
        self._noise_loader: Optional["NoiseLoader"] = None
        self._tts: Optional["TTSGenerator"] = None
        self._kokoro_tts: Optional["KokoroTTSGenerator"] = None
        self._coqui_tts: Optional["CoquiTTSGenerator"] = None
        self._edge_tts: Optional["EdgeTTSGenerator"] = None
    
    def _load_dependencies(self):
        """Lazy load dependencies."""
        if self._noise_loader is None:
            from .augmentation import NoiseLoader
            self._noise_loader = NoiseLoader()
        
        if self._tts is None:
            try:
                from ..tts import TTSGenerator
                self._tts = TTSGenerator(target_sample_rate=self.target_sr)
            except Exception:
                self._tts = None
        
        if self._kokoro_tts is None:
            try:
                from ..tts import KokoroTTSGenerator, KOKORO_AVAILABLE
                if KOKORO_AVAILABLE:
                    # Ensure spacy model is installed (required by Kokoro's misaki G2P)
                    self._ensure_spacy_model()
                    self._kokoro_tts = KokoroTTSGenerator(
                        target_sample_rate=self.target_sr,
                        use_gpu=True,
                    )
            except Exception:
                self._kokoro_tts = None
        
        if self._coqui_tts is None:
            try:
                from ..tts import CoquiTTSGenerator, COQUI_AVAILABLE
                if COQUI_AVAILABLE:
                    self._coqui_tts = CoquiTTSGenerator(
                        target_sample_rate=self.target_sr,
                        use_gpu=True,
                    )
            except Exception:
                self._coqui_tts = None
        
        if self._edge_tts is None:
            try:
                from ..tts import EdgeTTSGenerator, EDGE_TTS_AVAILABLE
                if EDGE_TTS_AVAILABLE:
                    self._edge_tts = EdgeTTSGenerator(
                        target_sample_rate=self.target_sr,
                    )
            except Exception:
                self._edge_tts = None
    
    def _ensure_spacy_model(self):
        """Ensure the spacy model required by Kokoro TTS is installed."""
        try:
            import spacy
            # Check if model is already installed
            if "en_core_web_sm" in spacy.util.get_installed_models():
                return
            
            print("  Installing spacy model for Kokoro TTS (first-time setup)...")
            import subprocess
            import sys
            
            # Download using spacy's download mechanism via subprocess
            model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", model_url],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                # Try with uv pip if regular pip fails
                result = subprocess.run(
                    ["uv", "pip", "install", model_url],
                    capture_output=True,
                    text=True,
                )
            
            if result.returncode == 0:
                print("  Spacy model installed successfully")
            else:
                print(f"  Warning: Could not install spacy model: {result.stderr}")
        except Exception as e:
            print(f"  Warning: Could not verify spacy model: {e}")
    
    def _apply_time_shift(self, audio: np.ndarray, shift: int) -> np.ndarray:
        """Apply time shift to audio."""
        if shift == 0:
            return audio
        
        result = np.zeros_like(audio)
        if shift > 0:
            result[shift:] = audio[:-shift]
        else:
            result[:shift] = audio[-shift:]
        return result
    
    def _apply_speed_change(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Apply speed change via resampling."""
        if speed == 1.0:
            return audio
        
        # Resample to change speed
        new_length = int(len(audio) / speed)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def _apply_pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Apply pitch shift."""
        if semitones == 0:
            return audio
        return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=semitones)
    
    def _apply_volume_change(self, audio: np.ndarray, db: float) -> np.ndarray:
        """Apply volume change in dB."""
        if db == 0:
            return audio
        gain = 10 ** (db / 20)
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
    
    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Add background noise at specified SNR."""
        if self._noise_loader is None or self._noise_loader.num_samples == 0:
            return audio
        
        noise = self._noise_loader.get_random_noise(self.target_duration, self.target_sr)
        
        # Calculate scaling for target SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            target_noise_power = signal_power / (10 ** (snr_db / 10))
            noise_scale = np.sqrt(target_noise_power / noise_power)
            noisy = audio + noise * noise_scale
            return np.clip(noisy, -1.0, 1.0).astype(np.float32)
        
        return audio
    
    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        """Pad or trim audio to target length."""
        if len(audio) == self.target_length:
            return audio
        elif len(audio) > self.target_length:
            start = (len(audio) - self.target_length) // 2
            return audio[start:start + self.target_length]
        else:
            pad_total = self.target_length - len(audio)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(audio, (pad_left, pad_right), mode='constant')
    
    def generate_samples_split(
        self,
        recordings: list[tuple[np.ndarray, int]],
        wake_word: str,
        target_count: int = 6000,
        use_tts: bool = True,
        use_noise: bool = True,
        train_tts_ratio: float = 0.50,  # 50% of training from TTS for voice diversity
        val_tts_ratio: float = 0.60,    # 60% of validation from TTS (UNSEEN voices only)
        val_split: float = 0.25,        # 25% validation
    ) -> tuple[list[tuple[np.ndarray, dict]], list[tuple[np.ndarray, dict]]]:
        """
        Generate samples with proper train/val split ensuring unseen TTS voices in validation.
        
        Training set composition:
        - 50% from user recordings (augmented)
        - 50% from TTS voices (training voices only)
        
        Validation set composition:
        - 40% from user recordings (augmented, different augmentations)
        - 60% from TTS voices (held-out voices NOT in training)
        
        This ensures the model is tested on completely unseen voices in validation,
        
        Args:
            recordings: List of (audio, sample_rate) tuples from user
            wake_word: The wake word text (for TTS generation)
            target_count: Target number of total samples
            use_tts: Generate TTS samples
            use_noise: Add noise variations
            train_tts_ratio: Ratio of TTS samples in training set
            val_tts_ratio: Ratio of TTS samples in validation set
            val_split: Validation set ratio
        
        Returns:
            Tuple of (train_samples, val_samples) where each sample is (audio, metadata)
        """
        self._load_dependencies()
        self._seen_hashes.clear()
        
        train_samples = []
        val_samples = []
        
        # Calculate target sizes
        val_size = int(target_count * val_split)
        train_size = target_count - val_size
        
        # Training: 65% recordings, 35% TTS
        train_rec_target = int(train_size * (1 - train_tts_ratio))
        train_tts_target = train_size - train_rec_target
        
        # Validation: 75% recordings, 25% TTS (unseen voices)
        val_rec_target = int(val_size * (1 - val_tts_ratio))
        val_tts_target = val_size - val_rec_target
        
        num_recordings = len(recordings)
        print(f"  Generating {target_count} positive samples from {num_recordings} recordings...")
        print(f"    Train: {train_size} ({train_rec_target} rec + {train_tts_target} TTS)")
        print(f"    Val: {val_size} ({val_rec_target} rec + {val_tts_target} TTS with unseen voices)")
        
        # Split TTS voices: 60% for training, 40% for validation (unseen)
        # More held-out voices = better generalization testing
        all_voices = []
        if use_tts and self._tts is not None:
            all_voices = list(self._tts.voice_names)
            random.shuffle(all_voices)
        
        val_voice_count = max(2, int(len(all_voices) * 0.4))  # 40% held out
        val_voices = set(all_voices[:val_voice_count])
        train_voices = set(all_voices[val_voice_count:])
        
        if all_voices:
            print(f"    TTS voices: {len(train_voices)} for training, {len(val_voices)} held-out for validation")
        
        # =====================================================================
        # Generate recording-based samples
        # =====================================================================
        rec_per_recording_train = train_rec_target // max(num_recordings, 1)
        rec_per_recording_val = val_rec_target // max(num_recordings, 1)
        
        for rec_idx, (audio, sr) in enumerate(recordings):
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0.01:
                audio = audio / max_val * 0.9
            
            train_count = 0
            val_count = 0
            
            # Generate variations - alternate between train and val
            aug_configs = []
            for speed in self.speed_variations:
                for pitch in self.pitch_shifts:
                    for volume in self.volume_variations:
                        for time_shift in self.time_shifts:
                            aug_configs.append((speed, pitch, volume, time_shift))
            
            random.shuffle(aug_configs)
            
            for i, (speed, pitch, volume, time_shift) in enumerate(aug_configs):
                # Alternate: even indices for train, odd for val
                is_train = (i % 3 != 0)  # 2/3 for train, 1/3 for val
                
                if is_train and train_count >= rec_per_recording_train:
                    continue
                if not is_train and val_count >= rec_per_recording_val:
                    continue
                
                # Apply augmentations
                aug = audio.copy()
                aug = self._apply_speed_change(aug, speed)
                aug = self._apply_pitch_shift(aug, pitch)
                aug = self._apply_time_shift(aug, time_shift)
                aug = self._pad_or_trim(aug)
                aug = self._apply_volume_change(aug, volume)
                
                # Check for duplicates
                aug_hash = compute_audio_hash(aug)
                if aug_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(aug_hash)
                
                metadata = {
                    "source": "recording",
                    "recording_idx": rec_idx,
                    "speed": speed,
                    "pitch": pitch,
                    "volume": volume,
                    "time_shift": time_shift,
                }
                
                if is_train:
                    train_samples.append((aug, metadata))
                    train_count += 1
                else:
                    val_samples.append((aug, metadata))
                    val_count += 1
                
                # Add noisy versions
                if use_noise and self._noise_loader and self._noise_loader.num_samples > 0:
                    for snr in self.noise_snr_levels[:2]:
                        noisy = self._add_noise(aug, snr)
                        noisy_hash = compute_audio_hash(noisy)
                        if noisy_hash in self._seen_hashes:
                            continue
                        self._seen_hashes.add(noisy_hash)
                        
                        metadata_noisy = metadata.copy()
                        metadata_noisy["snr_db"] = snr
                        
                        if is_train and train_count < rec_per_recording_train:
                            train_samples.append((noisy, metadata_noisy))
                            train_count += 1
                        elif not is_train and val_count < rec_per_recording_val:
                            val_samples.append((noisy, metadata_noisy))
                            val_count += 1
            
            print(f"    Recording {rec_idx + 1}: {train_count} train, {val_count} val samples")
        
        # =====================================================================
        # Generate TTS samples with voice separation
        # IMPORTANT: Ensure ALL voices are used at least once for diversity
        # =====================================================================
        if use_tts and self._tts is not None:
            # Training TTS samples (from training voices only)
            # First pass: ensure every training voice is used at least once
            train_tts_count = 0
            voices_used = set()
            
            # First: one sample per voice to ensure full coverage
            for voice in train_voices:
                try:
                    tts_audio, _ = self._tts.synthesize(
                        wake_word,
                        voice_name=voice,
                        length_scale=1.0,
                    )
                    tts_audio = self._pad_or_trim(tts_audio)
                    
                    max_val = np.abs(tts_audio).max()
                    if max_val > 0.01:
                        tts_audio = tts_audio / max_val * 0.9
                    
                    aug_hash = compute_audio_hash(tts_audio)
                    if aug_hash not in self._seen_hashes:
                        self._seen_hashes.add(aug_hash)
                        metadata = {
                            "source": "tts",
                            "voice": voice,
                            "speed": 1.0,
                            "pitch": 0,
                            "volume": 1.0,
                            "split": "train",
                        }
                        train_samples.append((tts_audio, metadata))
                        train_tts_count += 1
                        voices_used.add(voice)
                except Exception:
                    continue
            
            print(f"    TTS voices used (first pass): {len(voices_used)}/{len(train_voices)}")
            
            # Second pass: add variations until target reached
            for voice in train_voices:
                if train_tts_count >= train_tts_target:
                    break
                
                for speed in self.speed_variations:
                    if train_tts_count >= train_tts_target:
                        break
                    
                    try:
                        tts_audio, _ = self._tts.synthesize(
                            wake_word,
                            voice_name=voice,
                            length_scale=1.0 / speed,
                        )
                        tts_audio = self._pad_or_trim(tts_audio)
                        
                        max_val = np.abs(tts_audio).max()
                        if max_val > 0.01:
                            tts_audio = tts_audio / max_val * 0.9
                        
                        for pitch in self.pitch_shifts:
                            for volume in self.volume_variations[:3]:
                                if train_tts_count >= train_tts_target:
                                    break
                                
                                aug = self._apply_pitch_shift(tts_audio, pitch)
                                aug = self._apply_volume_change(aug, volume)
                                
                                aug_hash = compute_audio_hash(aug)
                                if aug_hash in self._seen_hashes:
                                    continue
                                self._seen_hashes.add(aug_hash)
                                
                                metadata = {
                                    "source": "tts",
                                    "voice": voice,
                                    "speed": speed,
                                    "pitch": pitch,
                                    "volume": volume,
                                    "split": "train",
                                }
                                
                                train_samples.append((aug, metadata))
                                train_tts_count += 1
                    except Exception:
                        continue
            
            print(f"    TTS training samples: {train_tts_count}")
            
            # Validation TTS samples (from held-out voices only)
            # First pass: ensure every validation voice is used at least once
            val_tts_count = 0
            val_voices_used = set()
            
            for voice in val_voices:
                try:
                    tts_audio, _ = self._tts.synthesize(
                        wake_word,
                        voice_name=voice,
                        length_scale=1.0,
                    )
                    tts_audio = self._pad_or_trim(tts_audio)
                    
                    max_val = np.abs(tts_audio).max()
                    if max_val > 0.01:
                        tts_audio = tts_audio / max_val * 0.9
                    
                    aug_hash = compute_audio_hash(tts_audio)
                    if aug_hash not in self._seen_hashes:
                        self._seen_hashes.add(aug_hash)
                        metadata = {
                            "source": "tts",
                            "voice": voice,
                            "speed": 1.0,
                            "pitch": 0,
                            "volume": 1.0,
                            "split": "val",
                        }
                        val_samples.append((tts_audio, metadata))
                        val_tts_count += 1
                        val_voices_used.add(voice)
                except Exception:
                    continue
            
            print(f"    TTS val voices used (first pass): {len(val_voices_used)}/{len(val_voices)}")
            
            # Second pass: add variations until target reached
            for voice in val_voices:
                if val_tts_count >= val_tts_target:
                    break
                
                for speed in self.speed_variations:
                    if val_tts_count >= val_tts_target:
                        break
                    
                    try:
                        tts_audio, _ = self._tts.synthesize(
                            wake_word,
                            voice_name=voice,
                            length_scale=1.0 / speed,
                        )
                        tts_audio = self._pad_or_trim(tts_audio)
                        
                        max_val = np.abs(tts_audio).max()
                        if max_val > 0.01:
                            tts_audio = tts_audio / max_val * 0.9
                        
                        for pitch in self.pitch_shifts:
                            for volume in self.volume_variations[:3]:
                                if val_tts_count >= val_tts_target:
                                    break
                                
                                aug = self._apply_pitch_shift(tts_audio, pitch)
                                aug = self._apply_volume_change(aug, volume)
                                
                                aug_hash = compute_audio_hash(aug)
                                if aug_hash in self._seen_hashes:
                                    continue
                                self._seen_hashes.add(aug_hash)
                                
                                metadata = {
                                    "source": "tts",
                                    "voice": voice,
                                    "speed": speed,
                                    "pitch": pitch,
                                    "volume": volume,
                                    "split": "val",
                                }
                                
                                val_samples.append((aug, metadata))
                                val_tts_count += 1
                    except Exception:
                        continue
            
            print(f"    TTS validation samples (unseen voices): {val_tts_count}")
        
        # =====================================================================
        # Generate Kokoro TTS samples (high-quality, diverse voices)
        # Uses ALL voices with speed (1.0, 1.5) and volume (-6, -3, 0, +3 dB) variations
        # Split voices: 70% train, 30% validation (unseen)
        # =====================================================================
        if self._kokoro_tts is not None:
            from ..tts import KOKORO_VOICES, KOKORO_SPEED_VARIATIONS, KOKORO_VOLUME_VARIATIONS
            
            kokoro_voices = list(KOKORO_VOICES.keys())
            random.shuffle(kokoro_voices)
            
            # Split voices for train/val
            kokoro_val_count = max(4, int(len(kokoro_voices) * 0.3))  # 30% for validation
            kokoro_val_voices = set(kokoro_voices[:kokoro_val_count])
            kokoro_train_voices = set(kokoro_voices[kokoro_val_count:])
            
            print(f"  Generating Kokoro TTS samples ({len(KOKORO_VOICES)} voices across all languages)...")
            print(f"    Kokoro voices: {len(kokoro_train_voices)} train, {len(kokoro_val_voices)} val (unseen)")
            print(f"    Speeds: {KOKORO_SPEED_VARIATIONS}, Volumes: {KOKORO_VOLUME_VARIATIONS} dB")
            
            kokoro_train_count = 0
            kokoro_val_count = 0
            
            # Training samples from Kokoro (train voices only)
            for voice_id in kokoro_train_voices:
                for speed in KOKORO_SPEED_VARIATIONS:  # [1.0, 1.5]
                    try:
                        audio, _ = self._kokoro_tts.synthesize(
                            wake_word,
                            voice_id=voice_id,
                            speed=speed,
                        )
                        audio = self._pad_or_trim(audio)
                        
                        # Normalize
                        max_val = np.abs(audio).max()
                        if max_val > 0.01:
                            audio = audio / max_val * 0.9
                        
                        voice_info = KOKORO_VOICES.get(voice_id, ("Unknown", "unknown", "unknown", "a"))
                        
                        # Apply volume variations
                        for volume_db in KOKORO_VOLUME_VARIATIONS:
                            aug = self._apply_volume_change(audio.copy(), volume_db)
                            
                            # Check for duplicates
                            audio_hash = compute_audio_hash(aug)
                            if audio_hash in self._seen_hashes:
                                continue
                            self._seen_hashes.add(audio_hash)
                            
                            metadata = {
                                "source": "kokoro_tts",
                                "voice_id": voice_id,
                                "voice_name": voice_info[0],
                                "gender": voice_info[1],
                                "accent": voice_info[2],
                                "speed": speed,
                                "volume_db": volume_db,
                                "split": "train",
                            }
                            
                            train_samples.append((aug, metadata))
                            kokoro_train_count += 1
                            
                            # Add noisy version for some samples (30% chance)
                            if use_noise and self._noise_loader and self._noise_loader.num_samples > 0:
                                if random.random() < 0.3:
                                    snr = random.choice([10, 15, 20])
                                    noisy = self._add_noise(aug, snr)
                                    noisy_hash = compute_audio_hash(noisy)
                                    if noisy_hash not in self._seen_hashes:
                                        self._seen_hashes.add(noisy_hash)
                                        metadata_noisy = metadata.copy()
                                        metadata_noisy["snr_db"] = snr
                                        train_samples.append((noisy, metadata_noisy))
                                        kokoro_train_count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"    Kokoro training samples: {kokoro_train_count}")
            
            # Validation samples from Kokoro (held-out voices only)
            # Only use base volume (0 dB) for validation - no augmentation
            for voice_id in kokoro_val_voices:
                for speed in KOKORO_SPEED_VARIATIONS:
                    try:
                        audio, _ = self._kokoro_tts.synthesize(
                            wake_word,
                            voice_id=voice_id,
                            speed=speed,
                        )
                        audio = self._pad_or_trim(audio)
                        
                        max_val = np.abs(audio).max()
                        if max_val > 0.01:
                            audio = audio / max_val * 0.9
                        
                        audio_hash = compute_audio_hash(audio)
                        if audio_hash in self._seen_hashes:
                            continue
                        self._seen_hashes.add(audio_hash)
                        
                        voice_info = KOKORO_VOICES.get(voice_id, ("Unknown", "unknown", "unknown", "a"))
                        metadata = {
                            "source": "kokoro_tts",
                            "voice_id": voice_id,
                            "voice_name": voice_info[0],
                            "gender": voice_info[1],
                            "accent": voice_info[2],
                            "speed": speed,
                            "volume_db": 0,
                            "split": "val",
                        }
                        
                        val_samples.append((audio, metadata))
                        kokoro_val_count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"    Kokoro validation samples (unseen voices): {kokoro_val_count}")
            
            # Cleanup Kokoro GPU memory after generation
            self._kokoro_tts.cleanup()
            self._kokoro_tts = None
        
        # =====================================================================
        # Generate Coqui TTS samples (VCTK 109 speakers, YourTTS multi-lingual)
        # =====================================================================
        if self._coqui_tts is not None:
            from ..tts import COQUI_MODELS, COQUI_SPEED_VARIATIONS, COQUI_VOLUME_VARIATIONS
            
            print(f"  Generating Coqui TTS samples ({len(COQUI_MODELS)} models)...")
            coqui_train_count = 0
            coqui_val_count = 0
            
            for model_key in self._coqui_tts.model_keys:
                speakers = self._coqui_tts.get_speakers(model_key)
                if not speakers:
                    continue
                
                # Split speakers 70/30
                random.shuffle(speakers)
                val_count = max(1, int(len(speakers) * 0.3))
                val_spk = set(speakers[:val_count])
                train_spk = set(speakers[val_count:])
                
                # Training samples
                for speaker in train_spk:
                    for speed in COQUI_SPEED_VARIATIONS:
                        try:
                            audio, _ = self._coqui_tts.synthesize(wake_word, model_key=model_key, speaker=speaker)
                            audio = self._pad_or_trim(audio)
                            max_val = np.abs(audio).max()
                            if max_val > 0.01:
                                audio = audio / max_val * 0.9
                            
                            for vol_db in COQUI_VOLUME_VARIATIONS:
                                aug = self._apply_volume_change(audio.copy(), vol_db)
                                h = compute_audio_hash(aug)
                                if h in self._seen_hashes:
                                    continue
                                self._seen_hashes.add(h)
                                train_samples.append((aug, {"source": "coqui_tts", "model": model_key, "speaker": speaker, "speed": speed, "vol": vol_db}))
                                coqui_train_count += 1
                        except Exception:
                            continue
                
                # Validation samples (held-out speakers)
                for speaker in val_spk:
                    try:
                        audio, _ = self._coqui_tts.synthesize(wake_word, model_key=model_key, speaker=speaker)
                        audio = self._pad_or_trim(audio)
                        max_val = np.abs(audio).max()
                        if max_val > 0.01:
                            audio = audio / max_val * 0.9
                        h = compute_audio_hash(audio)
                        if h not in self._seen_hashes:
                            self._seen_hashes.add(h)
                            val_samples.append((audio, {"source": "coqui_tts", "model": model_key, "speaker": speaker, "split": "val"}))
                            coqui_val_count += 1
                    except Exception:
                        continue
            
            print(f"    Coqui TTS: {coqui_train_count} train, {coqui_val_count} val samples")
            self._coqui_tts.cleanup()
            self._coqui_tts = None
        
        # =====================================================================
        # Generate Edge TTS samples (400+ voices)
        # =====================================================================
        if self._edge_tts is not None:
            from ..tts import EDGE_SPEED_VARIATIONS, EDGE_VOLUME_VARIATIONS
            
            voices = self._edge_tts.voice_ids
            print(f"  Generating Edge TTS samples ({len(voices)} voices)...")
            
            # Split voices 70/30
            random.shuffle(voices)
            val_count = max(5, int(len(voices) * 0.3))
            val_voices = set(voices[:val_count])
            train_voices = set(voices[val_count:])
            
            edge_train_count = 0
            edge_val_count = 0
            
            # Training samples
            for voice in train_voices:
                for speed in EDGE_SPEED_VARIATIONS:
                    try:
                        audio, _ = self._edge_tts.synthesize(wake_word, voice=voice, speed=speed)
                        audio = self._pad_or_trim(audio)
                        max_val = np.abs(audio).max()
                        if max_val > 0.01:
                            audio = audio / max_val * 0.9
                        
                        for vol_db in EDGE_VOLUME_VARIATIONS:
                            aug = self._apply_volume_change(audio.copy(), vol_db)
                            h = compute_audio_hash(aug)
                            if h in self._seen_hashes:
                                continue
                            self._seen_hashes.add(h)
                            train_samples.append((aug, {"source": "edge_tts", "voice": voice, "speed": speed, "vol": vol_db}))
                            edge_train_count += 1
                    except Exception:
                        continue
            
            # Validation samples (held-out voices)
            for voice in val_voices:
                try:
                    audio, _ = self._edge_tts.synthesize(wake_word, voice=voice, speed=1.0)
                    audio = self._pad_or_trim(audio)
                    max_val = np.abs(audio).max()
                    if max_val > 0.01:
                        audio = audio / max_val * 0.9
                    h = compute_audio_hash(audio)
                    if h not in self._seen_hashes:
                        self._seen_hashes.add(h)
                        val_samples.append((audio, {"source": "edge_tts", "voice": voice, "split": "val"}))
                        edge_val_count += 1
                except Exception:
                    continue
            
            print(f"    Edge TTS: {edge_train_count} train, {edge_val_count} val samples")
            self._edge_tts.cleanup()
            self._edge_tts = None
        
        print(f"  Total: {len(train_samples)} train, {len(val_samples)} val positive samples")
        return train_samples, val_samples
    
    def generate_samples(
        self,
        recordings: list[tuple[np.ndarray, int]],
        wake_word: str,
        target_count: int = 6000,
        use_tts: bool = True,
        use_noise: bool = True,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        """
        Generate massive number of positive samples (legacy method).
        
        Args:
            recordings: List of (audio, sample_rate) tuples from user
            wake_word: The wake word text (for TTS generation)
            target_count: Target number of samples to generate
            use_tts: Generate TTS samples in addition to augmented recordings
            use_noise: Add noise variations
        
        Yields:
            Tuple of (audio, metadata)
        """
        self._load_dependencies()
        self._seen_hashes.clear()
        
        sample_count = 0
        
        # Calculate samples per recording
        num_recordings = len(recordings)
        
        # Phase 1: Augment user recordings (primary source)
        # This should generate ~65% of samples
        recording_target = int(target_count * 0.65) // max(num_recordings, 1)
        
        print(f"  Generating {target_count} positive samples from {num_recordings} recordings...")
        
        for rec_idx, (audio, sr) in enumerate(recordings):
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 0.01:
                audio = audio / max_val * 0.9
            
            # Generate variations
            rec_samples = 0
            
            for speed in self.speed_variations:
                for pitch in self.pitch_shifts:
                    for volume in self.volume_variations:
                        for time_shift in self.time_shifts:
                            if rec_samples >= recording_target:
                                break
                            if sample_count >= target_count:
                                break
                            
                            # Apply augmentations
                            aug = audio.copy()
                            aug = self._apply_speed_change(aug, speed)
                            aug = self._apply_pitch_shift(aug, pitch)
                            aug = self._apply_time_shift(aug, time_shift)
                            aug = self._pad_or_trim(aug)
                            aug = self._apply_volume_change(aug, volume)
                            
                            # Check for duplicates
                            aug_hash = compute_audio_hash(aug)
                            if aug_hash in self._seen_hashes:
                                continue
                            self._seen_hashes.add(aug_hash)
                            
                            metadata = {
                                "source": "recording",
                                "recording_idx": rec_idx,
                                "speed": speed,
                                "pitch": pitch,
                                "volume": volume,
                                "time_shift": time_shift,
                            }
                            
                            yield aug, metadata
                            sample_count += 1
                            rec_samples += 1
                            
                            # Add noisy versions
                            if use_noise and self._noise_loader and self._noise_loader.num_samples > 0:
                                for snr in self.noise_snr_levels[:3]:  # Limit noise variations
                                    if rec_samples >= recording_target:
                                        break
                                    if sample_count >= target_count:
                                        break
                                    
                                    noisy = self._add_noise(aug, snr)
                                    noisy_hash = compute_audio_hash(noisy)
                                    if noisy_hash in self._seen_hashes:
                                        continue
                                    self._seen_hashes.add(noisy_hash)
                                    
                                    metadata_noisy = metadata.copy()
                                    metadata_noisy["snr_db"] = snr
                                    
                                    yield noisy, metadata_noisy
                                    sample_count += 1
                                    rec_samples += 1
            
            print(f"    Recording {rec_idx + 1}: generated {rec_samples} samples")
        
        # Phase 2: TTS-generated samples (~30% of target)
        if use_tts and self._tts is not None and sample_count < target_count:
            tts_target = target_count - sample_count
            print(f"  Generating {tts_target} TTS samples...")
            
            voices = self._tts.voice_names
            tts_count = 0
            
            for voice in voices:
                if sample_count >= target_count:
                    break
                
                for speed in self.speed_variations:
                    if sample_count >= target_count:
                        break
                    
                    try:
                        tts_audio, _ = self._tts.synthesize(
                            wake_word,
                            voice_name=voice,
                            length_scale=1.0 / speed,
                        )
                        tts_audio = self._pad_or_trim(tts_audio)
                        
                        # Normalize
                        max_val = np.abs(tts_audio).max()
                        if max_val > 0.01:
                            tts_audio = tts_audio / max_val * 0.9
                        
                        # Apply additional variations
                        for pitch in self.pitch_shifts:
                            for volume in self.volume_variations[:3]:
                                if sample_count >= target_count:
                                    break
                                
                                aug = self._apply_pitch_shift(tts_audio, pitch)
                                aug = self._apply_volume_change(aug, volume)
                                
                                aug_hash = compute_audio_hash(aug)
                                if aug_hash in self._seen_hashes:
                                    continue
                                self._seen_hashes.add(aug_hash)
                                
                                metadata = {
                                    "source": "tts",
                                    "voice": voice,
                                    "speed": speed,
                                    "pitch": pitch,
                                    "volume": volume,
                                }
                                
                                yield aug, metadata
                                sample_count += 1
                                tts_count += 1
                                
                                # Add noisy version
                                if use_noise and self._noise_loader and self._noise_loader.num_samples > 0:
                                    if sample_count >= target_count:
                                        break
                                    
                                    noisy = self._add_noise(aug, random.choice(self.noise_snr_levels))
                                    noisy_hash = compute_audio_hash(noisy)
                                    if noisy_hash not in self._seen_hashes:
                                        self._seen_hashes.add(noisy_hash)
                                        
                                        metadata_noisy = metadata.copy()
                                        metadata_noisy["noisy"] = True
                                        
                                        yield noisy, metadata_noisy
                                        sample_count += 1
                                        tts_count += 1
                    
                    except Exception as e:
                        print(f"    TTS failed for {voice}: {e}")
                        continue
            
            print(f"    Generated {tts_count} TTS samples")
        
        print(f"  Total positive samples generated: {sample_count}")
