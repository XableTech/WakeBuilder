#!/usr/bin/env python3
"""
Convert MP3 files to WAV format for training.

This script converts all MP3 files in a directory to WAV format,
resampling to 16kHz mono for wake word training.

By default, original MP3 files are DELETED after successful conversion
to avoid duplicates. Use --keep-original to preserve them.

Usage:
    python scripts/convert_mp3_to_wav.py data/negative/music
    python scripts/convert_mp3_to_wav.py data/negative/speech --sample-rate 16000
    python scripts/convert_mp3_to_wav.py data/negative/ --recursive
    python scripts/convert_mp3_to_wav.py data/negative/ --recursive --keep-original
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install librosa soundfile numpy")
    sys.exit(1)


def convert_mp3_to_wav(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> tuple[bool, str]:
    """
    Convert a single MP3 file to WAV format.
    
    Args:
        input_path: Path to input MP3 file
        output_path: Path to output WAV file
        target_sr: Target sample rate (default: 16000 Hz)
        mono: Convert to mono (default: True)
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Load audio with librosa (handles MP3, FLAC, OGG, etc.)
        audio, sr = librosa.load(input_path, sr=target_sr, mono=mono)
        
        # Normalize audio to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 0.01:
            audio = audio / max_val * 0.95
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV
        sf.write(output_path, audio, target_sr, subtype='PCM_16')
        
        return True, f"Converted: {input_path.name}"
    
    except Exception as e:
        return False, f"Failed: {input_path.name} - {e}"


def convert_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    target_sr: int = 16000,
    recursive: bool = False,
    keep_original: bool = False,
    max_workers: int = 4,
) -> tuple[int, int, int]:
    """
    Convert all MP3 files in a directory to WAV format.
    
    Args:
        input_dir: Input directory containing MP3 files
        output_dir: Output directory (default: same as input)
        target_sr: Target sample rate
        recursive: Process subdirectories recursively
        delete_original: Delete original MP3 files after conversion
        max_workers: Number of parallel workers
    
    Returns:
        Tuple of (success_count, fail_count, deleted_count)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Find all MP3 files
    pattern = "**/*.mp3" if recursive else "*.mp3"
    mp3_files = list(input_dir.glob(pattern))
    
    # Also check for uppercase extension
    mp3_files.extend(input_dir.glob(pattern.replace(".mp3", ".MP3")))
    
    if not mp3_files:
        print(f"No MP3 files found in {input_dir}")
        return 0, 0, 0
    
    print(f"Found {len(mp3_files)} MP3 files to convert")
    print(f"Target sample rate: {target_sr} Hz")
    print(f"Output directory: {output_dir}")
    print()
    
    success_count = 0
    fail_count = 0
    deleted_count = 0
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for mp3_path in mp3_files:
            # Compute relative path for output
            rel_path = mp3_path.relative_to(input_dir)
            wav_path = output_dir / rel_path.with_suffix('.wav')
            
            # Skip if WAV already exists
            if wav_path.exists():
                print(f"Skipped (exists): {rel_path}")
                success_count += 1
                continue
            
            future = executor.submit(
                convert_mp3_to_wav, mp3_path, wav_path, target_sr
            )
            futures[future] = (mp3_path, wav_path)
        
        # Process results
        for future in as_completed(futures):
            mp3_path, wav_path = futures[future]
            success, message = future.result()
            
            if success:
                success_count += 1
                print(f"  ✓ {message}")
                
                # Delete original by default (unless --keep-original)
                if not keep_original:
                    try:
                        mp3_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ⚠ Could not delete {mp3_path.name}: {e}")
            else:
                fail_count += 1
                print(f"  ✗ {message}")
    
    return success_count, fail_count, deleted_count


def get_audio_stats(directory: Path, recursive: bool = False) -> dict:
    """Get statistics about audio files in a directory."""
    pattern = "**/*" if recursive else "*"
    
    stats = {
        "mp3": 0,
        "wav": 0,
        "flac": 0,
        "ogg": 0,
        "other": 0,
        "total_size_mb": 0,
    }
    
    for ext in ["mp3", "wav", "flac", "ogg"]:
        # Use set to avoid double-counting on case-insensitive filesystems (Windows)
        files = set(directory.glob(f"{pattern}.{ext}"))
        files.update(directory.glob(f"{pattern}.{ext.upper()}"))
        stats[ext] = len(files)
        stats["total_size_mb"] += sum(f.stat().st_size for f in files) / (1024 * 1024)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to WAV format for wake word training"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing MP3 files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "-k", "--keep-original",
        action="store_true",
        help="Keep original MP3 files after conversion (default: delete them)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't convert"
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Show stats
    print(f"\n{'='*60}")
    print(f"Audio File Statistics: {args.input_dir}")
    print(f"{'='*60}")
    
    stats = get_audio_stats(args.input_dir, args.recursive)
    print(f"  MP3 files:  {stats['mp3']:,}")
    print(f"  WAV files:  {stats['wav']:,}")
    print(f"  FLAC files: {stats['flac']:,}")
    print(f"  OGG files:  {stats['ogg']:,}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print()
    
    if args.stats_only:
        return
    
    if stats['mp3'] == 0:
        print("No MP3 files to convert.")
        return
    
    # Convert
    print(f"{'='*60}")
    print("Converting MP3 to WAV...")
    print(f"{'='*60}")
    
    success, fail, deleted = convert_directory(
        args.input_dir,
        args.output_dir,
        args.sample_rate,
        args.recursive,
        args.keep_original,
        args.workers,
    )
    
    print()
    print(f"{'='*60}")
    print(f"Conversion complete!")
    print(f"  Success: {success}")
    print(f"  Failed:  {fail}")
    if not args.keep_original:
        print(f"  Deleted: {deleted} original MP3 files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
