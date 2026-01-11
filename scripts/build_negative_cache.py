#!/usr/bin/env python3
"""
Build negative data cache for fast training.

This script pre-processes all audio files in data/negative/ and caches
the 1-second chunks as numpy files for instant loading during training.

Two cache levels:
1. Audio cache: Raw 1-second audio chunks (~1 min to build, ~30s to load)
2. Spectrogram cache: Pre-computed mel spectrograms (~2 min to build, instant load)

Usage:
    python scripts/build_negative_cache.py              # Build audio cache
    python scripts/build_negative_cache.py --spectrograms  # Build spectrogram cache (recommended)
    python scripts/build_negative_cache.py --workers 8
    python scripts/build_negative_cache.py --clear      # Clear all caches
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wakebuilder.audio.real_data_loader import RealNegativeDataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Build negative data cache for fast training"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--spectrograms",
        action="store_true",
        help="Build spectrogram cache (instant loading, recommended)",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing cache before building"
    )
    parser.add_argument("--info", action="store_true", help="Show cache info only")

    args = parser.parse_args()

    loader = RealNegativeDataLoader()

    # Show current cache info
    cache_info = loader.get_cache_info()
    spec_cache_info = loader.get_spectrogram_cache_info()
    file_counts = loader.get_file_count()

    print(f"\n{'='*60}")
    print("Negative Data Cache Status")
    print(f"{'='*60}")
    print(f"  Source files: {file_counts['total']:,}")
    print(f"    - WAV:  {file_counts['wav']:,}")
    print(f"    - MP3:  {file_counts['mp3']:,}")
    print(f"    - FLAC: {file_counts['flac']:,}")
    print(f"    - OGG:  {file_counts['ogg']:,}")
    print()

    if cache_info["cached"]:
        print(f"  Audio cache: {cache_info['chunk_count']:,} chunks")
        print(f"    Created: {cache_info.get('created_at', 'unknown')}")
    else:
        print("  Audio cache: Not built")

    if spec_cache_info["cached"]:
        print(
            f"  Spectrogram cache: {spec_cache_info['count']:,} spectrograms (INSTANT LOAD)"
        )
        print(f"    Created: {spec_cache_info.get('created_at', 'unknown')}")
    else:
        print("  Spectrogram cache: Not built")
    print()

    if args.info:
        return

    if args.clear:
        print("Clearing existing caches...")
        loader.clear_cache()
        loader.clear_spectrogram_cache()
        print()

    if not loader.available:
        print("ERROR: No audio files found in data/negative/")
        sys.exit(1)

    # Build audio cache first if needed
    if not cache_info["cached"] or cache_info["chunk_count"] == 0:
        print(f"{'='*60}")
        print(f"Building audio cache with {args.workers} workers...")
        print(f"{'='*60}")
        print()

        chunk_count = loader.build_cache(max_workers=args.workers)

        print()
        print(f"  Audio cache: {chunk_count:,} chunks")

    # Build spectrogram cache if requested
    if args.spectrograms:
        print()
        print(f"{'='*60}")
        print("Building spectrogram cache (this enables instant loading)...")
        print(f"{'='*60}")
        print()

        # Import preprocessor
        from wakebuilder.audio import AudioPreprocessor

        preprocessor = AudioPreprocessor()

        spec_count = loader.build_spectrogram_cache(preprocessor)

        print()
        print(f"{'='*60}")
        print("Spectrogram cache built!")
        print(f"  Total spectrograms: {spec_count:,}")
        print("  Loading time: INSTANT (< 1 second)")
        print(f"{'='*60}")
    else:
        print()
        print(f"{'='*60}")
        print("Audio cache ready!")
        print(f"  Total chunks: {cache_info['chunk_count']:,}")
        print()
        print("TIP: Run with --spectrograms for instant loading:")
        print("  uv run python scripts/build_negative_cache.py --spectrograms")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
