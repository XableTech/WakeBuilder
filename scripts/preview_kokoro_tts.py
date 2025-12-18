#!/usr/bin/env python3
"""
Preview Kokoro TTS voices for WakeBuilder.

This script generates preview audio samples using all available Kokoro TTS voices
and saves them to a temporary folder for review.

Usage:
    uv run python scripts/preview_kokoro_tts.py "hey siri"
    uv run python scripts/preview_kokoro_tts.py "hello world" --voices af_heart am_michael
    uv run python scripts/preview_kokoro_tts.py "test" --speeds 0.5 1.0 1.5
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import soundfile as sf


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Preview Kokoro TTS voices for WakeBuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/preview_kokoro_tts.py "hey siri"
    uv run python scripts/preview_kokoro_tts.py "hello world" --voices af_heart am_michael bf_emma
    uv run python scripts/preview_kokoro_tts.py "test" --speeds 0.5 1.0
    uv run python scripts/preview_kokoro_tts.py "wake word" --list-voices
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="hello world",
        help="Text to synthesize (default: 'hello world')",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        help="Specific voice IDs to use (default: all voices)",
    )
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[1.0, 1.5],
        help="Speed variations (default: 1.0 1.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/temp/kokoro_tts_preview)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices and exit",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    
    args = parser.parse_args()
    
    # Import Kokoro generator
    try:
        from wakebuilder.tts.kokoro_generator import (
            KokoroTTSGenerator,
            KOKORO_VOICES,
            KOKORO_AVAILABLE,
            list_kokoro_voices,
        )
    except ImportError as e:
        print(f"[ERROR] Failed to import Kokoro TTS: {e}")
        print("\nPlease install Kokoro TTS with:")
        print("  uv add kokoro soundfile")
        return 1
    
    if not KOKORO_AVAILABLE:
        print("[ERROR] Kokoro TTS is not available.")
        print("\nPlease install Kokoro TTS with:")
        print("  uv add kokoro soundfile")
        return 1
    
    # List voices if requested
    if args.list_voices:
        print("=" * 70)
        print("Available Kokoro TTS Voices")
        print("=" * 70)
        print(f"\nTotal voices: {len(KOKORO_VOICES)}")
        
        # Group voices by language/accent
        languages = {}
        for voice_id, info in KOKORO_VOICES.items():
            name, gender, accent, lang_code = info
            key = (accent, gender)
            if key not in languages:
                languages[key] = []
            languages[key].append((voice_id, name))
        
        # Print by category
        categories = [
            (("american", "female"), "American English Female"),
            (("american", "male"), "American English Male"),
            (("british", "female"), "British English Female"),
            (("british", "male"), "British English Male"),
            (("spanish", "female"), "Spanish Female"),
            (("spanish", "male"), "Spanish Male"),
            (("french", "female"), "French Female"),
            (("hindi", "female"), "Hindi Female"),
            (("hindi", "male"), "Hindi Male"),
            (("italian", "female"), "Italian Female"),
            (("italian", "male"), "Italian Male"),
            (("japanese", "female"), "Japanese Female"),
            (("japanese", "male"), "Japanese Male"),
            (("portuguese", "female"), "Portuguese Female"),
            (("portuguese", "male"), "Portuguese Male"),
            (("chinese", "female"), "Chinese Female"),
            (("chinese", "male"), "Chinese Male"),
        ]
        
        for key, label in categories:
            if key in languages:
                voices_list = languages[key]
                print(f"\n--- {label} ({len(voices_list)} voices) ---")
                for voice_id, name in sorted(voices_list):
                    print(f"  {voice_id:18} - {name}")
        
        return 0
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "temp" / "kokoro_tts_preview"
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine voices to use
    if args.voices:
        voices = args.voices
        # Validate voice IDs
        invalid = [v for v in voices if v not in KOKORO_VOICES]
        if invalid:
            print(f"[ERROR] Invalid voice IDs: {invalid}")
            print(f"Use --list-voices to see available voices")
            return 1
    else:
        voices = list(KOKORO_VOICES.keys())
    
    print("=" * 70)
    print("Kokoro TTS Preview Generator")
    print("=" * 70)
    print(f"\nText: \"{args.text}\"")
    print(f"Voices: {len(voices)}")
    print(f"Speeds: {args.speeds}")
    print(f"Total samples: {len(voices) * len(args.speeds)}")
    print(f"Output: {output_dir}")
    print(f"GPU: {'Disabled' if args.no_gpu else 'Auto-detect'}")
    
    # Initialize generator
    print("\n" + "-" * 70)
    print("Initializing Kokoro TTS...")
    
    start_time = time.time()
    
    try:
        generator = KokoroTTSGenerator(
            target_sample_rate=16000,
            use_gpu=not args.no_gpu,
        )
        print(f"Using device: {'GPU (CUDA)' if generator.using_gpu else 'CPU'}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Kokoro TTS: {e}")
        return 1
    
    # Generate samples
    print("\n" + "-" * 70)
    print("Generating audio samples...")
    
    generated = 0
    failed = 0
    
    for voice_id in voices:
        voice_info = KOKORO_VOICES[voice_id]
        display_name, gender, accent, lang_code = voice_info
        
        for speed in args.speeds:
            try:
                # Generate audio
                audio, sr = generator.synthesize(
                    args.text,
                    voice_id=voice_id,
                    speed=speed,
                )
                
                # Create filename
                speed_str = f"{speed:.1f}".replace(".", "p")
                filename = f"{voice_id}_speed{speed_str}.wav"
                filepath = output_dir / filename
                
                # Save audio
                sf.write(filepath, audio, sr)
                
                duration = len(audio) / sr
                print(f"  [OK] {filename} ({duration:.2f}s)")
                generated += 1
                
            except Exception as e:
                print(f"  [FAIL] {voice_id} @ {speed}x: {e}")
                failed += 1
    
    # Cleanup GPU memory
    print("\n" + "-" * 70)
    print("Cleaning up...")
    generator.cleanup()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Generated: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    
    if generated > 0:
        print(f"\n[OK] Preview files saved to: {output_dir}")
        print("\nYou can play the files with any audio player.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
