#!/usr/bin/env python3
"""
Preview Edge TTS voices for WakeBuilder.

This script generates preview audio samples using Microsoft Edge TTS
with all available voices, saving them to a temporary folder.

Edge TTS is free and provides 400+ high-quality neural voices across 100+ languages.

Package: edge-tts

Usage:
    uv run python scripts/preview_edge_tts.py "hey siri"
    uv run python scripts/preview_edge_tts.py "hello world" --locale en
    uv run python scripts/preview_edge_tts.py "bonjour" --locale fr
    uv run python scripts/preview_edge_tts.py --list-voices
    uv run python scripts/preview_edge_tts.py "test" --gender Female
"""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path


async def get_voices():
    """Get all available Edge TTS voices."""
    import edge_tts

    voices = await edge_tts.list_voices()
    return voices


def parse_voice_file(filepath: Path) -> list[dict]:
    """Parse the voice list file to extract voice information."""
    voices = []
    current_voice = {}

    with open(filepath, "r", encoding="utf-8") as f:
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


def list_voices(
    voices: list[dict], locale_filter: str = None, gender_filter: str = None
):
    """List all available voices with optional filtering."""
    print("=" * 70)
    print("Available Edge TTS Voices")
    print("=" * 70)

    # Apply filters
    filtered = voices
    if locale_filter:
        filtered = [
            v
            for v in filtered
            if v.get("Locale", "").lower().startswith(locale_filter.lower())
        ]
    if gender_filter:
        filtered = [
            v for v in filtered if v.get("Gender", "").lower() == gender_filter.lower()
        ]

    # Group by locale
    by_locale = {}
    for voice in filtered:
        locale = voice.get("Locale", "unknown")
        if locale not in by_locale:
            by_locale[locale] = []
        by_locale[locale].append(voice)

    print(f"\nTotal voices: {len(filtered)}")
    print(f"Locales: {len(by_locale)}")

    for locale in sorted(by_locale.keys()):
        locale_voices = by_locale[locale]
        print(f"\n--- {locale} ({len(locale_voices)} voices) ---")
        for v in locale_voices:
            gender = v.get("Gender", "?")[0]  # M or F
            short_name = v.get("ShortName", "?")
            print(f"  [{gender}] {short_name}")

    return filtered


async def generate_previews(
    text: str,
    voices: list[dict],
    output_dir: Path,
    locale_filter: str = None,
    gender_filter: str = None,
    max_voices: int = None,
):
    """Generate preview audio samples for all voices."""
    import edge_tts

    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply filters
    filtered = voices
    if locale_filter:
        filtered = [
            v
            for v in filtered
            if v.get("Locale", "").lower().startswith(locale_filter.lower())
        ]
    if gender_filter:
        filtered = [
            v for v in filtered if v.get("Gender", "").lower() == gender_filter.lower()
        ]

    if max_voices and len(filtered) > max_voices:
        filtered = filtered[:max_voices]

    print("=" * 70)
    print("Edge TTS Preview Generator")
    print("=" * 70)
    print(f'\nText: "{text}"')
    print(f"Voices: {len(filtered)}")
    print(f"Locale filter: {locale_filter or 'all'}")
    print(f"Gender filter: {gender_filter or 'all'}")
    print(f"Output: {output_dir}")

    start_time = time.time()
    generated = 0
    failed = 0

    print("\n" + "-" * 70)
    print("Generating audio samples...")

    for i, voice in enumerate(filtered):
        short_name = voice.get("ShortName", "unknown")
        locale = voice.get("Locale", "unknown")
        gender = voice.get("Gender", "unknown")

        try:
            # Sanitize filename
            safe_name = re.sub(r"[^\w\-]", "_", short_name)
            filename = f"edge_{safe_name}.wav"
            filepath = output_dir / filename

            # Generate audio
            communicate = edge_tts.Communicate(text, short_name)
            await communicate.save(str(filepath))

            # Get file size as proxy for success
            if filepath.exists() and filepath.stat().st_size > 0:
                print(
                    f"  [{i+1:3d}/{len(filtered)}] [OK] {short_name} ({gender}, {locale})"
                )
                generated += 1
            else:
                print(f"  [{i+1:3d}/{len(filtered)}] [FAIL] {short_name} - Empty file")
                failed += 1

        except Exception as e:
            print(f"  [{i+1:3d}/{len(filtered)}] [FAIL] {short_name}: {e}")
            failed += 1

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Generated: {generated}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(generated,1):.2f}s per voice)")
    print(f"  Output: {output_dir}")

    if generated > 0:
        print(f"\n[OK] Preview files saved to: {output_dir}")

    return generated, failed


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Preview Edge TTS voices for WakeBuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available voices
    uv run python scripts/preview_edge_tts.py --list-voices
    
    # List only English voices
    uv run python scripts/preview_edge_tts.py --list-voices --locale en
    
    # Generate with all voices
    uv run python scripts/preview_edge_tts.py "hey siri"
    
    # Generate with English voices only
    uv run python scripts/preview_edge_tts.py "hey siri" --locale en
    
    # Generate with female voices only
    uv run python scripts/preview_edge_tts.py "hey siri" --gender Female
    
    # Generate with first 50 voices
    uv run python scripts/preview_edge_tts.py "hey siri" --max-voices 50
    
    # Generate with European languages
    uv run python scripts/preview_edge_tts.py "hello" --locale en,fr,de,es,it
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="hello world",
        help="Text to synthesize (default: 'hello world')",
    )
    parser.add_argument(
        "--locale",
        "-l",
        type=str,
        help="Filter by locale prefix (e.g., 'en', 'fr', 'de', or 'en,fr,de' for multiple)",
    )
    parser.add_argument(
        "--gender",
        "-g",
        type=str,
        choices=["Male", "Female"],
        help="Filter by gender",
    )
    parser.add_argument(
        "--max-voices",
        "-n",
        type=int,
        help="Maximum number of voices to generate",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: data/temp/edge_tts_preview)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices and exit",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Fetch voices from Edge TTS API instead of local file",
    )

    args = parser.parse_args()

    # Get voices
    voice_file = Path(__file__).parent / "list of voices available in Edge TTS.txt"

    if args.use_api:
        print("Fetching voices from Edge TTS API...")
        try:
            voices = asyncio.run(get_voices())
        except Exception as e:
            print(f"[ERROR] Failed to fetch voices: {e}")
            print("Falling back to local file...")
            voices = parse_voice_file(voice_file)
    else:
        if voice_file.exists():
            voices = parse_voice_file(voice_file)
        else:
            print("Voice file not found, fetching from API...")
            try:
                voices = asyncio.run(get_voices())
            except Exception as e:
                print(f"[ERROR] Failed to fetch voices: {e}")
                return 1

    print(f"Loaded {len(voices)} voices")

    # Handle multiple locales
    locale_filter = args.locale
    if locale_filter and "," in locale_filter:
        # Multiple locales - filter manually
        locales = [l.strip() for l in locale_filter.split(",")]
        voices = [
            v
            for v in voices
            if any(v.get("Locale", "").lower().startswith(l.lower()) for l in locales)
        ]
        locale_filter = None  # Already filtered

    # List voices
    if args.list_voices:
        list_voices(voices, locale_filter, args.gender)
        return 0

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "temp" / "edge_tts_preview"
    else:
        output_dir = args.output_dir

    # Check edge-tts is installed
    try:
        import edge_tts
    except ImportError:
        print("[ERROR] edge-tts not installed.")
        print("\nPlease install with:")
        print("  uv add edge-tts")
        return 1

    # Generate previews
    try:
        generated, failed = asyncio.run(
            generate_previews(
                text=args.text,
                voices=voices,
                output_dir=output_dir,
                locale_filter=locale_filter,
                gender_filter=args.gender,
                max_voices=args.max_voices,
            )
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
