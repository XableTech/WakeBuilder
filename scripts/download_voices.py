#!/usr/bin/env python3
"""
Download Piper TTS voice models for WakeBuilder.

This script downloads a diverse set of English voice models from Hugging Face
to enable synthetic data augmentation during wake word training.

Voice models are stored in the tts_voices directory and include both
.onnx model files and their corresponding .onnx.json metadata files.
"""

import json
import sys
import urllib.request
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.config import Config

# Hugging Face base URL for Piper voices
HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Voice models to download - selected for diversity in gender, accent, and quality
# Format: (language, locale, voice_name, quality)
# Quality options: low, medium, high (higher = better quality but larger file)
VOICE_MODELS = [
    # Female voices - diverse selection
    ("en", "en_US", "amy", "medium"),  # Clear female voice
    ("en", "en_US", "lessac", "medium"),  # Professional female voice
    ("en", "en_US", "kathleen", "low"),  # Casual female voice
    ("en", "en_US", "kristin", "medium"),  # Natural female voice
    # Male voices - diverse selection
    ("en", "en_US", "ryan", "medium"),  # Clear male voice
    ("en", "en_US", "joe", "medium"),  # Natural male voice
    ("en", "en_US", "danny", "low"),  # Casual male voice
    ("en", "en_US", "john", "medium"),  # Professional male voice
    # British English for accent diversity
    ("en", "en_GB", "alan", "medium"),  # British male
    ("en", "en_GB", "alba", "medium"),  # British female (Scottish)
]


def get_voice_url(lang: str, locale: str, voice: str, quality: str) -> tuple[str, str]:
    """
    Construct download URLs for voice model files.

    Args:
        lang: Language code (e.g., 'en')
        locale: Locale code (e.g., 'en_US')
        voice: Voice name (e.g., 'amy')
        quality: Quality level (e.g., 'medium')

    Returns:
        Tuple of (onnx_url, json_url)
    """
    base_name = f"{locale}-{voice}-{quality}"
    voice_path = f"{lang}/{locale}/{voice}/{quality}"

    onnx_url = f"{HF_BASE_URL}/{voice_path}/{base_name}.onnx"
    json_url = f"{HF_BASE_URL}/{voice_path}/{base_name}.onnx.json"

    return onnx_url, json_url


def download_file(
    url: str,
    dest_path: Path,
    show_progress: bool = True,
) -> bool:
    """
    Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Destination file path
        show_progress: Whether to show download progress

    Returns:
        True if download successful, False otherwise
    """
    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file already exists
        if dest_path.exists():
            print(f"  [SKIP] Already exists: {dest_path.name}")
            return True

        print(f"  Downloading: {dest_path.name}...")

        # Download with progress reporting
        def report_progress(block_num: int, block_size: int, total_size: int) -> None:
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)

        if show_progress:
            print()  # New line after progress

        print(f"  [OK] Downloaded: {dest_path.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"  [ERROR] URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def verify_voice_model(onnx_path: Path, json_path: Path) -> bool:
    """
    Verify that a voice model is valid.

    Args:
        onnx_path: Path to .onnx file
        json_path: Path to .onnx.json file

    Returns:
        True if valid, False otherwise
    """
    # Check files exist
    if not onnx_path.exists():
        return False
    if not json_path.exists():
        return False

    # Check JSON is valid
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Verify required fields
        required_fields = ["audio", "espeak"]
        for field in required_fields:
            if field not in config:
                print(f"  [WARN] Missing field '{field}' in {json_path.name}")
                return False

        return True

    except json.JSONDecodeError:
        print(f"  [ERROR] Invalid JSON: {json_path.name}")
        return False


def download_voice(
    lang: str,
    locale: str,
    voice: str,
    quality: str,
    voices_dir: Path,
) -> bool:
    """
    Download a single voice model.

    Args:
        lang: Language code
        locale: Locale code
        voice: Voice name
        quality: Quality level
        voices_dir: Directory to save voice files

    Returns:
        True if successful, False otherwise
    """
    voice_name = f"{locale}-{voice}-{quality}"
    print(f"\nDownloading voice: {voice_name}")

    # Get URLs
    onnx_url, json_url = get_voice_url(lang, locale, voice, quality)

    # Define destination paths
    onnx_path = voices_dir / f"{voice_name}.onnx"
    json_path = voices_dir / f"{voice_name}.onnx.json"

    # Download files
    onnx_ok = download_file(onnx_url, onnx_path)
    json_ok = download_file(json_url, json_path)

    if not onnx_ok or not json_ok:
        return False

    # Verify model
    if not verify_voice_model(onnx_path, json_path):
        print(f"  [ERROR] Verification failed for {voice_name}")
        return False

    print(f"  [OK] Voice ready: {voice_name}")
    return True


def create_voices_index(voices_dir: Path) -> None:
    """
    Create an index file listing all available voices.

    Args:
        voices_dir: Directory containing voice files
    """
    voices = []

    for onnx_file in sorted(voices_dir.glob("*.onnx")):
        json_file = onnx_file.with_suffix(".onnx.json")

        if not json_file.exists():
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            voice_info = {
                "name": onnx_file.stem,
                "onnx_file": onnx_file.name,
                "json_file": json_file.name,
                "sample_rate": config.get("audio", {}).get("sample_rate", 22050),
                "language": config.get("espeak", {}).get("voice", "en-us"),
            }
            voices.append(voice_info)

        except (json.JSONDecodeError, KeyError):
            continue

    # Save index
    index_path = voices_dir / "voices_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "voices": voices,
                "count": len(voices),
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Created voices index: {index_path.name}")
    print(f"     Total voices available: {len(voices)}")


def main() -> int:
    """Main execution function."""
    print("=" * 70)
    print("WakeBuilder - Piper TTS Voice Download Script")
    print("=" * 70)

    # Configuration
    config = Config()
    voices_dir = Path(config.TTS_VOICES_DIR)

    print(f"\nVoices directory: {voices_dir}")
    print(f"Voices to download: {len(VOICE_MODELS)}")

    # Ensure directory exists
    voices_dir.mkdir(parents=True, exist_ok=True)

    # Download each voice
    successful = 0
    failed = 0

    for lang, locale, voice, quality in VOICE_MODELS:
        if download_voice(lang, locale, voice, quality, voices_dir):
            successful += 1
        else:
            failed += 1

    # Create index
    print("\n" + "=" * 70)
    print("Creating voices index...")
    create_voices_index(voices_dir)

    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(VOICE_MODELS)}")

    if failed > 0:
        print("\n[WARN] Some voices failed to download.")
        print("       You can re-run this script to retry failed downloads.")
        return 1

    print("\n[OK] All voices downloaded successfully!")
    print(f"\nVoices are stored in: {voices_dir}")
    print("\nNext steps:")
    print("  1. Implement TTS generator wrapper")
    print(
        '  2. Test voice synthesis with: uv run python -c "from wakebuilder.tts import TTSGenerator; ..."'
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
