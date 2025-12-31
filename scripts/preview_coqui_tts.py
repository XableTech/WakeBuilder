#!/usr/bin/env python3
"""
Preview Coqui TTS voices for WakeBuilder.

This script generates preview audio samples using Coqui TTS models
with multiple speakers and languages, saving them to a temporary folder.

Package: coqui-tts (supports Python 3.10 - 3.13)

Usage:
    uv run python scripts/preview_coqui_tts.py "hey siri"
    uv run python scripts/preview_coqui_tts.py "hello world" --model vctk --num-speakers 10
    uv run python scripts/preview_coqui_tts.py "bonjour" --model your_tts --language fr-fr
    uv run python scripts/preview_coqui_tts.py --list-models
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import soundfile as sf


# Add espeak-ng to PATH on Windows if installed
if sys.platform == "win32":
    espeak_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
    ]
    for esp_path in espeak_paths:
        if os.path.exists(esp_path) and esp_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = esp_path + os.pathsep + os.environ.get("PATH", "")
            break


# Model configurations with their speakers and languages
COQUI_MODELS = {
    "vctk": {
        "model_name": "tts_models/en/vctk/vits",
        "description": "109 English speakers, high quality VITS",
        "multi_speaker": True,
        "languages": ["en"],
    },
    "your_tts": {
        "model_name": "tts_models/multilingual/multi-dataset/your_tts",
        "description": "Multi-lingual model (EN, FR, PT)",
        "multi_speaker": True,
        "languages": ["en", "fr-fr", "pt-br"],
    },
    "tortoise": {
        "model_name": "tts_models/en/multi-dataset/tortoise-v2",
        "description": "High quality single speaker (slow)",
        "multi_speaker": False,
        "languages": ["en"],
    },
    "ljspeech": {
        "model_name": "tts_models/en/ljspeech/vits",
        "description": "Single speaker, very natural",
        "multi_speaker": False,
        "languages": ["en"],
    },
    # European languages
    "german": {
        "model_name": "tts_models/de/thorsten/vits",
        "description": "German single speaker VITS",
        "multi_speaker": False,
        "languages": ["de"],
    },
    "czech": {
        "model_name": "tts_models/cs/cv/vits",
        "description": "Czech VITS",
        "multi_speaker": False,
        "languages": ["cs"],
    },
    "slovak": {
        "model_name": "tts_models/sk/cv/vits",
        "description": "Slovak VITS",
        "multi_speaker": False,
        "languages": ["sk"],
    },
    "slovenian": {
        "model_name": "tts_models/sl/cv/vits",
        "description": "Slovenian VITS",
        "multi_speaker": False,
        "languages": ["sl"],
    },
    "catalan": {
        "model_name": "tts_models/ca/custom/vits",
        "description": "Catalan VITS",
        "multi_speaker": False,
        "languages": ["ca"],
    },
    "portuguese": {
        "model_name": "tts_models/pt/cv/vits",
        "description": "Portuguese VITS",
        "multi_speaker": False,
        "languages": ["pt"],
    },
}


def list_models():
    """List all available Coqui TTS models."""
    print("=" * 70)
    print("Available Coqui TTS Models")
    print("=" * 70)
    
    # Group by type
    multi_speaker = {k: v for k, v in COQUI_MODELS.items() if v["multi_speaker"]}
    single_speaker = {k: v for k, v in COQUI_MODELS.items() if not v["multi_speaker"]}
    
    print(f"\n--- Multi-Speaker Models ({len(multi_speaker)}) ---")
    for key, info in multi_speaker.items():
        langs = ", ".join(info["languages"])
        print(f"  {key:18} - {info['description']} [{langs}]")
    
    print(f"\n--- Single-Speaker Models ({len(single_speaker)}) ---")
    
    # Group by language region
    european = ["german", "french", "spanish", "italian_female", "italian_male", 
                "dutch", "polish", "hungarian", "finnish", "ukrainian", "bulgarian",
                "czech", "danish", "greek", "croatian", "romanian", "slovak", 
                "slovenian", "swedish", "catalan", "portuguese"]
    english = ["ljspeech", "jenny", "tortoise"]
    
    print("\n  English:")
    for key in english:
        if key in single_speaker:
            info = single_speaker[key]
            print(f"    {key:18} - {info['description']}")
    
    print("\n  European Languages:")
    for key in european:
        if key in single_speaker:
            info = single_speaker[key]
            langs = ", ".join(info["languages"])
            print(f"    {key:18} - {info['description']} [{langs}]")


def list_speakers(model_key: str):
    """List speakers for a specific model."""
    from TTS.api import TTS
    
    if model_key not in COQUI_MODELS:
        print(f"[ERROR] Unknown model: {model_key}")
        return
    
    model_info = COQUI_MODELS[model_key]
    print(f"\nLoading model: {model_info['model_name']}...")
    
    tts = TTS(model_info["model_name"])
    
    if hasattr(tts, "speakers") and tts.speakers:
        print(f"\n--- Speakers ({len(tts.speakers)}) ---")
        for i, speaker in enumerate(tts.speakers):
            print(f"  {i:3d}. {speaker}")
    else:
        print("  Single speaker model (no speaker selection)")
    
    if hasattr(tts, "languages") and tts.languages:
        print(f"\n--- Languages ({len(tts.languages)}) ---")
        for lang in tts.languages:
            print(f"  • {lang}")


def generate_previews(
    text: str,
    model_keys: list[str],
    output_dir: Path,
    num_speakers: int = 5,
    language: str | None = None,
    use_gpu: bool = True,
):
    """Generate preview audio samples."""
    from TTS.api import TTS
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Coqui TTS Preview Generator")
    print("=" * 70)
    print(f"\nText: \"{text}\"")
    print(f"Models: {model_keys}")
    print(f"Max speakers per model: {num_speakers}")
    print(f"Language override: {language or 'auto'}")
    print(f"Output: {output_dir}")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    
    start_time = time.time()
    generated = 0
    failed = 0
    
    for model_key in model_keys:
        if model_key not in COQUI_MODELS:
            print(f"\n[WARN] Unknown model: {model_key}, skipping...")
            continue
        
        model_info = COQUI_MODELS[model_key]
        model_name = model_info["model_name"]
        
        print(f"\n" + "-" * 70)
        print(f"Model: {model_key} ({model_info['description']})")
        print(f"Loading: {model_name}...")
        
        try:
            tts = TTS(model_name, gpu=use_gpu)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            failed += 1
            continue
        
        # Get speakers
        speakers = []
        if hasattr(tts, "speakers") and tts.speakers:
            if num_speakers == -1:
                speakers = tts.speakers  # All speakers
            else:
                speakers = tts.speakers[:num_speakers]
        else:
            speakers = [None]  # Single speaker
        
        # Get languages
        languages = [language] if language else model_info["languages"]
        if hasattr(tts, "languages") and tts.languages:
            # Filter to supported languages
            languages = [l for l in languages if l in tts.languages]
            if not languages:
                languages = [tts.languages[0]]  # Default to first
        
        print(f"Speakers: {len(speakers)}, Languages: {languages}")
        
        for speaker in speakers:
            for lang in languages:
                try:
                    # Build filename - sanitize speaker name (remove newlines, special chars)
                    speaker_str = speaker if speaker else "default"
                    speaker_str = re.sub(r'[\n\r\t/\\:*?"<>|]', '_', speaker_str).strip()
                    lang_str = lang.replace("-", "_") if lang else "default"
                    filename = f"coqui_{model_key}_{speaker_str}_{lang_str}.wav"
                    filepath = output_dir / filename
                    
                    # Build TTS kwargs
                    kwargs = {"text": text, "file_path": str(filepath)}
                    if speaker:
                        kwargs["speaker"] = speaker
                    if lang and hasattr(tts, "languages") and tts.languages:
                        kwargs["language"] = lang
                    
                    # Generate
                    tts.tts_to_file(**kwargs)
                    
                    # Get duration
                    audio, sr = sf.read(filepath)
                    duration = len(audio) / sr
                    
                    print(f"  [OK] {filename} ({duration:.2f}s)")
                    generated += 1
                    
                except Exception as e:
                    print(f"  [FAIL] {speaker or 'default'} / {lang}: {e}")
                    failed += 1
    
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
    
    return generated, failed


def main() -> int:
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Preview Coqui TTS voices for WakeBuilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available models
    uv run python scripts/preview_coqui_tts.py --list-models
    
    # List speakers for a model
    uv run python scripts/preview_coqui_tts.py --list-speakers vctk
    
    # Generate with VCTK (109 English speakers)
    uv run python scripts/preview_coqui_tts.py "hey siri" --model vctk --num-speakers 10
    
    # Generate with YourTTS in French
    uv run python scripts/preview_coqui_tts.py "bonjour le monde" --model your_tts --language fr-fr
    
    # Generate with multiple European languages
    uv run python scripts/preview_coqui_tts.py "hello" --model german french spanish italian_female
    
    # Generate all European language samples
    uv run python scripts/preview_coqui_tts.py "test" --european
        """,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="hello world",
        help="Text to synthesize (default: 'hello world')",
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        default=["vctk"],
        help="Model(s) to use (default: vctk)",
    )
    parser.add_argument(
        "--num-speakers", "-n",
        type=int,
        default=5,
        help="Max speakers per multi-speaker model (default: 5, use -1 for all)",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        help="Language code for multilingual models (e.g., en, fr-fr, pt-br)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: data/temp/coqui_tts_preview)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--list-speakers",
        type=str,
        metavar="MODEL",
        help="List speakers for a specific model and exit",
    )
    parser.add_argument(
        "--european",
        action="store_true",
        help="Use all European language models",
    )
    parser.add_argument(
        "--all-english",
        action="store_true",
        help="Use all English models (vctk, ljspeech, jenny)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use ALL available models (multi-speaker + European languages)",
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        list_models()
        return 0
    
    # List speakers
    if args.list_speakers:
        try:
            list_speakers(args.list_speakers)
        except Exception as e:
            print(f"[ERROR] {e}")
            return 1
        return 0
    
    # Determine models to use
    models = args.model
    
    if args.all:
        models = list(COQUI_MODELS.keys())
    elif args.european:
        models = [
            "german", "french", "spanish", "italian_female", "italian_male",
            "dutch", "polish", "hungarian", "finnish", "ukrainian", "bulgarian",
            "czech", "danish", "greek", "croatian", "romanian", "slovak",
            "slovenian", "swedish", "catalan", "portuguese"
        ]
    elif args.all_english:
        models = ["vctk", "ljspeech", "jenny"]
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "temp" / "coqui_tts_preview"
    else:
        output_dir = args.output_dir
    
    # Generate previews
    try:
        generated, failed = generate_previews(
            text=args.text,
            model_keys=models,
            output_dir=output_dir,
            num_speakers=args.num_speakers,
            language=args.language,
            use_gpu=not args.no_gpu,
        )
    except ImportError as e:
        print(f"[ERROR] Failed to import Coqui TTS: {e}")
        print("\nPlease install Coqui TTS with:")
        print("  uv add coqui-tts")
        return 1
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
