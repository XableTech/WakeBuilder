"""
Coqui TTS Complete Model Discovery Script
Extract ALL multi-speaker voices you haven't used yet

Based on your usage:
- VCTK: 109 voices ✓ (already used)
- YourTTS: Multiple speakers ✓ (already used)
- Catalan models ✓ (already used)
Total so far: 390+ voices

This script finds the remaining multi-speaker models
"""

from TTS.api import TTS
import os


def discover_all_models():
    """List all available Coqui TTS models"""
    print("=" * 70)
    print("DISCOVERING ALL COQUI TTS MODELS")
    print("=" * 70)

    tts = TTS()
    all_models = tts.list_models()

    # Filter for TTS models only (not vocoders)
    tts_models = [m for m in all_models if m.startswith("tts_models/")]

    print(f"\nTotal TTS models available: {len(tts_models)}\n")

    # Categorize by language
    by_language = {}
    for model in tts_models:
        parts = model.split("/")
        if len(parts) >= 3:
            lang = parts[1]
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(model)

    return tts_models, by_language


def find_multispeaker_models():
    """Identify which models are multi-speaker"""
    print("\n" + "=" * 70)
    print("IDENTIFYING MULTI-SPEAKER MODELS")
    print("=" * 70)

    tts_models, by_language = discover_all_models()

    multispeaker_models = []
    single_speaker_models = []

    # Known multi-speaker models (you've already used these)
    already_used = [
        "tts_models/en/vctk/vits",  # 109 speakers
        "tts_models/multilingual/multi-dataset/your_tts",
        "tts_models/ca/custom/vits",  # Catalan
    ]

    # Models to check for multi-speaker capability
    print("\nChecking models for multi-speaker support...\n")

    for model_name in tts_models:
        if model_name in already_used:
            print(f"✓ {model_name} (ALREADY USED)")
            continue

        # Skip XTTS (requires license)
        if "xtts" in model_name.lower():
            print(f"⊗ {model_name} (REQUIRES LICENSE)")
            continue

        # Check if model has multiple speakers
        try:
            print(f"Checking: {model_name}...", end=" ")
            tts_temp = TTS(model_name)

            if hasattr(tts_temp, "speakers") and tts_temp.speakers:
                speaker_count = len(tts_temp.speakers)
                print(f"✓ MULTI-SPEAKER ({speaker_count} speakers)")
                multispeaker_models.append(
                    {
                        "model": model_name,
                        "speakers": tts_temp.speakers,
                        "count": speaker_count,
                    }
                )
            else:
                print("✗ Single speaker")
                single_speaker_models.append(model_name)

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")

    return multispeaker_models, single_speaker_models, already_used


def analyze_language_models():
    """Find multi-speaker models by language"""
    print("\n" + "=" * 70)
    print("LANGUAGE-SPECIFIC MULTI-SPEAKER MODELS")
    print("=" * 70)

    _, by_language = discover_all_models()

    # Languages with known multi-speaker datasets
    priority_languages = [
        "en",
        "de",
        "es",
        "fr",
        "it",
        "pt",
        "pl",
        "nl",  # European
        "multilingual",  # Multi-language models
        "ja",
        "zh",
        "ko",  # Asian
    ]

    print("\nModels by priority languages:\n")

    for lang in priority_languages:
        if lang in by_language:
            print(f"\n{lang.upper()} ({len(by_language[lang])} models):")
            for model in by_language[lang]:
                # Highlight potentially multi-speaker models
                if any(
                    keyword in model
                    for keyword in ["vctk", "vits", "multi", "your_tts"]
                ):
                    print(f"  🎯 {model}")
                else:
                    print(f"     {model}")


def extract_all_speakers(model_name, output_dir="coqui_voices"):
    """Extract all speakers from a multi-speaker model"""
    print(f"\n{'=' * 70}")
    print(f"EXTRACTING SPEAKERS FROM: {model_name}")
    print(f"{'=' * 70}\n")

    os.makedirs(output_dir, exist_ok=True)

    try:
        tts = TTS(model_name)

        if not hasattr(tts, "speakers") or not tts.speakers:
            print("❌ This model doesn't have multiple speakers")
            return 0

        speakers = tts.speakers
        print(f"Found {len(speakers)} speakers!\n")

        text = "The quick brown fox jumps over the lazy dog."

        for i, speaker in enumerate(speakers):
            output_path = os.path.join(
                output_dir, f"{model_name.replace('/', '_')}_{i:03d}_{speaker}.wav"
            )

            print(f"Generating {i+1}/{len(speakers)}: {speaker}...", end=" ")

            try:
                tts.tts_to_file(text=text, speaker=speaker, file_path=output_path)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")

        print(f"\n✅ Generated {len(speakers)} voices to '{output_dir}'")
        return len(speakers)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 0


def discover_new_models_only():
    """Find models you haven't used yet"""
    print("=" * 70)
    print("NEW MODELS TO EXPLORE (excluding already used)")
    print("=" * 70)

    # Models you've already used
    used = {
        "tts_models/en/vctk/vits": 109,
        "tts_models/multilingual/multi-dataset/your_tts": "?",
        "tts_models/ca/custom/vits": "?",
    }

    print("\nAlready extracted:")
    for model, count in used.items():
        print(f"  ✓ {model} ({count} speakers)")

    print("\nTotal voices so far: 390+\n")
    print("=" * 70)

    # Potential new multi-speaker models to check
    candidates = [
        # Other interesting models
        "tts_models/multilingual/multi-dataset/bark",
    ]

    print("\nChecking candidate models for multi-speaker support:\n")

    new_multispeaker = []

    for model in candidates:
        if model in used:
            continue

        try:
            print(f"Checking: {model}...", end=" ")
            tts = TTS(model)

            if hasattr(tts, "speakers") and tts.speakers:
                count = len(tts.speakers)
                print(f"✓ FOUND! ({count} speakers)")
                new_multispeaker.append(
                    {"model": model, "count": count, "speakers": tts.speakers}
                )
            else:
                print("✗ Single speaker")

        except Exception:
            print("✗ Not available or error")

    if new_multispeaker:
        print("\n" + "=" * 70)
        print("🎉 NEW MULTI-SPEAKER MODELS FOUND!")
        print("=" * 70)

        total_new_voices = 0
        for model_info in new_multispeaker:
            total_new_voices += model_info["count"]
            print(f"\n{model_info['model']}")
            print(f"  Speakers: {model_info['count']}")
            print(f"  Examples: {', '.join(model_info['speakers'])}")

        print(f"\n{'=' * 70}")
        print(f"POTENTIAL NEW VOICES: {total_new_voices}")
        print(f"CURRENT TOTAL: 390+ + {total_new_voices} = {390 + total_new_voices}+")
        print(f"{'=' * 70}")

        return new_multispeaker
    else:
        print("\n❌ No new multi-speaker models found in candidates")
        return []


def bulk_extract_new_voices():
    """Extract all voices from newly discovered models"""
    new_models = discover_new_models_only()

    if not new_models:
        print("\nNo new models to extract from.")
        return

    print("\n" + "=" * 70)
    print("BULK EXTRACTION")
    print("=" * 70)

    choice = input("\nExtract all speakers from new models? (y/n): ").strip().lower()

    if choice != "y":
        print("Extraction cancelled.")
        return

    total_extracted = 0

    for model_info in new_models:
        count = extract_all_speakers(model_info["model"])
        total_extracted += count

    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print(f"Total new voices extracted: {total_extracted}")
    print(f"Grand total: 390+ + {total_extracted} = {390 + total_extracted}+ voices!")
    print("=" * 70)


# ============================================================================
# MAIN MENU
# ============================================================================

if __name__ == "__main__":
    print("\n🎤 COQUI TTS MODEL DISCOVERY TOOL 🎤\n")
    print("Choose an option:")
    print("1. Discover ALL available models")
    print("2. Find NEW multi-speaker models (recommended)")
    print("3. Extract speakers from specific model")
    print("4. Bulk extract all new voices")
    print("5. Analyze models by language")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        discover_all_models()
    elif choice == "2":
        discover_new_models_only()
    elif choice == "3":
        model = input("Enter model name: ").strip()
        extract_all_speakers(model)
    elif choice == "4":
        bulk_extract_new_voices()
    elif choice == "5":
        analyze_language_models()
    else:
        print("Invalid choice!")
