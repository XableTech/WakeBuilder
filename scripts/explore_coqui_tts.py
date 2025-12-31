"""
Coqui TTS Voice Exploration Script
===================================
Explores all available Coqui TTS models and their voices.
Useful for investigating multi-speaker models for positive sample generation.

Package: coqui-tts (supports Python 3.10 - 3.13)
Repository: https://github.com/idiap/coqui-ai-TTS (community fork)

Usage:
    uv add coqui-tts
    uv run python scripts/explore_coqui_tts.py --list-models
    uv run python scripts/explore_coqui_tts.py --list-voices tts_models/en/vctk/vits
    uv run python scripts/explore_coqui_tts.py --generate "Hello world" --model tts_models/en/vctk/vits --num-speakers 5
"""

import argparse
import os
from pathlib import Path


def list_all_models():
    """List all available Coqui TTS models."""
    from TTS.api import TTS
    
    print("=" * 80)
    print("AVAILABLE COQUI TTS MODELS")
    print("=" * 80)
    
    models = TTS().list_models()
    
    # Categorize models
    tts_models = [m for m in models if m.startswith("tts_models")]
    vocoder_models = [m for m in models if m.startswith("vocoder_models")]
    voice_conversion = [m for m in models if m.startswith("voice_conversion")]
    
    print(f"\n📢 TTS Models ({len(tts_models)}):")
    print("-" * 40)
    
    # Group by language
    by_lang = {}
    for model in tts_models:
        parts = model.split("/")
        if len(parts) >= 2:
            lang = parts[1]
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(model)
    
    for lang in sorted(by_lang.keys()):
        print(f"\n  [{lang.upper()}] ({len(by_lang[lang])} models)")
        for model in by_lang[lang]:
            print(f"    • {model}")
    
    print(f"\n🔊 Vocoder Models ({len(vocoder_models)}):")
    print("-" * 40)
    for model in vocoder_models:  # Show all
        print(f"  • {model}")
    
    if voice_conversion:
        print(f"\n🎭 Voice Conversion Models ({len(voice_conversion)}):")
        print("-" * 40)
        for model in voice_conversion:
            print(f"  • {model}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED MULTI-SPEAKER MODELS FOR WAKE WORD GENERATION:")
    print("=" * 80)
    
    recommended = [
        ("tts_models/en/vctk/vits", "109 English speakers, high quality"),
        ("tts_models/en/ljspeech/vits", "Single speaker, very natural"),
        ("tts_models/multilingual/multi-dataset/your_tts", "Multi-lingual, voice cloning capable"),
        ("tts_models/multilingual/multi-dataset/xtts_v2", "Latest XTTS, 17 languages, voice cloning"),
        ("tts_models/en/jenny/jenny", "High quality female voice"),
    ]
    
    for model, desc in recommended:
        if model in tts_models:
            print(f"  ✅ {model}")
            print(f"     {desc}")
        else:
            print(f"  ❌ {model} (not available)")
            print(f"     {desc}")
    
    return models


def list_model_voices(model_name: str):
    """List all voices/speakers for a specific model."""
    from TTS.api import TTS
    
    print(f"\n{'=' * 80}")
    print(f"VOICES FOR MODEL: {model_name}")
    print("=" * 80)
    
    try:
        tts = TTS(model_name)
        
        # Check for speakers
        if hasattr(tts, 'speakers') and tts.speakers:
            print(f"\n👥 Speakers ({len(tts.speakers)}):")
            print("-" * 40)
            for i, speaker in enumerate(tts.speakers):
                print(f"  {i:3d}. {speaker}")
        else:
            print("\n  ℹ️  This is a single-speaker model (no speaker selection)")
        
        # Check for languages
        if hasattr(tts, 'languages') and tts.languages:
            print(f"\n🌍 Languages ({len(tts.languages)}):")
            print("-" * 40)
            for lang in tts.languages:
                print(f"  • {lang}")
        
        return tts
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return None


def generate_samples(
    text: str,
    model_name: str,
    output_dir: str = "tts_samples",
    num_speakers: int = 5,
    language: str = None
):
    """Generate audio samples with multiple speakers."""
    from TTS.api import TTS
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"GENERATING SAMPLES")
    print("=" * 80)
    print(f"  Text: \"{text}\"")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_path}")
    
    try:
        tts = TTS(model_name)
        
        # Multi-speaker model
        if hasattr(tts, 'speakers') and tts.speakers:
            speakers_to_use = tts.speakers[:num_speakers]
            print(f"\n  Generating {len(speakers_to_use)} samples...")
            print("-" * 40)
            
            for i, speaker in enumerate(speakers_to_use):
                safe_speaker = speaker.replace("/", "_").replace(" ", "_")
                filename = output_path / f"sample_{i:03d}_{safe_speaker}.wav"
                
                kwargs = {"text": text, "speaker": speaker, "file_path": str(filename)}
                if language and hasattr(tts, 'languages') and tts.languages:
                    kwargs["language"] = language
                
                tts.tts_to_file(**kwargs)
                print(f"  ✅ Generated: {filename.name}")
        
        # Single-speaker model
        else:
            filename = output_path / "sample_single_speaker.wav"
            kwargs = {"text": text, "file_path": str(filename)}
            if language and hasattr(tts, 'languages') and tts.languages:
                kwargs["language"] = language
            
            tts.tts_to_file(**kwargs)
            print(f"  ✅ Generated: {filename.name}")
        
        print(f"\n🎉 Done! Samples saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error generating samples: {e}")
        raise


def explore_vits_models():
    """Specifically explore VITS models which are known for quality."""
    from TTS.api import TTS
    
    print("\n" + "=" * 80)
    print("EXPLORING VITS MODELS (High Quality Multi-Speaker)")
    print("=" * 80)
    
    vits_models = [
        "tts_models/en/vctk/vits",           # 109 speakers
        "tts_models/en/ljspeech/vits",       # Single speaker, high quality
        "tts_models/de/thorsten/vits",       # German
        "tts_models/fr/mai/vits",            # French
        "tts_models/es/mai/vits",            # Spanish (if available)
    ]
    
    all_models = TTS().list_models()
    
    for model in vits_models:
        if model not in all_models:
            print(f"\n❌ {model} - Not available")
            continue
            
        print(f"\n📦 {model}")
        print("-" * 60)
        
        try:
            tts = TTS(model)
            
            if hasattr(tts, 'speakers') and tts.speakers:
                print(f"   Speakers: {len(tts.speakers)}")
                # Show first 10 speakers
                for speaker in tts.speakers[:10]:
                    print(f"     • {speaker}")
                if len(tts.speakers) > 10:
                    print(f"     ... and {len(tts.speakers) - 10} more")
            else:
                print("   Single speaker model")
                
            if hasattr(tts, 'languages') and tts.languages:
                print(f"   Languages: {', '.join(tts.languages)}")
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")


def interactive_demo():
    """Interactive demo to test different voices."""
    from TTS.api import TTS
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    
    print("\n" + "=" * 80)
    print("INTERACTIVE COQUI TTS DEMO")
    print("=" * 80)
    
    # Load VCTK model (109 speakers)
    print("\nLoading tts_models/en/vctk/vits (109 speakers)...")
    tts = TTS("tts_models/en/vctk/vits")
    
    print(f"\nAvailable speakers: {len(tts.speakers)}")
    print("\nCommands:")
    print("  list          - List all speakers")
    print("  say <text>    - Generate with random speaker")
    print("  speaker <n>   - Select speaker by index")
    print("  quit          - Exit")
    
    current_speaker_idx = 0
    
    while True:
        try:
            cmd = input(f"\n[Speaker {current_speaker_idx}: {tts.speakers[current_speaker_idx]}] > ").strip()
            
            if cmd == "quit":
                break
            elif cmd == "list":
                for i, s in enumerate(tts.speakers):
                    marker = "→" if i == current_speaker_idx else " "
                    print(f"  {marker} {i:3d}. {s}")
            elif cmd.startswith("speaker "):
                try:
                    idx = int(cmd.split()[1])
                    if 0 <= idx < len(tts.speakers):
                        current_speaker_idx = idx
                        print(f"Selected: {tts.speakers[idx]}")
                    else:
                        print(f"Invalid index. Range: 0-{len(tts.speakers)-1}")
                except ValueError:
                    print("Usage: speaker <number>")
            elif cmd.startswith("say "):
                text = cmd[4:]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tts.tts_to_file(
                        text=text,
                        speaker=tts.speakers[current_speaker_idx],
                        file_path=f.name
                    )
                    # Play audio
                    data, samplerate = sf.read(f.name)
                    sd.play(data, samplerate)
                    sd.wait()
                    os.unlink(f.name)
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Explore Coqui TTS models and voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python explore_coqui_tts.py --list-models
  
  # List voices for a specific model
  python explore_coqui_tts.py --list-voices tts_models/en/vctk/vits
  
  # Generate samples with multiple speakers
  python explore_coqui_tts.py --generate "Hey Siri" --model tts_models/en/vctk/vits --num-speakers 10
  
  # Explore all VITS models
  python explore_coqui_tts.py --explore-vits
  
  # Interactive demo (requires sounddevice)
  python explore_coqui_tts.py --interactive
        """
    )
    
    parser.add_argument("--list-models", action="store_true",
                        help="List all available TTS models")
    parser.add_argument("--list-voices", type=str, metavar="MODEL",
                        help="List voices for a specific model")
    parser.add_argument("--generate", type=str, metavar="TEXT",
                        help="Generate audio samples with the given text")
    parser.add_argument("--model", type=str, default="tts_models/en/vctk/vits",
                        help="Model to use for generation (default: tts_models/en/vctk/vits)")
    parser.add_argument("--num-speakers", type=int, default=5,
                        help="Number of speakers to generate (default: 5)")
    parser.add_argument("--output-dir", type=str, default="tts_samples",
                        help="Output directory for samples (default: tts_samples)")
    parser.add_argument("--language", type=str,
                        help="Language code for multilingual models")
    parser.add_argument("--explore-vits", action="store_true",
                        help="Explore all VITS models in detail")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive demo (requires sounddevice)")
    
    args = parser.parse_args()
    
    # Default action if no args
    if not any([args.list_models, args.list_voices, args.generate, 
                args.explore_vits, args.interactive]):
        args.list_models = True
    
    if args.list_models:
        list_all_models()
    
    if args.list_voices:
        list_model_voices(args.list_voices)
    
    if args.generate:
        generate_samples(
            text=args.generate,
            model_name=args.model,
            output_dir=args.output_dir,
            num_speakers=args.num_speakers,
            language=args.language
        )
    
    if args.explore_vits:
        explore_vits_models()
    
    if args.interactive:
        interactive_demo()


if __name__ == "__main__":
    main()
