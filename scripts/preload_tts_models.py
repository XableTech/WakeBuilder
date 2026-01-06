#!/usr/bin/env python3
"""
Pre-download TTS models for WakeBuilder Docker build.

This script downloads all Coqui TTS and Kokoro TTS models during 
Docker image build to avoid runtime downloads.

Usage:
    python scripts/preload_tts_models.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_coqui_models():
    """
    Pre-download all Coqui TTS models.
    
    Models from coqui_generator.py:
    - vctk: tts_models/en/vctk/vits (109 speakers)
    - your_tts: tts_models/multilingual/multi-dataset/your_tts
    - tortoise: tts_models/en/multi-dataset/tortoise-v2
    - ljspeech: tts_models/en/ljspeech/vits
    - german: tts_models/de/thorsten/vits
    - czech: tts_models/cs/cv/vits
    - slovak: tts_models/sk/cv/vits
    - slovenian: tts_models/sl/cv/vits
    - catalan: tts_models/ca/custom/vits
    - portuguese: tts_models/pt/cv/vits
    """
    print("=" * 70)
    print("Downloading Coqui TTS Models")
    print("=" * 70)
    
    try:
        from TTS.api import TTS
    except ImportError:
        print("[SKIP] Coqui TTS not installed, skipping...")
        return False
    
    # All Coqui models from coqui_generator.py (excluding Tortoise - too memory intensive)
    COQUI_MODELS = [
        ("vctk", "tts_models/en/vctk/vits"),
        ("your_tts", "tts_models/multilingual/multi-dataset/your_tts"),
        # NOTE: Tortoise-v2 removed - requires 8GB+ RAM
        ("ljspeech", "tts_models/en/ljspeech/vits"),
        ("german", "tts_models/de/thorsten/vits"),
        ("czech", "tts_models/cs/cv/vits"),
        ("slovak", "tts_models/sk/cv/vits"),
        ("slovenian", "tts_models/sl/cv/vits"),
        ("catalan", "tts_models/ca/custom/vits"),
        ("portuguese", "tts_models/pt/cv/vits"),
    ]
    
    successful = 0
    failed = 0
    
    for model_key, model_name in COQUI_MODELS:
        print(f"\nDownloading: {model_key} ({model_name})")
        try:
            # This downloads the model to ~/.local/share/tts/
            tts = TTS(model_name)
            print(f"  [OK] {model_key}")
            successful += 1
            del tts
        except Exception as e:
            print(f"  [ERROR] {model_key}: {e}")
            failed += 1
    
    print(f"\nCoqui TTS: {successful} successful, {failed} failed")
    return failed == 0


def download_kokoro_model():
    """
    Pre-download Kokoro TTS model and ALL 40 voice packs.
    
    All voices from kokoro_generator.py:
    - American English Female (11): af_heart, af_bella, af_nicole, af_aoede, af_kore, 
      af_sarah, af_nova, af_sky, af_alloy, af_jessica, af_river
    - American English Male (9): am_michael, am_fenrir, am_puck, am_echo, am_eric, 
      am_liam, am_onyx, am_santa, am_adam
    - British English Female (4): bf_emma, bf_isabella, bf_alice, bf_lily
    - British English Male (4): bm_george, bm_fable, bm_lewis, bm_daniel
    - Spanish (3): ef_dora, em_alex, em_santa
    - French (1): ff_siwis
    - Hindi (4): hf_alpha, hf_beta, hm_omega, hm_psi
    - Italian (2): if_sara, im_nicola
    - Portuguese (3): pf_dora, pm_alex, pm_santa
    - Chinese (8): zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi, 
      zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang
    """
    print("\n" + "=" * 70)
    print("Downloading Kokoro TTS Model and All Voice Packs")
    print("=" * 70)
    
    try:
        from kokoro import KModel, KPipeline
    except ImportError:
        print("[SKIP] Kokoro TTS not installed, skipping...")
        return False
    
    # ALL 40 Kokoro voices from kokoro_generator.py
    ALL_KOKORO_VOICES = {
        # American English Female (11 voices) - lang_code='a'
        "af_heart": "a", "af_bella": "a", "af_nicole": "a", "af_aoede": "a",
        "af_kore": "a", "af_sarah": "a", "af_nova": "a", "af_sky": "a",
        "af_alloy": "a", "af_jessica": "a", "af_river": "a",
        # American English Male (9 voices) - lang_code='a'
        "am_michael": "a", "am_fenrir": "a", "am_puck": "a", "am_echo": "a",
        "am_eric": "a", "am_liam": "a", "am_onyx": "a", "am_santa": "a", "am_adam": "a",
        # British English Female (4 voices) - lang_code='b'
        "bf_emma": "b", "bf_isabella": "b", "bf_alice": "b", "bf_lily": "b",
        # British English Male (4 voices) - lang_code='b'
        "bm_george": "b", "bm_fable": "b", "bm_lewis": "b", "bm_daniel": "b",
        # Spanish (3 voices) - lang_code='e'
        "ef_dora": "e", "em_alex": "e", "em_santa": "e",
        # French (1 voice) - lang_code='f'
        "ff_siwis": "f",
        # Hindi (4 voices) - lang_code='h'
        "hf_alpha": "h", "hf_beta": "h", "hm_omega": "h", "hm_psi": "h",
        # Italian (2 voices) - lang_code='i'
        "if_sara": "i", "im_nicola": "i",
        # Brazilian Portuguese (3 voices) - lang_code='p'
        "pf_dora": "p", "pm_alex": "p", "pm_santa": "p",
        # Mandarin Chinese (8 voices) - lang_code='z'
        "zf_xiaobei": "z", "zf_xiaoni": "z", "zf_xiaoxiao": "z", "zf_xiaoyi": "z",
        "zm_yunjian": "z", "zm_yunxi": "z", "zm_yunxia": "z", "zm_yunyang": "z",
    }
    
    try:
        # Download the main model
        print("Loading Kokoro model (this downloads from HuggingFace)...")
        model = KModel(repo_id='hexgrad/Kokoro-82M')
        print("  [OK] Kokoro model downloaded")
        
        # Create pipelines for each language code
        pipelines = {}
        for lang_code in ['a', 'b', 'e', 'f', 'h', 'i', 'p', 'z']:
            pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False)
        
        # Pre-load ALL 40 voice packs
        print(f"\nDownloading all {len(ALL_KOKORO_VOICES)} voice packs...")
        successful = 0
        failed = 0
        
        for voice_id, lang_code in ALL_KOKORO_VOICES.items():
            try:
                pipeline = pipelines[lang_code]
                voice_pack = pipeline.load_voice(voice_id)
                print(f"  [OK] {voice_id}")
                successful += 1
            except Exception as e:
                print(f"  [ERROR] {voice_id}: {e}")
                failed += 1
        
        del model
        print(f"\nKokoro TTS: {successful} voices downloaded, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print(f"  [ERROR] Failed to download Kokoro model: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 70)
    print("WakeBuilder - TTS Model Pre-download Script")
    print("=" * 70)
    print("\nThis script pre-downloads TTS models for Docker builds.")
    print("Piper TTS voices are downloaded separately via download_voices.py\n")
    
    coqui_ok = download_coqui_models()
    kokoro_ok = download_kokoro_model()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Coqui TTS:  {'OK' if coqui_ok else 'PARTIAL/FAILED'}")
    print(f"  Kokoro TTS: {'OK' if kokoro_ok else 'PARTIAL/FAILED'}")
    
    # Return 0 even if some downloads fail (models will download at runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
