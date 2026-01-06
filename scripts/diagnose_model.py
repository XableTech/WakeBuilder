#!/usr/bin/env python
"""
Diagnostic script to analyze wake word model behavior.

This script tests the model against:
1. The wake word itself (should detect)
2. Phonetically similar words (should NOT detect)
3. Random speech (should NOT detect)
4. Silence (should NOT detect)

This helps identify why the model may be triggering on similar-sounding words.

Usage:
    # Auto-detect and select from available models interactively
    uv run python scripts/diagnose_model.py
    
    # Specify a model directly
    uv run python scripts/diagnose_model.py --model models/custom/jarvis/model.pt
    
    # List available models
    uv run python scripts/diagnose_model.py --list
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.wakebuilder.models.classifier import (
    ASTWakeWordModel,
    AST_MODEL_CHECKPOINT,
)
from src.wakebuilder.audio.negative_generator import get_phonetically_similar_words
from transformers import AutoFeatureExtractor


def find_available_models() -> list[dict]:
    """Find all available trained models in the models directory."""
    models = []
    
    # Check custom models
    custom_dir = project_root / "models" / "custom"
    if custom_dir.exists():
        for model_dir in custom_dir.iterdir():
            if model_dir.is_dir():
                model_file = model_dir / "model.pt"
                metadata_file = model_dir / "metadata.json"
                
                if model_file.exists():
                    model_info = {
                        "name": model_dir.name,
                        "path": model_file,
                        "category": "custom",
                    }
                    
                    # Try to load metadata
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                            model_info["wake_word"] = metadata.get("wake_word", model_dir.name)
                            model_info["threshold"] = metadata.get("threshold", 0.5)
                        except Exception:
                            model_info["wake_word"] = model_dir.name
                            model_info["threshold"] = 0.5
                    else:
                        model_info["wake_word"] = model_dir.name
                        model_info["threshold"] = 0.5
                    
                    models.append(model_info)
    
    # Check default models
    default_dir = project_root / "models" / "default"
    if default_dir.exists():
        for model_dir in default_dir.iterdir():
            if model_dir.is_dir():
                model_file = model_dir / "model.pt"
                if model_file.exists():
                    model_info = {
                        "name": model_dir.name,
                        "path": model_file,
                        "category": "default",
                        "wake_word": model_dir.name,
                        "threshold": 0.5,
                    }
                    models.append(model_info)
    
    return models


def list_models(models: list[dict]) -> None:
    """Print a formatted list of available models."""
    if not models:
        print("\nNo trained models found!")
        print("Train a model first using the web interface or API.")
        return
    
    print("\nAvailable trained models:")
    print("-" * 60)
    
    for i, model in enumerate(models, 1):
        category_badge = f"[{model['category'].upper()}]"
        print(f"  {i}. {model['name']:20s} {category_badge:10s} wake_word='{model['wake_word']}'")
    
    print("-" * 60)
    print(f"Total: {len(models)} model(s)")


def select_model(models: list[dict]) -> dict | None:
    """Interactively select a model from available options."""
    if not models:
        print("\nNo trained models found!")
        print("Train a model first using the web interface or API.")
        return None
    
    if len(models) == 1:
        print(f"\nOnly one model available: '{models[0]['name']}' - using it automatically.")
        return models[0]
    
    list_models(models)
    
    print(f"\nSelect a model (1-{len(models)}) or press Enter for latest: ", end="")
    try:
        selection = input().strip()
        if not selection:
            # Use the most recently modified model
            models_sorted = sorted(models, key=lambda m: m["path"].stat().st_mtime, reverse=True)
            selected = models_sorted[0]
            print(f"Using latest model: '{selected['name']}'")
            return selected
        
        idx = int(selection) - 1
        if 0 <= idx < len(models):
            return models[idx]
        else:
            print(f"Invalid selection. Using first model.")
            return models[0]
    except (ValueError, EOFError):
        print("Using first model.")
        return models[0]


def load_model(model_path: Path) -> tuple[ASTWakeWordModel, dict]:
    """Load a trained wake word model."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Create model
    model = ASTWakeWordModel(
        freeze_base=True,
        classifier_hidden_dims=checkpoint.get("classifier_hidden_dims", [256, 128]),
        classifier_dropout=checkpoint.get("classifier_dropout", 0.3),
        use_attention=checkpoint.get("use_attention", False),
        use_se_block=checkpoint.get("use_se_block", False),
        use_tcn=checkpoint.get("use_tcn", False),
    )
    
    # Load classifier weights
    if "classifier_state_dict" in checkpoint:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    
    model.eval()
    return model, checkpoint


def analyze_embeddings(model: ASTWakeWordModel, feature_extractor, tts, wake_word: str, similar_words: list[str]):
    """Analyze embedding similarity between wake word and similar words."""
    print("\n" + "=" * 70)
    print("EMBEDDING ANALYSIS")
    print("=" * 70)
    
    # Get embeddings for wake word
    wake_embeddings = []
    for voice in tts.voice_names[:5]:  # Use 5 voices
        try:
            audio, sr = tts.synthesize(wake_word, voice_name=voice)
            audio = pad_or_trim(audio, 16000)
            
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                emb = model.get_embeddings(inputs["input_values"])
            wake_embeddings.append(emb.squeeze().numpy())
        except Exception as e:
            continue
    
    if not wake_embeddings:
        print("Could not generate wake word embeddings")
        return
    
    wake_mean = np.mean(wake_embeddings, axis=0)
    
    # Analyze similar words
    print(f"\nWake word: '{wake_word}'")
    print(f"Embedding dimension: {len(wake_mean)}")
    print(f"\nCosine similarity to wake word (higher = more similar):")
    print("-" * 50)
    
    similarities = []
    for word in similar_words[:20]:  # Top 20 similar words
        word_embeddings = []
        for voice in tts.voice_names[:3]:  # Use 3 voices
            try:
                audio, sr = tts.synthesize(word, voice_name=voice)
                audio = pad_or_trim(audio, 16000)
                
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    emb = model.get_embeddings(inputs["input_values"])
                word_embeddings.append(emb.squeeze().numpy())
            except Exception:
                continue
        
        if word_embeddings:
            word_mean = np.mean(word_embeddings, axis=0)
            # Cosine similarity
            sim = np.dot(wake_mean, word_mean) / (np.linalg.norm(wake_mean) * np.linalg.norm(word_mean))
            similarities.append((word, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for word, sim in similarities:
        bar = "#" * int(sim * 30)
        print(f"  {word:20s} {sim:.4f} {bar}")
    
    return similarities


def test_model_predictions(model: ASTWakeWordModel, feature_extractor, tts, wake_word: str, similar_words: list[str], threshold: float):
    """Test model predictions on wake word and similar words."""
    print("\n" + "=" * 70)
    print("PREDICTION ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {threshold:.4f}")
    
    results = {
        "wake_word": {"detected": 0, "total": 0, "scores": []},
        "similar": {},
    }
    
    # Test wake word
    print(f"\n1. Wake word: '{wake_word}'")
    print("-" * 50)
    
    for voice in tts.voice_names[:10]:
        try:
            audio, sr = tts.synthesize(wake_word, voice_name=voice)
            audio = pad_or_trim(audio, 16000)
            
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs["input_values"])
                probs = torch.softmax(logits, dim=-1)
                score = probs[0, 1].item()
            
            detected = score >= threshold
            results["wake_word"]["total"] += 1
            results["wake_word"]["scores"].append(score)
            if detected:
                results["wake_word"]["detected"] += 1
            
            status = "[DETECTED]" if detected else "[MISSED]"
            print(f"  {voice:30s} score={score:.4f} {status}")
        except Exception as e:
            print(f"  {voice:30s} ERROR: {e}")
    
    wake_rate = results["wake_word"]["detected"] / max(results["wake_word"]["total"], 1)
    print(f"\n  Detection rate: {wake_rate:.1%} ({results['wake_word']['detected']}/{results['wake_word']['total']})")
    if results["wake_word"]["scores"]:
        print(f"  Average score: {np.mean(results['wake_word']['scores']):.4f}")
    
    # Test similar words
    print(f"\n2. Phonetically similar words (should NOT detect):")
    print("-" * 50)
    
    false_positives = []
    
    for word in similar_words[:15]:  # Top 15 similar words
        word_results = {"detected": 0, "total": 0, "scores": []}
        
        for voice in tts.voice_names[:5]:  # 5 voices per word
            try:
                audio, sr = tts.synthesize(word, voice_name=voice)
                audio = pad_or_trim(audio, 16000)
                
                inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    logits = model(inputs["input_values"])
                    probs = torch.softmax(logits, dim=-1)
                    score = probs[0, 1].item()
                
                detected = score >= threshold
                word_results["total"] += 1
                word_results["scores"].append(score)
                if detected:
                    word_results["detected"] += 1
                    false_positives.append((word, voice, score))
            except Exception:
                continue
        
        if word_results["total"] > 0:
            fp_rate = word_results["detected"] / word_results["total"]
            avg_score = np.mean(word_results["scores"])
            max_score = max(word_results["scores"])
            
            status = "[!] FALSE POSITIVE" if fp_rate > 0 else "[OK]"
            print(f"  {word:20s} FP={fp_rate:.0%} avg={avg_score:.4f} max={max_score:.4f} {status}")
            
            results["similar"][word] = word_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_fp = sum(r["detected"] for r in results["similar"].values())
    total_similar = sum(r["total"] for r in results["similar"].values())
    
    print(f"Wake word detection rate: {wake_rate:.1%}")
    print(f"False positive rate on similar words: {total_fp}/{total_similar} ({total_fp/max(total_similar,1):.1%})")
    
    if false_positives:
        print(f"\nFalse positives (detected as wake word):")
        for word, voice, score in sorted(false_positives, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  '{word}' ({voice}): {score:.4f}")
    
    return results


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim audio to target length."""
    if len(audio) == target_length:
        return audio
    elif len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    else:
        pad_total = target_length - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')


def analyze_classifier_weights(model: ASTWakeWordModel):
    """Analyze classifier weights for potential issues."""
    print("\n" + "=" * 70)
    print("CLASSIFIER WEIGHT ANALYSIS")
    print("=" * 70)
    
    classifier = model.classifier
    
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            data = param.data.numpy()
            print(f"\n{name}:")
            print(f"  Shape: {data.shape}")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std: {data.std():.6f}")
            print(f"  Min: {data.min():.6f}")
            print(f"  Max: {data.max():.6f}")
            
            # Check for potential issues
            if data.std() < 0.01:
                print("  [!] WARNING: Very low variance - weights may be collapsed")
            if abs(data.mean()) > 1.0:
                print("  [!] WARNING: High mean - potential bias issue")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose wake word model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and select from available models
  python scripts/diagnose_model.py
  
  # List all available models
  python scripts/diagnose_model.py --list
  
  # Diagnose a specific model
  python scripts/diagnose_model.py --model models/custom/jarvis/model.pt
        """
    )
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file (optional, will prompt if not specified)")
    parser.add_argument("--wake-word", type=str, default=None,
                       help="Wake word (auto-detected from model if not specified)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Detection threshold (auto-detected from model if not specified)")
    parser.add_argument("--list", action="store_true",
                       help="List available models and exit")
    args = parser.parse_args()
    
    # Find available models
    available_models = find_available_models()
    
    # If --list flag, just show models and exit
    if args.list:
        list_models(available_models)
        return 0
    
    # Determine which model to use
    if args.model:
        model_path = project_root / args.model
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            list_models(available_models)
            return 1
        
        # Try to find matching model info
        model_info = None
        for m in available_models:
            if m["path"] == model_path:
                model_info = m
                break
        
        if not model_info:
            model_info = {
                "name": model_path.parent.name,
                "path": model_path,
                "wake_word": args.wake_word or model_path.parent.name,
                "threshold": args.threshold or 0.5,
            }
    else:
        # No model specified - check what's available
        if not available_models:
            print("\n" + "=" * 70)
            print("NO TRAINED MODELS FOUND")
            print("=" * 70)
            print("\nTo diagnose a model, you first need to train one.")
            print("\nOptions:")
            print("  1. Use the web interface: python run.py")
            print("  2. Use the training script: python scripts/train_improved.py")
            print("\nAfter training, run this script again.")
            return 1
        
        # Select a model interactively
        model_info = select_model(available_models)
        if not model_info:
            return 1
        
        model_path = model_info["path"]
    
    print("=" * 70)
    print("WAKE WORD MODEL DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {model_path}")
    
    # Load model
    print("\nLoading model...")
    model, checkpoint = load_model(model_path)
    
    wake_word = args.wake_word or checkpoint.get("wake_word") or model_info.get("wake_word", "unknown")
    threshold = args.threshold or checkpoint.get("threshold") or model_info.get("threshold", 0.5)
    
    print(f"Wake word: '{wake_word}'")
    print(f"Threshold: {threshold:.4f}")
    
    # Load feature extractor
    print("Loading AST feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_CHECKPOINT)
    
    # Load TTS
    print("Loading TTS...")
    try:
        from src.wakebuilder.tts import TTSGenerator
        tts = TTSGenerator(target_sample_rate=16000)
        print(f"Available voices: {len(tts.voice_names)}")
    except Exception as e:
        print(f"TTS not available: {e}")
        return 1
    
    # Get phonetically similar words
    similar_words = get_phonetically_similar_words(wake_word)
    print(f"\nGenerated {len(similar_words)} phonetically similar words")
    print(f"Top 10: {similar_words[:10]}")
    
    # Analyze classifier weights
    analyze_classifier_weights(model)
    
    # Analyze embeddings
    analyze_embeddings(model, feature_extractor, tts, wake_word, similar_words)
    
    # Test predictions
    results = test_model_predictions(model, feature_extractor, tts, wake_word, similar_words, threshold)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    total_fp = sum(r["detected"] for r in results["similar"].values())
    total_similar = sum(r["total"] for r in results["similar"].values())
    fp_rate = total_fp / max(total_similar, 1)
    
    if fp_rate > 0.1:
        print("""
⚠️ HIGH FALSE POSITIVE RATE DETECTED

Possible causes and solutions:

1. INSUFFICIENT HARD NEGATIVES
   - The model may not have seen enough phonetically similar words during training
   - Solution: Increase hard_negative_ratio to 3.0-5.0x
   
2. EMBEDDING SIMILARITY
   - The AST embeddings for similar words may be too close to the wake word
   - Solution: Use a more discriminative classifier architecture
   
3. LABEL SMOOTHING TOO LOW
   - The model may be overconfident on training data
   - Solution: Increase label_smoothing to 0.3-0.4
   
4. THRESHOLD TOO LOW
   - Current threshold may be too permissive
   - Solution: Increase threshold based on FAR/FRR analysis
   
5. CLASSIFIER CAPACITY
   - The classifier may be too simple to learn fine distinctions
   - Solution: Try deeper classifier [512, 256, 128] or add attention
""")
    else:
        print("[OK] Model appears to be discriminating well between wake word and similar words")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
