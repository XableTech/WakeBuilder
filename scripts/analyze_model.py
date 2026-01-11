#!/usr/bin/env python3
"""
Analyze what the model has learned by comparing positive vs negative samples.

This script tests trained AST wake word models to understand:
1. Model confidence on negative samples
2. Effect of spectrogram normalization shifts
3. TTS-generated wake word confidence
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.real_data_loader import RealNegativeDataLoader
from wakebuilder.models.classifier import (
    ASTWakeWordModel,
    AST_MODEL_CHECKPOINT,
)


def load_model_from_checkpoint(model_path: Path) -> tuple:
    """
    Load a trained AST wake word model from checkpoint.

    The model saves only the classifier weights (classifier_state_dict)
    because the base AST model is frozen and always loaded fresh.

    Returns:
        Tuple of (model, metadata_dict)
    """
    data = torch.load(model_path, map_location="cpu", weights_only=False)

    # Get configuration from saved data
    classifier_dims = data.get("classifier_hidden_dims", [256, 128])
    classifier_dropout = data.get("classifier_dropout", 0.3)
    use_attention = data.get("use_attention", False)
    use_se_block = data.get("use_se_block", False)
    use_tcn = data.get("use_tcn", False)

    print("Loading model with config:")
    print(f"  Classifier dims: {classifier_dims}")
    print(f"  Use attention: {use_attention}")
    print(f"  Use SE block: {use_se_block}")
    print(f"  Use TCN: {use_tcn}")

    # Create model with the same configuration
    model = ASTWakeWordModel(
        freeze_base=True,
        classifier_hidden_dims=classifier_dims,
        classifier_dropout=classifier_dropout,
        use_attention=use_attention,
        use_se_block=use_se_block,
        use_tcn=use_tcn,
    )

    # Load only the classifier weights (base model is frozen and loaded fresh)
    if "classifier_state_dict" in data:
        model.classifier.load_state_dict(data["classifier_state_dict"])
    elif "model_state_dict" in data:
        # Backward compatibility with old format
        model.load_state_dict(data["model_state_dict"], strict=False)
    else:
        raise KeyError(
            f"No classifier weights found. Available keys: {list(data.keys())}"
        )

    model.eval()

    return model, data


def main():
    # Load model
    models_dir = Path(__file__).parent.parent / "models" / "custom"
    models = list(models_dir.glob("*/model.pt"))
    if not models:
        print("No models found in models/custom/")
        print("Train a model first using the web interface or API.")
        return

    model_path = models[-1]
    wake_word = model_path.parent.name
    print(f"Model: {wake_word}")
    print(f"Path: {model_path}")
    print()

    try:
        model, metadata = load_model_from_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\nThreshold: {metadata.get('threshold', 'N/A')}")

    # Import feature extractor for AST
    from transformers import AutoFeatureExtractor

    print("\nLoading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_CHECKPOINT)

    # Load some negative samples
    print("\n=== Negative samples (should output low confidence) ===")
    neg_loader = RealNegativeDataLoader(target_sample_rate=16000)
    neg_stats = []

    for audio, audio_metadata in neg_loader.load_from_cache(max_samples=10):
        # Process through AST feature extractor
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(inputs["input_values"])
            probs = F.softmax(outputs, dim=1)
            conf = probs[0, 1].item()

        neg_stats.append(conf)
        print(f"  conf={conf:.4f}")

    if neg_stats:
        print(
            f"\nNegative stats: mean={np.mean(neg_stats):.4f}, std={np.std(neg_stats):.4f}"
        )
        print(f"  Min: {min(neg_stats):.4f}, Max: {max(neg_stats):.4f}")
    else:
        print("\nNo negative samples found in cache.")
        print("Run 'python scripts/build_negative_cache.py' to build the cache.")

    # Test with TTS-generated wake word
    print(f"\n=== Testing TTS-generated '{wake_word}' ===")

    try:
        from wakebuilder.tts import TTSGenerator

        tts = TTSGenerator(target_sample_rate=16000)
        result = tts.synthesize(wake_word)
        audio, sr = result if isinstance(result, tuple) else (result, 16000)

        # Process through AST feature extractor
        inputs = feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(inputs["input_values"])
            probs = F.softmax(outputs, dim=1)
            conf = probs[0, 1].item()

        print(f"Wake word '{wake_word}' confidence: {conf:.4f}")

        threshold = metadata.get("threshold", 0.5)
        if conf > threshold:
            print(
                f"  [OK] Model correctly identifies wake word (threshold: {threshold:.2f})"
            )
        else:
            print(f"  [WARNING] Model confidence is below threshold ({threshold:.2f})")

    except ImportError as e:
        print(f"TTS not available: {e}")
    except Exception as e:
        print(f"Error testing TTS: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n=== Summary ===")
    if neg_stats:
        threshold = metadata.get("threshold", 0.5)
        high_conf_negs = sum(1 for c in neg_stats if c > threshold)
        print(f"False positives in test set: {high_conf_negs}/{len(neg_stats)}")

        if high_conf_negs == 0:
            print("  [OK] No false positives detected")
        else:
            print("  [WARNING] Some false positives detected")


if __name__ == "__main__":
    main()
