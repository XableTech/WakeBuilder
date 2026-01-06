#!/usr/bin/env python3
"""
Compare training data vs real audio for AST wake word models.

This script analyzes the differences between:
1. Raw TTS audio
2. Augmented TTS (training-style)
3. Real negative audio

This helps identify if models are learning TTS-specific features
instead of actual wake word patterns.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.preprocessing import AudioPreprocessor
from wakebuilder.audio.real_data_loader import MassivePositiveAugmenter, RealNegativeDataLoader
from wakebuilder.tts import TTSGenerator
from wakebuilder.models.classifier import ASTWakeWordModel, AST_MODEL_CHECKPOINT
from transformers import AutoFeatureExtractor


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
        raise KeyError(f"No classifier weights found. Available keys: {list(data.keys())}")
    
    model.eval()
    
    return model, data


def get_model_confidence(model, feature_extractor, audio: np.ndarray, sr: int = 16000) -> float:
    """Get model confidence for audio input."""
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model(inputs['input_values'])
        probs = F.softmax(outputs, dim=1)
        return probs[0, 1].item()


def main():
    preprocessor = AudioPreprocessor()
    
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
    
    try:
        model, metadata = load_model_from_checkpoint(model_path)
        feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_CHECKPOINT)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Raw TTS (no augmentation)
    print("\n--- Raw TTS (no augmentation) ---")
    try:
        tts = TTSGenerator(target_sample_rate=16000)
        result = tts.synthesize(wake_word)
        audio, sr = result if isinstance(result, tuple) else (result, 16000)
        spec = preprocessor.process_audio(audio, sr)
        print(f"Spec stats: min={spec.min():.2f}, max={spec.max():.2f}, mean={spec.mean():.2f}, std={spec.std():.2f}")
        
        conf = get_model_confidence(model, feature_extractor, audio, sr)
        print(f"Confidence: {conf:.4f}")
    except Exception as e:
        print(f"Error with TTS: {e}")
    
    # 2. Augmented TTS (like training)
    print("\n--- Augmented TTS (training-style) ---")
    try:
        augmenter = MassivePositiveAugmenter(target_sample_rate=16000)
        count = 0
        tts_confidences = []
        for audio, audio_metadata in augmenter.generate_samples(
            recordings=[],
            wake_word=wake_word,
            target_count=5,
            use_tts=True,
            use_noise=False,
        ):
            spec = preprocessor.process_audio(audio, 16000)
            conf = get_model_confidence(model, feature_extractor, audio, 16000)
            tts_confidences.append(conf)
            print(f"  [{audio_metadata.get('quality_aug', 'unknown')}] spec_mean={spec.mean():.2f}, conf={conf:.4f}")
            count += 1
            if count >= 5:
                break
    except Exception as e:
        print(f"Error with augmented TTS: {e}")
        tts_confidences = []
    
    # 3. Training negatives
    print("\n--- Training Negatives ---")
    neg_loader = RealNegativeDataLoader(target_sample_rate=16000)
    neg_confidences = []
    count = 0
    for audio, audio_metadata in neg_loader.load_from_cache(max_samples=5, shuffle=True):
        spec = preprocessor.process_audio(audio, 16000)
        conf = get_model_confidence(model, feature_extractor, audio, 16000)
        neg_confidences.append(conf)
        print(f"  spec_mean={spec.mean():.2f}, conf={conf:.4f}")
        count += 1
        if count >= 5:
            break
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATION")
    print("=" * 60)
    print("""
The model outputs confidence between 0 and 1:
- High confidence (>threshold) = Wake word detected
- Low confidence (<threshold) = Not wake word

If TTS samples get high confidence but model doesn't work on real audio,
the model may have learned TTS-specific features instead of the wake word.
    """)
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    threshold = metadata.get('threshold', 0.5)
    print(f"\nThreshold: {threshold:.4f}")
    
    if tts_confidences:
        print(f"\nTTS Positives (n={len(tts_confidences)}):")
        print(f"  Mean confidence: {np.mean(tts_confidences):.4f}")
        print(f"  Above threshold: {sum(1 for c in tts_confidences if c > threshold)}/{len(tts_confidences)}")
    
    if neg_confidences:
        print(f"\nReal Negatives (n={len(neg_confidences)}):")
        print(f"  Mean confidence: {np.mean(neg_confidences):.4f}")
        print(f"  False positives: {sum(1 for c in neg_confidences if c > threshold)}/{len(neg_confidences)}")
    
    # Quality assessment
    if tts_confidences and neg_confidences:
        tts_mean = np.mean(tts_confidences)
        neg_mean = np.mean(neg_confidences)
        separation = tts_mean - neg_mean
        
        print(f"\nClass separation: {separation:.4f}")
        if separation > 0.5:
            print("  [GOOD] Strong separation between positives and negatives")
        elif separation > 0.2:
            print("  [OK] Moderate separation, consider more training")
        else:
            print("  [WARNING] Poor separation, model may need retraining")


if __name__ == "__main__":
    main()
