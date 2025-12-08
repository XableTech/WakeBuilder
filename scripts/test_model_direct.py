"""Test model directly with TTS and compare to real audio characteristics."""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.preprocessing import AudioPreprocessor
from wakebuilder.tts import TTSGenerator
from wakebuilder.models.classifier import create_model

def main():
    # Load model
    models_dir = Path(__file__).parent.parent / "models" / "custom"
    models = list(models_dir.glob("*/model.pt"))
    if not models:
        print("No models found!")
        return
    
    model_path = models[-1]
    wake_word = model_path.parent.name
    print(f"Model: {wake_word}")
    print(f"Path: {model_path}")
    
    data = torch.load(model_path, map_location="cpu")
    model = create_model(data.get("model_type", "bc_resnet"), num_classes=2)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    
    # Check model mode
    print(f"Model training mode: {model.training}")
    
    # Check BatchNorm stats
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            print(f"{name}: running_mean={module.running_mean.mean():.4f}, running_var={module.running_var.mean():.4f}")
            break
    
    # Generate TTS of wake word
    tts = TTSGenerator(target_sample_rate=16000)
    preprocessor = AudioPreprocessor()
    
    print(f"\n--- Testing TTS '{wake_word}' ---")
    result = tts.synthesize(wake_word)
    audio, sr = result if isinstance(result, tuple) else (result, 16000)
    
    # Process
    spec = preprocessor.process_audio(audio, sr)
    print(f"Spec: shape={spec.shape}, min={spec.min():.2f}, max={spec.max():.2f}, mean={spec.mean():.2f}")
    
    # Check tensor shape - this is what the model receives
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
    print(f"Tensor shape: {spec_tensor.shape}")  # Should be (1, 96, 80) or (1, 1, 96, 80)
    
    # Inference
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = F.softmax(outputs, dim=1)
        print(f"Outputs: {outputs.numpy()}")
        print(f"Confidence: {probs[0, 1].item():.4f}")
    
    # Skip different voice test - API doesn't support it directly
    
    # Test with non-wake-word
    print(f"\n--- Testing non-wake-word 'hello' ---")
    result = tts.synthesize("hello")
    audio, sr = result if isinstance(result, tuple) else (result, 16000)
    spec = preprocessor.process_audio(audio, sr)
    print(f"Spec: shape={spec.shape}, min={spec.min():.2f}, max={spec.max():.2f}, mean={spec.mean():.2f}")
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = F.softmax(outputs, dim=1)
        print(f"Outputs: {outputs.numpy()}")
        print(f"Confidence: {probs[0, 1].item():.4f}")

def test_with_training_sample():
    """Test using the exact same preprocessing as training."""
    import numpy as np
    from wakebuilder.audio.real_data_loader import MassivePositiveAugmenter
    
    # Load model
    models_dir = Path(__file__).parent.parent / "models" / "custom"
    models = list(models_dir.glob("*/model.pt"))
    if not models:
        print("No models found!")
        return
    
    model_path = models[-1]
    wake_word = model_path.parent.name
    print(f"\n=== Testing with training-style sample ===")
    print(f"Model: {wake_word}")
    
    data = torch.load(model_path, map_location="cpu")
    model = create_model(data.get("model_type", "bc_resnet"), num_classes=2)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    
    # Generate a sample using the same augmenter as training
    augmenter = MassivePositiveAugmenter(target_sample_rate=16000)
    preprocessor = AudioPreprocessor()
    
    # Get one TTS sample
    for audio, metadata in augmenter.generate_samples(
        recordings=[],  # No user recordings
        wake_word=wake_word,
        target_count=1,
        use_tts=True,
        use_noise=False,
    ):
        print(f"Generated sample: {metadata}")
        spec = preprocessor.process_audio(audio, 16000)
        print(f"Spec: shape={spec.shape}, min={spec.min():.2f}, max={spec.max():.2f}, mean={spec.mean():.2f}")
        
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        print(f"Tensor shape: {spec_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(spec_tensor)
            probs = F.softmax(outputs, dim=1)
            print(f"Outputs: {outputs.numpy()}")
            print(f"Confidence: {probs[0, 1].item():.4f}")
        break


if __name__ == "__main__":
    main()
    test_with_training_sample()
