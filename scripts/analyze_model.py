"""Analyze what the model has learned by comparing positive vs negative samples."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.preprocessing import AudioPreprocessor
from wakebuilder.audio.real_data_loader import RealNegativeDataLoader
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
    
    data = torch.load(model_path, map_location="cpu")
    model = create_model(data.get("model_type", "bc_resnet"), num_classes=2)
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    
    preprocessor = AudioPreprocessor()
    
    # Load some negative samples
    print("\n=== Negative samples (should output low confidence) ===")
    neg_loader = RealNegativeDataLoader(target_sample_rate=16000)
    neg_specs = []
    for audio, metadata in neg_loader.load_from_cache(max_samples=10):
        spec = preprocessor.process_audio(audio, 16000)
        neg_specs.append(spec)
        
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        with torch.no_grad():
            outputs = model(spec_tensor)
            probs = F.softmax(outputs, dim=1)
            conf = probs[0, 1].item()
        print(f"  Spec mean={spec.mean():.2f}, conf={conf:.4f}")
    
    # Compute average negative spectrogram stats
    neg_specs = np.array(neg_specs)
    print(f"\nNegative stats: mean={neg_specs.mean():.2f}, std={neg_specs.std():.2f}")
    print(f"  Per-sample means: min={neg_specs.mean(axis=(1,2)).min():.2f}, max={neg_specs.mean(axis=(1,2)).max():.2f}")
    
    # The key insight: what spectrogram mean does the model expect for positives?
    print("\n=== Testing different spectrogram normalizations ===")
    
    # Create a synthetic "positive-like" spectrogram by shifting the mean
    from wakebuilder.tts import TTSGenerator
    tts = TTSGenerator(target_sample_rate=16000)
    result = tts.synthesize(wake_word)
    audio, sr = result if isinstance(result, tuple) else (result, 16000)
    spec = preprocessor.process_audio(audio, sr)
    
    print(f"Original TTS spec: mean={spec.mean():.2f}")
    
    # Try different mean shifts
    for shift in [-10, -5, 0, 5, 10, 15, 20]:
        shifted = spec + shift
        spec_tensor = torch.from_numpy(shifted).float().unsqueeze(0)
        with torch.no_grad():
            outputs = model(spec_tensor)
            probs = F.softmax(outputs, dim=1)
            conf = probs[0, 1].item()
        print(f"  Shift {shift:+3d} dB -> mean={shifted.mean():.2f}, conf={conf:.4f}")

if __name__ == "__main__":
    main()
