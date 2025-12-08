"""Compare training data spectrograms vs real audio spectrograms."""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wakebuilder.audio.preprocessing import AudioPreprocessor
from wakebuilder.audio.real_data_loader import MassivePositiveAugmenter, RealNegativeDataLoader
from wakebuilder.tts import TTSGenerator
from wakebuilder.models.classifier import create_model
import torch.nn.functional as F

def main():
    preprocessor = AudioPreprocessor()
    
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
    
    print("\n" + "="*60)
    print("TRAINING DATA ANALYSIS")
    print("="*60)
    
    # 1. Raw TTS (no augmentation)
    print("\n--- Raw TTS (no augmentation) ---")
    tts = TTSGenerator(target_sample_rate=16000)
    audio, sr = tts.synthesize(wake_word)
    spec = preprocessor.process_audio(audio, sr)
    print(f"Spec stats: min={spec.min():.2f}, max={spec.max():.2f}, mean={spec.mean():.2f}, std={spec.std():.2f}")
    
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
    with torch.no_grad():
        outputs = model(spec_tensor)
        conf = F.softmax(outputs, dim=1)[0, 1].item()
    print(f"Model output: {outputs.numpy()}, Confidence: {conf:.4f}")
    
    # 2. Augmented TTS (like training)
    print("\n--- Augmented TTS (training-style) ---")
    augmenter = MassivePositiveAugmenter(target_sample_rate=16000)
    count = 0
    for audio, metadata in augmenter.generate_samples(
        recordings=[],
        wake_word=wake_word,
        target_count=5,
        use_tts=True,
        use_noise=False,
    ):
        spec = preprocessor.process_audio(audio, 16000)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        with torch.no_grad():
            outputs = model(spec_tensor)
            conf = F.softmax(outputs, dim=1)[0, 1].item()
        print(f"  [{metadata.get('quality_aug', 'unknown')}] mean={spec.mean():.2f}, conf={conf:.4f}")
        count += 1
        if count >= 5:
            break
    
    # 3. Training negatives
    print("\n--- Training Negatives ---")
    neg_loader = RealNegativeDataLoader(target_sample_rate=16000)
    count = 0
    for audio, metadata in neg_loader.load_from_cache(max_samples=5, shuffle=True):
        spec = preprocessor.process_audio(audio, 16000)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        with torch.no_grad():
            outputs = model(spec_tensor)
            conf = F.softmax(outputs, dim=1)[0, 1].item()
        print(f"  mean={spec.mean():.2f}, conf={conf:.4f}")
        count += 1
        if count >= 5:
            break
    
    print("\n" + "="*60)
    print("KEY OBSERVATION")
    print("="*60)
    print("""
The model outputs:
- Positive logit > Negative logit → Class 0 (NOT wake word)
- Negative logit > Positive logit → Class 1 (wake word)

If training positives get high confidence but real audio gets low confidence,
the model learned TTS-specific features, not the wake word pattern.
    """)
    
    # Check what spectrogram features differ
    print("\n" + "="*60)
    print("SPECTROGRAM FEATURE COMPARISON")
    print("="*60)
    
    # Collect stats from multiple samples
    tts_means = []
    tts_stds = []
    neg_means = []
    neg_stds = []
    
    # TTS samples
    for audio, _ in augmenter.generate_samples([], wake_word, 20, True, False):
        spec = preprocessor.process_audio(audio, 16000)
        tts_means.append(spec.mean())
        tts_stds.append(spec.std())
    
    # Negative samples
    for audio, _ in neg_loader.load_from_cache(max_samples=20, shuffle=True):
        spec = preprocessor.process_audio(audio, 16000)
        neg_means.append(spec.mean())
        neg_stds.append(spec.std())
    
    print(f"\nTTS Positives (n={len(tts_means)}):")
    print(f"  Mean: {np.mean(tts_means):.2f} ± {np.std(tts_means):.2f}")
    print(f"  Std:  {np.mean(tts_stds):.2f} ± {np.std(tts_stds):.2f}")
    
    print(f"\nReal Negatives (n={len(neg_means)}):")
    print(f"  Mean: {np.mean(neg_means):.2f} ± {np.std(neg_means):.2f}")
    print(f"  Std:  {np.mean(neg_stds):.2f} ± {np.std(neg_stds):.2f}")
    
    print("\nIf these distributions are very different, the model may be")
    print("using these statistics as shortcuts instead of learning the wake word.")

if __name__ == "__main__":
    main()
