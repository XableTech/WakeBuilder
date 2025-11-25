# Phase 2: Training Pipeline Documentation

## Overview

Phase 2 implements a complete wake word model training pipeline, from data augmentation to model export. The pipeline is designed to train models competitive with commercial solutions like Picovoice Porcupine.

## Architecture

### Model Options

| Model | Use Case | Latency | Size | Parameters | Expected Accuracy |
|-------|----------|---------|------|------------|-------------------|
| **TC-ResNet** | Production (speed) | ~0.8ms | 261KB | 67K | 96-97% |
| **BC-ResNet** | Accuracy-critical | ~6ms | 500KB | 128K | 97-98% |

Both models feature:
- Proper residual connections for stable training
- Configurable width multipliers (`scale` / `width_mult`)
- Squeeze-and-Excitation attention (BC-ResNet)
- Embedding extraction for transfer learning

### Pipeline Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Input    │────▶│  Data Augmenter  │────▶│  Spectrogram    │
│  (hi-alexa.wav) │     │  + TTS Generator │     │  (80 mel bins)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Exported Model │◀────│    Trainer       │◀────│  DataLoader     │
│  + Metadata     │     │  (Early Stop)    │     │  (Balanced)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Usage

### Quick Start

```bash
# Quick test (5 epochs, ~50 samples) - for pipeline verification
uv run python scripts/train_wake_word.py \
    --wake-word "hi alexa" \
    --audio hi-alexa.wav \
    --model tc_resnet \
    --quick

# Full training for production
uv run python scripts/train_wake_word.py \
    --wake-word "hi alexa" \
    --audio hi-alexa.wav \
    --model tc_resnet \
    --epochs 50
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wake-word` | "hi alexa" | The wake word phrase |
| `--audio` | "hi-alexa.wav" | Path to recorded wake word |
| `--model` | "tc_resnet" | Model type: `tc_resnet` or `bc_resnet` |
| `--epochs` | 30 | Number of training epochs |
| `--output-dir` | "models/" | Output directory for trained model |
| `--quick` | False | Quick mode for testing (fewer samples/epochs) |

## Output Structure

```
models/
└── hi_alexa_tc_resnet/          # Wake word + model type
    ├── model.pt                  # PyTorch checkpoint (weights + config)
    ├── metadata.json             # Wake word, threshold, metrics
    └── training_history.json     # Per-epoch training metrics

data/
└── augmented/
    └── hi_alexa/                 # Augmented samples for inspection
        ├── positive/             # Positive samples (wake word)
        │   ├── positive_000.wav
        │   ├── positive_001.wav
        │   └── ...
        └── negative/             # Negative samples (non-wake-word)
            ├── negative_000.wav
            └── ...
```

### Metadata Format

```json
{
  "wake_word": "hi alexa_tc_resnet",
  "threshold": 0.45,
  "model_type": "tc_resnet",
  "parameters": 66938,
  "training_config": {
    "batch_size": 32,
    "num_epochs": 30,
    "learning_rate": 0.001
  },
  "metrics": {
    "best_val_loss": 0.15,
    "val_accuracy": 0.96,
    "val_f1": 0.92
  },
  "version": "1.0.0",
  "framework": "pytorch",
  "original_wake_word": "hi alexa"
}
```

## Achieving 95%+ Accuracy

To reach production-level accuracy (95%+), ensure:

### 1. Sufficient Training Data

| Mode | Positive Samples | Negative Samples | Expected Accuracy |
|------|------------------|------------------|-------------------|
| Quick | 50 | 100 | 80-85% |
| Default | 1000 | 1000 | 90-95% |
| Production | 2000+ | 2000+ | 95-98% |

### 2. Data Quality

- **Multiple recordings**: Record the wake word 5-10 times with variations
- **Different speakers**: Use TTS with multiple voices (automatic)
- **Noise conditions**: Pipeline adds noise augmentation automatically
- **Balanced classes**: Pipeline uses weighted sampling for imbalanced data

### 3. Training Configuration

```python
# Recommended for 95%+ accuracy
TrainingConfig(
    model_type="bc_resnet",    # Higher accuracy model
    num_epochs=50,             # More epochs
    batch_size=32,             # Standard batch size
    learning_rate=1e-3,        # Default LR with OneCycleLR
    patience=15,               # Early stopping patience
    label_smoothing=0.1,       # Regularization
    mixup_alpha=0.2,           # Data augmentation
)
```

### 4. Why Accuracy May Be Low

| Issue | Cause | Solution |
|-------|-------|----------|
| <85% accuracy | Too few samples | Use `--epochs 50` without `--quick` |
| High FAR | Insufficient negatives | Add more negative recordings |
| High FRR | Insufficient positives | Record more wake word samples |
| Overfitting | Small dataset | Increase augmentation, add dropout |
| Underfitting | Too few epochs | Increase `--epochs` to 50-100 |

## Threshold Calibration

The pipeline automatically calibrates the detection threshold by:

1. Computing predictions on validation set
2. Calculating FAR/FRR at 100 threshold values
3. Selecting threshold that minimizes `2*FAR + FRR` (prioritizes low false activations)

### Threshold Report Example

```
Optimal Threshold: 0.450
  FAR (False Acceptance Rate): 2.5%
  FRR (False Rejection Rate): 5.0%
  Accuracy: 96.2%
  F1 Score: 0.94
```

## Loading Trained Models

```python
import torch
import json
from pathlib import Path
from wakebuilder.models import create_model
from wakebuilder.audio import AudioPreprocessor

# Load model
model_dir = Path("models/hi_alexa_tc_resnet")
with open(model_dir / "metadata.json") as f:
    metadata = json.load(f)

checkpoint = torch.load(model_dir / "model.pt", weights_only=True)
model = create_model(
    model_type=checkpoint["model_type"],
    n_mels=checkpoint["n_mels"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Run inference
preprocessor = AudioPreprocessor(n_mels=80)
spectrogram = preprocessor.process_audio(audio, sample_rate=16000)
spec_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()

with torch.no_grad():
    logits = model(spec_tensor)
    prob = torch.softmax(logits, dim=1)[0, 1].item()

detected = prob >= metadata["threshold"]
print(f"Wake word detected: {detected} (prob={prob:.3f})")
```

## Files Reference

| File | Description |
|------|-------------|
| `scripts/train_wake_word.py` | End-to-end training script |
| `scripts/benchmark_models.py` | Model speed/size benchmarking |
| `scripts/verify_model.py` | Verify exported model loads correctly |
| `src/wakebuilder/models/classifier.py` | BC-ResNet and TC-ResNet architectures |
| `src/wakebuilder/models/trainer.py` | Training loop, data prep, threshold calibration |
| `tests/test_training.py` | Unit tests for training pipeline |

## Performance Benchmarks

Tested on CPU (Intel i7):

| Model | Inference | Throughput | Size |
|-------|-----------|------------|------|
| TC-ResNet | 0.8ms | 1,272/sec | 261KB |
| BC-ResNet | 6ms | 160/sec | 500KB |
| BC-ResNet (small, scale=0.5) | 5ms | 184/sec | 137KB |

All models are **200-500x faster than real-time**, suitable for continuous wake word detection.

## Next Steps (Phase 3)

- Real-time inference engine
- Streaming audio processing
- ONNX export for deployment
- Mobile optimization
