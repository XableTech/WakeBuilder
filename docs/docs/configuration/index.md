# Configuration

Customize WakeBuilder's behavior through various configuration options.

---

## Overview

WakeBuilder provides extensive configuration options for:

- Audio processing parameters
- Training hyperparameters
- Data augmentation settings
- Model architecture options
- API configuration

---

## In This Section

<div class="grid cards" markdown>

- :material-waveform:{ .lg .middle } **Audio Settings**

    ---

    Sample rate, mel spectrogram parameters, and audio processing.

    [:octicons-arrow-right-24: Audio Settings](audio-settings.md)

- :material-tune:{ .lg .middle } **Training Hyperparameters**

    ---

    Learning rate, batch size, epochs, and optimization settings.

    [:octicons-arrow-right-24: Hyperparameters](hyperparameters.md)

- :material-shuffle-variant:{ .lg .middle } **Data Augmentation**

    ---

    TTS settings, speed variations, noise levels, and sample ratios.

    [:octicons-arrow-right-24: Augmentation](augmentation.md)

- :material-cog:{ .lg .middle } **Model Options**

    ---

    Classifier architecture, attention, SE blocks, and TCN.

    [:octicons-arrow-right-24: Model Options](model-options.md)

</div>

---

## Configuration File

All settings are centralized in `src/wakebuilder/config.py`:

```python
# Audio processing parameters
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 80,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "fmin": 0,
    "fmax": 8000,
    "duration": 1.0,
}

# Training parameters
TRAINING_CONFIG = {
    "embedding_dim": 768,
    "hidden_dims": [256, 128],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 50,
    "early_stopping_patience": 5,
    "validation_split": 0.2,
}
```

---

## Web Interface Configuration

Most settings can be configured through the **Advanced Options** panel in the training wizard:

### Data Generation

| Setting | Default | Description |
|---------|---------|-------------|
| Target Positive Samples | 5,000 | Total positive training samples |
| Max Negative Chunks | Auto | Cached negative data samples |
| Negative Ratio | 2.0x | Real negatives per positive |
| Hard Negative Ratio | 4.0x | Similar words per positive |

### Training Parameters

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Batch Size | 32 | 8-256 | Samples per gradient update |
| Max Epochs | 100 | 10-500 | Maximum training iterations |
| Early Stopping | 8 | 0-15 | Patience epochs |
| Learning Rate | 0.0001 | 0.000001-0.1 | Optimizer step size |
| Dropout | 0.5 | 0-0.7 | Regularization strength |
| Label Smoothing | 0.1 | 0-0.3 | Confidence calibration |
| Mixup Alpha | 0.5 | 0-1.0 | Data mixing intensity |

### Model Enhancements

| Option | Default | Description |
|--------|---------|-------------|
| Focal Loss | ✓ | Better hard example handling |
| Focal Alpha | 0.5 | Class weight balance |
| Focal Gamma | 2.0 | Focusing strength |
| Self-Attention | ✓ | Attention pooling layer |
| SE Block | ✓ | Channel attention |
| TCN Block | ✓ | Temporal convolutions |
| Classifier Layers | 256, 128 | Hidden dimensions |

---

## Environment Variables

Override defaults with environment variables:

```bash
# Server configuration
WAKEBUILDER_HOST=0.0.0.0
WAKEBUILDER_PORT=8000

# GPU selection
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=info
```

Create a `.env` file in the project root to persist settings.

---

## Quick Reference

### Recommended Presets

=== "Fast Training"

    - Target Positives: 3,000
    - Hard Negative Ratio: 2.0x
    - Batch Size: 64
    - Max Epochs: 50
    - Early Stopping: 5

=== "Balanced (Default)"

    - Target Positives: 5,000
    - Hard Negative Ratio: 4.0x
    - Batch Size: 32
    - Max Epochs: 100
    - Early Stopping: 8

=== "Maximum Accuracy"

    - Target Positives: 8,000
    - Hard Negative Ratio: 6.0x
    - Batch Size: 32
    - Max Epochs: 200
    - Early Stopping: 15
