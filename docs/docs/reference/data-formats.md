# Data Formats

File formats, schemas, and data structures used in WakeBuilder.

---

## Audio Formats

### Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Preferred format |
| MP3 | `.mp3` | Converted automatically |
| FLAC | `.flac` | Lossless compression |
| OGG | `.ogg` | Vorbis audio |

### Internal Audio Format

All audio is converted to:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Channels | 1 (mono) |
| Bit depth | 32-bit float |
| Duration | ~1 second |
| Normalization | Peak normalized to ±1.0 |

---

## Model Files

### PyTorch Model (`.pt`)

Contains the trained classifier weights:

```python
{
    "classifier_state_dict": OrderedDict(...),
    "config": {
        "embedding_dim": 768,
        "hidden_dims": [256, 128],
        "dropout": 0.3,
        "use_attention": True,
        "use_se_block": True,
        "use_tcn": True
    }
}
```

### Metadata JSON (`.json`)

```json
{
  "wake_word": "jarvis",
  "model_type": "ast",
  "created_at": "2026-01-06T15:00:00Z",
  "version": "0.1.0",
  
  "threshold": 0.65,
  
  "metrics": {
    "accuracy": 0.971,
    "f1_score": 0.943,
    "precision": 0.952,
    "recall": 0.934,
    "far": 0.023,
    "frr": 0.066
  },
  
  "threshold_analysis": {
    "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "far_values": [0.15, 0.08, 0.04, 0.023, 0.012, 0.005, 0.001],
    "frr_values": [0.01, 0.02, 0.03, 0.066, 0.12, 0.22, 0.45]
  },
  
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "dropout": 0.5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.5,
    "use_focal_loss": true,
    "focal_alpha": 0.5,
    "focal_gamma": 2.0,
    "use_attention": true,
    "use_se_block": true,
    "use_tcn": true,
    "classifier_dims": [256, 128],
    "epochs_trained": 45,
    "early_stopped": true
  },
  
  "data_stats": {
    "recordings": 3,
    "positive_samples": 5000,
    "negative_samples": 10000,
    "hard_negatives": 20000,
    "train_size": 26250,
    "val_size": 8750
  },
  
  "trainable_params": 234567,
  
  "recordings": [
    "recording_001.wav",
    "recording_002.wav",
    "recording_003.wav"
  ]
}
```

### ONNX Model (`.onnx`)

ONNX format for deployment:

**Inputs:**

| Name | Shape | Type |
|------|-------|------|
| `input_values` | `[1, 1024, 128]` | float32 |

**Outputs:**

| Name | Shape | Type |
|------|-------|------|
| `logits` | `[1, 2]` | float32 |

---

## Cache Formats

### Negative Audio Chunks

Stored as NumPy arrays:

```
data/cache/negative_chunks/
├── chunk_00001.npy   # (16000,) float32 array
├── chunk_00002.npy
└── ...
```

Each chunk:

- 1 second of audio
- 16,000 samples at 16kHz
- Float32 normalized to ±1.0

### Chunk Metadata

```json
{
  "total_chunks": 47438,
  "chunk_duration": 1.0,
  "sample_rate": 16000,
  "overlap": 0.5,
  "source_files": 1250,
  "categories": {
    "music": 15000,
    "speech": 20000,
    "ambient": 8000,
    "silence": 4438
  },
  "created_at": "2026-01-05T10:00:00Z"
}
```

---

## Configuration

### config.py Structure

```python
# Audio Configuration
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

# Data Augmentation
AUGMENTATION_CONFIG = {
    "speed_range": [0.9, 1.1],
    "speed_fast": [1.3, 1.5],
    "pitch_range": [-2, 2],
    "volume_range": [0.7, 1.3],
    "noise_snr_range": [-20, -5],
}

# Training Defaults
TRAINING_DEFAULTS = {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "max_epochs": 100,
    "early_stopping_patience": 8,
    "dropout": 0.5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.5,
    "validation_split": 0.25,
}

# AST Model
AST_CONFIG = {
    "model_checkpoint": "MIT/ast-finetuned-speech-commands-v2",
    "embedding_dim": 768,
    "freeze_base": True,
}

# Classifier Architecture
CLASSIFIER_CONFIG = {
    "hidden_dims": [256, 128],
    "use_attention": True,
    "use_se_block": True,
    "use_tcn": True,
    "num_classes": 2,
}
```

---

## TTS Voice Files

### Piper Voice Model

```
tts_voices/
├── en_US-amy-low.onnx         # ONNX model
├── en_US-amy-low.onnx.json    # Voice configuration
```

### Voice Configuration JSON

```json
{
  "audio": {
    "sample_rate": 22050
  },
  "espeak": {
    "voice": "en-us"
  },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8
  },
  "phoneme_type": "espeak",
  "phoneme_map": {}
}
```

---

## API Schemas

### Training Start Request

```json
{
  "wake_word": "string (4-12 chars)",
  "audio_files": ["file1.wav", "file2.wav"],
  "batch_size": 32,
  "learning_rate": 0.0001,
  "num_epochs": 100,
  "dropout": 0.5,
  "use_focal_loss": true,
  "use_attention": true,
  "target_positive_samples": 5000,
  "negative_ratio": 2.0,
  "hard_negative_ratio": 4.0
}
```

### Training Status Response

```json
{
  "job_id": "string (UUID)",
  "status": "pending|starting|training|completed|failed",
  "phase": "string (current phase)",
  "progress": 0.0-100.0,
  "current_epoch": 0,
  "total_epochs": 100,
  "metrics": {
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_accuracy": 0.0,
    "val_f1": 0.0
  },
  "elapsed_time": "0m 0s",
  "error": "string (if failed)"
}
```

### Model Response

```json
{
  "id": "string",
  "wake_word": "string",
  "type": "custom|default",
  "created_at": "ISO 8601 datetime",
  "threshold": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "f1_score": 0.0-1.0,
  "has_onnx": true
}
```

### Test Result Response

```json
{
  "detected": true,
  "confidence": 0.0-1.0,
  "threshold": 0.0-1.0,
  "model_id": "string",
  "processing_time_ms": 0
}
```

---

## Directory Structure

### Complete Project Layout

```
WakeBuilder/
├── src/wakebuilder/          # Source code
├── frontend/                 # Web interface
├── models/
│   ├── default/             # Pre-trained models
│   │   └── {name}/
│   │       ├── {name}.pt
│   │       └── {name}.json
│   └── custom/              # User models
│       └── {name}/
│           ├── {name}.pt
│           ├── {name}.json
│           └── {name}.onnx
├── data/
│   ├── negative/            # UNAC dataset
│   │   ├── music/
│   │   ├── speech/
│   │   └── ambient/
│   └── cache/
│       └── negative_chunks/
├── tts_voices/              # Piper voices
├── recordings/              # Temporary
├── scripts/                 # Utilities
├── tests/                   # Test suite
└── docs/                    # Documentation
```
