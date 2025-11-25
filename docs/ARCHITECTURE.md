# WakeBuilder Architecture

## System Overview

WakeBuilder is a local wake word training platform built with a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                             │
│                    (HTML/CSS/JavaScript)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Training   │  │   Models    │  │   Testing (WebSocket)   │  │
│  │  Endpoints  │  │  Endpoints  │  │      Endpoints          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Job Manager                             │  │
│  │              (Threading-based queue)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    Audio     │  │     Data     │  │      Classifier      │   │
│  │ Preprocessor │  │  Augmenter   │  │       Trainer        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              Neural Network Models                         │  │
│  │         (TC-ResNet / BC-ResNet classifiers)               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
WakeBuilder/
├── src/wakebuilder/
│   ├── __init__.py              # Package version
│   ├── config.py                # Centralized configuration
│   │
│   ├── audio/                   # Audio processing
│   │   ├── preprocessing.py     # Mel spectrogram conversion
│   │   ├── augmentation.py      # Data augmentation (TTS, noise, pitch)
│   │   └── negative_generator.py # Negative example generation
│   │
│   ├── models/                  # Neural networks
│   │   ├── classifier.py        # TC-ResNet & BC-ResNet architectures
│   │   └── trainer.py           # Training loop & threshold calibration
│   │
│   ├── tts/                     # Text-to-speech
│   │   └── generator.py         # Piper TTS wrapper
│   │
│   └── backend/                 # FastAPI backend
│       ├── main.py              # App entry point
│       ├── schemas.py           # Pydantic models
│       ├── jobs.py              # Background job manager
│       └── routes/
│           ├── training.py      # Training endpoints
│           ├── models.py        # Model management
│           └── testing.py       # Testing endpoints
│
├── models/
│   ├── base/                    # Base embedding model
│   ├── default/                 # Pre-trained wake words
│   └── custom/                  # User-trained models
│
├── data/
│   └── temp/                    # Temporary training files
│
├── tts_voices/                  # Piper TTS voice models
│
├── tests/                       # Test suite
│
└── docs/                        # Documentation
```

---

## Component Details

### 1. FastAPI Backend (`backend/`)

#### Main Application (`main.py`)
- FastAPI app with CORS middleware
- Lifespan handler for startup/shutdown
- OpenAPI documentation (Swagger/ReDoc)
- Error handlers

#### Schemas (`schemas.py`)
Pydantic models for request/response validation:
- `TrainingRequest`, `TrainingStatusResponse`
- `ModelMetadata`, `ModelListResponse`
- `TestFileResponse`, `DetectionEvent`

#### Job Manager (`jobs.py`)
Threading-based background job system:
```python
JobManager
├── create_job()      # Create new job
├── start_job()       # Start in background thread
├── get_job()         # Get job by ID
├── get_all_jobs()    # List all jobs
└── delete_job()      # Remove completed job
```

**Job States:**
```
PENDING → VALIDATING → AUGMENTING → GENERATING_NEGATIVES
    → TRAINING → CALIBRATING → SAVING → COMPLETED
                                    └→ FAILED
```

#### Routes

**Training (`routes/training.py`):**
- `POST /api/train/start` - Validates input, creates job, starts training
- `GET /api/train/status/{job_id}` - Returns detailed progress
- `GET /api/train/download/{job_id}` - Streams model ZIP

**Models (`routes/models.py`):**
- Scans `models/default/` and `models/custom/` directories
- Loads metadata from `metadata.json` files
- Supports filtering by category

**Testing (`routes/testing.py`):**
- File-based: Load audio, run inference, return result
- WebSocket: Stream audio chunks, return detection events

---

### 2. Audio Processing (`audio/`)

#### Preprocessor (`preprocessing.py`)
Converts raw audio to mel spectrograms:
```
Audio (16kHz) → Normalize → Mel Spectrogram (80 bins) → Pad/Trim
```

**Parameters:**
- Sample rate: 16,000 Hz
- Mel bins: 80
- FFT size: 512
- Hop length: 160 (10ms)
- Window length: 400 (25ms)

#### Augmenter (`augmentation.py`)
Expands training data through:
1. **TTS Generation** - Multiple voices via Piper
2. **Speed Variation** - 0.8x to 1.2x
3. **Pitch Shifting** - ±2 semitones
4. **Noise Mixing** - Various SNR levels (-5 to -20 dB)
5. **Volume Scaling** - 0.7x to 1.3x

#### Negative Generator (`negative_generator.py`)
Creates non-wake-word examples:
- Phonetically similar words
- Random speech phrases
- Silence with noise
- Pure noise samples

---

### 3. Neural Networks (`models/`)

#### Classifier Architectures (`classifier.py`)

**TC-ResNet (Temporal Convolutional ResNet):**
- Treats mel bins as channels, applies 1D convolution over time
- Fast inference (~0.6ms on CPU)
- ~64K parameters, ~250KB size
- Best for production/real-time

**BC-ResNet (Broadcasted Residual Network):**
- 2D convolution with broadcasted residual learning
- Sub-spectral normalization
- Squeeze-and-Excitation attention
- Higher accuracy (~98% on Speech Commands)
- ~128K parameters, ~468KB size

```python
# Model creation
model = create_model(
    model_type="bc_resnet",  # or "tc_resnet"
    num_classes=2,
    n_mels=80
)
```

#### Trainer (`trainer.py`)

**Training Pipeline:**
1. Prepare data (augmentation + negatives)
2. Create model
3. Train with:
   - AdamW optimizer
   - OneCycleLR scheduler
   - Label smoothing
   - Mixup augmentation
   - Early stopping
4. Calibrate threshold (FAR/FRR analysis)
5. Save model + metadata

**Threshold Calibration:**
```python
optimal_threshold, metrics = calibrate_threshold(
    model, val_loader, device, num_thresholds=100
)
# Returns threshold that minimizes 2*FAR + FRR
```

---

### 4. Configuration (`config.py`)

Centralized settings:

```python
# Audio
SAMPLE_RATE = 16000
N_MELS = 80
DURATION = 1.0  # seconds

# Training
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20

# Augmentation
SPEED_VARIATIONS = [0.8, 0.9, 1.0, 1.1, 1.2]
PITCH_SHIFTS = [-2, -1, 0, 1, 2]
NOISE_LEVELS = [-20, -15, -10, -5]  # SNR dB
NEGATIVE_RATIO = 4  # negatives per positive

# Wake word validation
MIN_LENGTH = 2
MAX_LENGTH = 30
MAX_WORDS = 2
```

---

## Data Flow

### Training Flow

```
User Recordings (3-5 WAV files)
        │
        ▼
┌───────────────────┐
│   Validation      │ ← Check format, duration
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Augmentation     │ ← TTS + noise + pitch + speed
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Negative Gen      │ ← Similar words + random speech + noise
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Preprocessing    │ ← Audio → Mel Spectrogram
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Training       │ ← BC-ResNet/TC-ResNet
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Calibration     │ ← Find optimal threshold
└───────────────────┘
        │
        ▼
    model.pt + metadata.json
```

### Inference Flow

```
Audio Input (file or stream)
        │
        ▼
┌───────────────────┐
│  Preprocessing    │ ← Resample → Normalize → Mel Spec
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Model         │ ← Forward pass
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Softmax        │ ← Get probability
└───────────────────┘
        │
        ▼
    confidence >= threshold → DETECTED
```

---

## Model File Format

Each trained model is stored as a directory:

```
models/custom/hey_computer/
├── model.pt              # PyTorch checkpoint
├── metadata.json         # Configuration & metrics
└── training_history.json # Epoch-by-epoch metrics
```

**model.pt contents:**
```python
{
    "model_state_dict": {...},  # Weights
    "model_type": "bc_resnet",
    "n_mels": 80,
    "base_channels": 16,
    "scale": 1.0
}
```

**metadata.json:**
```json
{
    "wake_word": "Hey Computer",
    "threshold": 0.65,
    "model_type": "bc_resnet",
    "parameters": 128000,
    "metrics": {
        "val_accuracy": 0.952,
        "val_f1": 0.948
    },
    "threshold_analysis": [...]
}
```

---

## Threading Model

```
Main Thread (FastAPI/Uvicorn)
    │
    ├── HTTP Request Handlers
    │
    └── WebSocket Handlers
            │
            └── Audio processing loop

Background Thread (per training job)
    │
    └── Training Pipeline
        ├── Data augmentation
        ├── Model training
        └── Threshold calibration
```

**Concurrency:**
- One training job at a time (configurable)
- Multiple WebSocket connections supported
- Thread-safe job status updates via locks

---

## Error Handling

1. **Validation Errors** - Caught at Pydantic schema level
2. **Training Errors** - Caught in job thread, stored in `job.error`
3. **File Errors** - Wrapped with HTTPException
4. **WebSocket Errors** - Sent as JSON error messages

---

## Future Considerations

1. **Scaling** - Replace threading with Celery for distributed training
2. **GPU Support** - Auto-detect and use CUDA when available
3. **Model Versioning** - Track model lineage and comparisons
4. **Batch Training** - Train multiple wake words in parallel
5. **Export Formats** - ONNX, TFLite for edge deployment
