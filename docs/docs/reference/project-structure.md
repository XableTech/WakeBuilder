# Project Structure

Complete reference for WakeBuilder's codebase organization.

---

## Repository Overview

```
WakeBuilder/
├── src/                      # Source code
│   └── wakebuilder/          # Main package
├── frontend/                 # Web interface
├── models/                   # Trained models
├── data/                     # Training data
├── tts_voices/               # Piper TTS voice models
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── docs/                     # Documentation (this site)
├── project_spec/             # Project specifications
└── recordings/               # Temporary recordings
```

---

## Source Code (`src/wakebuilder/`)

### Main Package

```
src/wakebuilder/
├── __init__.py               # Package initialization, version
├── config.py                 # Centralized configuration
├── audio/                    # Audio processing
├── backend/                  # FastAPI server
├── models/                   # ML models
└── tts/                      # TTS providers
```

---

### Audio Module (`audio/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `augmentation.py` | Data augmentation (speed, pitch, noise) |
| `negative_generator.py` | Hard negative generation |
| `preprocessing.py` | Audio preprocessing, mel spectrograms |
| `real_data_loader.py` | UNAC dataset loading and caching |

#### Key Classes

| Class | Location | Description |
|-------|----------|-------------|
| `DataAugmenter` | `augmentation.py` | Audio augmentation pipeline |
| `NoiseLoader` | `augmentation.py` | Background noise loading |
| `AugmentedSample` | `augmentation.py` | Augmented audio container |
| `NegativeExampleGenerator` | `negative_generator.py` | Hard negative generator |
| `RealNegativeDataLoader` | `real_data_loader.py` | Cached negative data loader |
| `AudioPreprocessor` | `preprocessing.py` | Mel spectrogram computation |

---

### Backend Module (`backend/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `main.py` | FastAPI application entry |
| `jobs.py` | Training job management |
| `schemas.py` | Pydantic data models |
| `routes/` | API endpoint definitions |

#### Routes (`routes/`)

| File | Endpoint Prefix | Purpose |
|------|-----------------|---------|
| `training.py` | `/api/train` | Training job management |
| `models.py` | `/api/models` | Model CRUD operations |
| `testing.py` | `/api/test` | Model testing and WebSocket |
| `cache.py` | `/api/cache` | Negative cache management |

#### Key Classes

| Class | Location | Description |
|-------|----------|-------------|
| `JobInfo` | `jobs.py` | Training job state |
| `TrainingStartResponse` | `schemas.py` | API response model |
| `ModelLoader` | `routes/testing.py` | Cached model loader |

---

### Models Module (`models/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports |
| `classifier.py` | AST classifier architecture |
| `trainer.py` | Training pipeline |

#### Key Classes

| Class | Location | Description |
|-------|----------|-------------|
| `ASTWakeWordModel` | `classifier.py` | Complete model (AST + classifier) |
| `WakeWordClassifier` | `classifier.py` | Trainable classifier head |
| `SelfAttentionPooling` | `classifier.py` | Attention mechanism |
| `SqueezeExcitation` | `classifier.py` | SE block |
| `TemporalConvBlock` | `classifier.py` | TCN block |
| `ASTTrainer` | `trainer.py` | Training orchestration |
| `TrainingConfig` | `trainer.py` | Training configuration |
| `FocalLoss` | `trainer.py` | Custom loss function |
| `ASTDataset` | `trainer.py` | PyTorch dataset |

---

### TTS Module (`tts/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Module exports, factory functions |
| `generator.py` | Piper TTS wrapper |
| `edge_generator.py` | Edge TTS wrapper |
| `kokoro_generator.py` | Kokoro TTS wrapper |
| `coqui_generator.py` | Coqui TTS wrapper |

#### Key Classes

| Class | Location | Description |
|-------|----------|-------------|
| `TTSGenerator` | `generator.py` | Piper TTS interface |
| `VoiceInfo` | `generator.py` | Voice metadata |
| `EdgeTTSGenerator` | `edge_generator.py` | Edge TTS interface |
| `KokoroTTSGenerator` | `kokoro_generator.py` | Kokoro TTS interface |
| `CoquiTTSGenerator` | `coqui_generator.py` | Coqui TTS interface |

---

## Frontend (`frontend/`)

```
frontend/
├── index.html            # Single-page application
├── css/
│   └── styles.css        # All styling
├── js/
│   ├── app.js            # Main application logic
│   ├── api.js            # Backend communication
│   ├── audio.js          # Audio recording
│   ├── charts.js         # Training visualizations
│   ├── trainer.js        # Training workflow
│   ├── tester.js         # Model testing
│   └── utils.js          # Utility functions
└── assets/
    └── favicon.svg       # Site icon
```

---

## Data Directories

### Models (`models/`)

```
models/
├── default/              # Pre-trained default models
│   └── assistant/
│       ├── assistant.pt
│       └── assistant.json
└── custom/               # User-trained models
    └── jarvis/
        ├── jarvis.pt
        ├── jarvis.json
        └── jarvis.onnx
```

### Data (`data/`)

```
data/
├── negative/             # UNAC dataset
│   ├── music/
│   ├── speech/
│   ├── ambient/
│   └── ...
├── cache/
│   └── negative_chunks/  # Pre-processed audio chunks
└── temp/                 # Temporary training files
```

### TTS Voices (`tts_voices/`)

```
tts_voices/
├── en_US-amy-low.onnx
├── en_US-amy-low.onnx.json
├── en_US-arctic-medium.onnx
├── en_US-arctic-medium.onnx.json
└── ... (85+ voice models)
```

---

## Scripts (`scripts/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_voices.py` | Download Piper TTS voices | `python scripts/download_voices.py` |
| `prepare_noise.py` | Prepare noise samples | `python scripts/prepare_noise.py` |
| `build_negative_cache.py` | Pre-process negative data | `python scripts/build_negative_cache.py` |
| `diagnose_model.py` | Debug trained models | `python scripts/diagnose_model.py` |
| `analyze_model.py` | Analyze model architecture | `python scripts/analyze_model.py` |
| `convert_mp3_to_wav.py` | Convert audio formats | `python scripts/convert_mp3_to_wav.py` |
| `preview_edge_tts.py` | Test Edge TTS voices | `python scripts/preview_edge_tts.py` |
| `preview_kokoro_tts.py` | Test Kokoro TTS voices | `python scripts/preview_kokoro_tts.py` |

---

## Tests (`tests/`)

| File | Tests |
|------|-------|
| `test_api.py` | API endpoint testing |
| `test_ast_model.py` | AST model architecture |
| `test_augmentation.py` | Data augmentation |
| `test_config.py` | Configuration loading |
| `test_foundation.py` | Core functionality |
| `test_hard_negatives.py` | Hard negative generation |
| `test_preprocessing.py` | Audio preprocessing |
| `test_training.py` | Training pipeline |
| `test_training_data.py` | Dataset creation |
| `conftest.py` | Pytest fixtures |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies |
| `requirements.txt` | Pip dependencies |
| `pytest.ini` | Pytest configuration |
| `Dockerfile` | Container build instructions |
| `docker-compose.yml` | Container orchestration |
| `.env.example` | Environment variable template |
| `.gitignore` | Git ignore rules |
| `run.py` | Setup and run script |
| `clean.py` | Cleanup script |

---

## Project Specifications (`project_spec/`)

| File | Description |
|------|-------------|
| `wakebuilder_project_description.md` | High-level project overview |
| `wakebuilder_project_plan.md` | Implementation phases |
| `wakebuilder_todo.md` | Feature backlog |
