# WakeBuilder 🎙️

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

**WakeBuilder** is a comprehensive training platform that enables you to create custom wake word detection models entirely on your local machine—no cloud services, no subscriptions, and no machine learning expertise required.

## 🌟 Features

- **🏠 100% Local Processing**: All training happens on your CPU. Your voice data never leaves your machine.
- **🎯 Simple Interface**: Web-based UI guides you through the entire process in minutes.
- **🚀 Fast Training**: Create production-quality models in 5-15 minutes on typical hardware.
- **🔊 Few-Shot Learning**: Train effective models with just 3-5 voice recordings.
- **🎨 Sophisticated Augmentation**: Automatic generation of hundreds of training variations.
- **🐳 Docker Ready**: One-command deployment with all dependencies included.
- **🆓 Open Source**: Apache 2.0 licensed—use it for anything, commercial or personal.

## 🎯 What is a Wake Word?

A wake word (like "Hey Siri" or "Alexa") is a special phrase that activates a voice assistant. WakeBuilder lets you create your own custom wake words like "Phoenix", "Hey Computer", or any phrase you choose.

## 🏗️ Architecture

WakeBuilder uses a three-layer architecture:

1. **Pre-trained Base Model**: A speech understanding model that already knows what human speech sounds like across diverse speakers and accents.

2. **Wake Word Classifier**: A small neural network trained specifically for your custom wake word using transfer learning.

3. **Training Orchestration**: FastAPI backend that manages data augmentation, model training, evaluation, and real-time testing.

## 📋 Prerequisites

- **Docker & Docker Compose** (recommended) OR
- **Python 3.12+** with `uv` package manager
- **8GB RAM** minimum (16GB recommended)
- **Multi-core CPU** (training will be faster)
- **Microphone** for recording wake word samples

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/wakebuilder.git
cd wakebuilder

# Start WakeBuilder
docker-compose up

# Open your browser
# Navigate to http://localhost:8000
```

### Using Python & uv

```bash
# Clone the repository
git clone https://github.com/yourusername/wakebuilder.git
cd wakebuilder

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the application
uvicorn src.wakebuilder.backend.main:app --host 0.0.0.0 --port 8000

# Open your browser
# Navigate to http://localhost:8000
```

## 📂 Negative Data Setup

WakeBuilder requires negative audio samples (speech that does NOT contain your wake word) to train effective models. On first startup, the application will check if negative data is available.

### Automatic Download (Recommended)

If negative data is missing, the home page will display a download panel. Click **"Download Dataset"** to automatically download the UNAC (Universal Negative Audio Corpus) dataset.

The download progress will be displayed with percentage completion. The dataset is approximately 500MB and will be extracted automatically to the `data/negative/` folder.

### Manual Download

If you prefer to download the dataset manually:

1. Download the UNAC dataset from: https://www.kaggle.com/datasets/rajichisami/universal-negative-audio-corpus-unac
2. Extract the audio files (`.wav`, `.mp3`, `.flac`, or `.ogg`)
3. Place them in the `data/negative/` folder

The application requires at least 100 audio files for training. More files (1000+) will produce better models.

## 📖 How to Use

### 1. Create a New Wake Word

1. Click **"Create New Wake Word"** on the home page
2. Enter your desired wake word (1-2 words, e.g., "Phoenix" or "Hey Computer")
3. Record 3-5 clear samples of yourself saying the wake word
4. Click **"Start Training"**

### 2. Wait for Training

Training typically takes 5-15 minutes. You'll see real-time progress updates:
- ✅ Generating synthetic voice variations
- ✅ Creating negative examples
- ✅ Training classifier network
- ✅ Evaluating model performance

### 3. Test Your Model

After training completes:
- Speak your wake word into the microphone
- Watch for visual feedback when detected
- Adjust sensitivity slider to fine-tune behavior
- Download the model for use with WakeEngine

## 🏗️ Project Structure

```
WakeBuilder/
├── src/
│   └── wakebuilder/
│       ├── training/          # Training pipeline and data augmentation
│       ├── backend/            # FastAPI web server and API endpoints
│       └── frontend/           # Web UI (HTML, CSS, JavaScript)
├── models/
│   ├── default/               # Pre-trained default wake words
│   └── custom/                # Your custom trained models
├── data/
│   └── temp/                  # Temporary storage for recordings
├── tts_voices/                # Piper TTS voice models
├── tests/                     # Unit and integration tests
├── project_spec/              # Project documentation
├── pyproject.toml             # Project configuration and dependencies
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker deployment configuration
└── README.md                  # This file
```

## 🔧 Configuration

WakeBuilder can be configured via environment variables or the `config.py` file:

### Key Configuration Parameters

- **Audio Processing**: Sample rate (16kHz), mel spectrograms (80 bins)
- **Data Augmentation**: Speed variations, pitch shifts, noise levels
- **Training**: Learning rate (0.001), batch size (32), max epochs (50)
- **Model Architecture**: Embedding dim (512), hidden layers [256, 128]

See `src/wakebuilder/config.py` for all configurable parameters.

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/wakebuilder --cov-report=html

# Run specific test file
uv run pytest tests/test_training.py
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install with development dependencies
uv sync --group dev

# Run code formatting
uv run black src/
uv run ruff check src/ --fix

# Run type checking
uv run mypy src/
```

### Code Style

- **Formatter**: Black
- **Linter**: Ruff
- **Type Checker**: mypy
- **Docstrings**: Google style

## 📊 Technical Architecture

### System Overview

WakeBuilder uses **Audio Spectrogram Transformer (AST)** with transfer learning. The pre-trained AST model (`MIT/ast-finetuned-speech-commands-v2`) is frozen and used as a feature extractor, while a custom classifier head is trained for your specific wake word.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ User Records │───▶│ Augmentation │───▶│ Positive Samples     │  │
│  │ 3-5 samples  │    │ + TTS Voices │    │ (2000+ variations)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    AST Feature Extractor                      │  │
│  │              (MIT/ast-finetuned-speech-commands-v2)          │  │
│  │                         FROZEN                                │  │
│  │                                                               │  │
│  │   Audio (16kHz, 1s) ──▶ Spectrogram ──▶ 768-dim Embedding    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Classifier Head                            │  │
│  │                       TRAINABLE                               │  │
│  │                                                               │  │
│  │   768 ──▶ LayerNorm ──▶ Linear(256) ──▶ BatchNorm ──▶ GELU   │  │
│  │       ──▶ Dropout(0.5) ──▶ Linear(128) ──▶ BatchNorm ──▶ GELU│  │
│  │       ──▶ Dropout(0.5) ──▶ Linear(2) ──▶ Softmax             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│                              [Wake Word: 0.92, Not Wake: 0.08]      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### What Gets Fed Into the Model

| Component | Shape | Description |
|-----------|-------|-------------|
| **Raw Audio** | `(16000,)` | 1 second of audio at 16kHz sample rate |
| **Spectrogram** | `(128, 1024)` | Mel spectrogram computed by AST feature extractor |
| **AST Embedding** | `(768,)` | Fixed-size embedding from frozen AST model |
| **Classifier Output** | `(2,)` | Probability for [wake_word, not_wake_word] |

### Data Augmentation Pipeline

#### Positive Samples (Wake Word)

From just 3-5 user recordings, we generate **2000+ positive samples**:

| Augmentation | Variations | Description |
|--------------|------------|-------------|
| **TTS Voices** | 85 voices | Piper TTS with diverse accents/genders |
| **Speed** | 0.9x, 0.95x, 1.0x, 1.05x, 1.1x | Time stretching |
| **Pitch** | -2, -1, 0, +1, +2 semitones | Pitch shifting |
| **Volume** | 0.7x to 1.3x | Amplitude scaling |
| **Time Shift** | -0.1s to +0.1s | Random offset |
| **Noise** | 5dB, 10dB, 15dB, 20dB SNR | Background noise injection |

**Voice Coverage**: ALL 85 TTS voices are used at least once to ensure the model generalizes across different speakers.

#### Negative Samples

Two types of negative samples are generated:

**1. Real Negatives (from LibriSpeech/CommonVoice)**
- Random speech that doesn't contain the wake word
- Target: **1.5x positive samples** (when max=0)
- Chunked into 1-second segments

**2. Hard Negatives (Phonetically Similar Words)**
- Generated algorithmically from the wake word
- Target: **3x positive samples**
- Critical for preventing false positives

Example for wake word "samix":
```
CRITICAL (Pure Prefixes):     sa, sam, sami, saa, sae
HIGH (Prefix Extensions):     samer, sammy, samson, samuel
HIGH (Suffixes):              amix, mix, ix
HIGH (Edit Distance 1):       smix, asamix, samx
MEDIUM (Phonetic Variations): hey samix, hi samix
```

### Data Split

| Set | Positive | Hard Negatives | Real Negatives | Total |
|-----|----------|----------------|----------------|-------|
| **Train (75%)** | ~1500 | ~4500 | ~2250 | ~8250 |
| **Validation (25%)** | ~500 | ~1500 | ~750 | ~2750 |

**Important**: Validation uses **unseen TTS voices** (34 held-out voices) to test generalization.

### Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Batch Size** | 32 | Samples per gradient update |
| **Learning Rate** | 0.0005 | Step size for optimizer |
| **Max Epochs** | 100 | Early stopping halts when converged |
| **Dropout** | 0.5 | Regularization to prevent overfitting |
| **Label Smoothing** | 0.25 | Prevents overconfident predictions |
| **Mixup Alpha** | 0.5 | Data augmentation during training |
| **Weight Decay** | 0.001 | L2 regularization |
| **Patience** | 15 | Epochs to wait before early stopping |

### Classifier Architecture

The trainable classifier head has **~230K parameters**:

```python
WakeWordClassifier(
    input_norm=LayerNorm(768),           # Normalize AST embeddings
    classifier=Sequential(
        Linear(768, 256),                 # 196,864 params
        BatchNorm1d(256),                 # 512 params
        GELU(),                           # Smooth activation
        Dropout(0.5),                     # Regularization
        Linear(256, 128),                 # 32,896 params
        BatchNorm1d(128),                 # 256 params
        GELU(),
        Dropout(0.5),
        Linear(128, 2),                   # 258 params
    )
)
# Total: ~230,786 trainable parameters
```

### Why It Works with Few Samples

WakeBuilder uses **transfer learning**. The AST base model (87M parameters) already understands speech patterns from training on Speech Commands dataset with 35 different words. We freeze this knowledge and only train a small classifier head (~230K parameters) to recognize your specific wake word.

### Inference Pipeline

```
Audio Input (1s @ 16kHz)
         │
         ▼
┌─────────────────────┐
│  AST Feature        │
│  Extractor          │ ──▶ 768-dim embedding
│  (Frozen, 87M)      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Classifier Head    │
│  (Trained, 230K)    │ ──▶ [0.92, 0.08]
└─────────────────────┘
         │
         ▼
    Threshold Check
    (default: 0.5-0.7)
         │
         ▼
   Wake Word Detected!
```

## 🎯 Default Wake Words

WakeBuilder ships with pre-trained models:

**Single Words**: Computer, Assistant, System, Listen, Voice

**Two Words**: Hey There, Wake Up, Hi Computer, Hi Assistant

These are ready to use immediately for testing and demonstrations.

## 📦 Model Output

Each trained model produces:

1. **ONNX Model File** (`.onnx`): Neural network weights in open format
2. **Metadata File** (`.json`): Contains:
   - Wake word text
   - Creation timestamp
   - Recommended detection threshold
   - Performance metrics (accuracy, false positive/negative rates)

## 🔒 Privacy & Security

- **No Cloud**: Everything runs locally on your machine
- **No Telemetry**: No data collection or phone-home features
- **Temporary Storage**: Voice recordings are deleted after training
- **Open Source**: Full transparency—audit the code yourself

## 🚧 Current Status

**Beta Release** - All core functionality implemented and working:

- ✅ Project structure and configuration
- ✅ Dependency management
- ✅ Audio preprocessing pipeline (Phase 1)
- ✅ Training pipeline with AST (Phase 2)
- ✅ FastAPI backend (Phase 3)
- ✅ Web interface (Phase 4)
- 🚧 Docker deployment (Phase 5)
- 🚧 Testing and optimization (Phase 6)

## 🗺️ Roadmap

### Phase 1: Foundation ✅
- [x] Base speech embedding model research and selection
- [x] AST model integration via Hugging Face Transformers
- [x] Audio preprocessing pipeline implementation
- [x] Development environment validation

### Phase 2: Training Pipeline ✅
- [x] Data augmentation system (TTS, speed, pitch, volume, noise)
- [x] Hard negative generator (phonetically similar words)
- [x] Classifier training loop with early stopping
- [x] Model evaluation and threshold calibration

### Phase 3: Backend ✅
- [x] FastAPI endpoints
- [x] Job management system
- [x] WebSocket for real-time testing
- [x] File storage and organization

### Phase 4: Frontend ✅
- [x] Home page and model dashboard
- [x] Training wizard
- [x] Progress tracking interface
- [x] Real-time testing interface

### Phase 5: Deployment (In Progress)
- [ ] Dockerfile
- [ ] Docker Compose configuration
- [x] Piper TTS integration (85 voices)
- [ ] Default model training

### Phase 6: Polish
- [ ] Comprehensive testing
- [ ] Performance optimization
- [x] Documentation
- [ ] Example projects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Workflow

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests and linting (`uv run pytest && uv run ruff check src/`)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Piper TTS**: High-quality local text-to-speech
- **ONNX Runtime**: Efficient cross-platform inference
- **PyTorch**: Deep learning framework
- **FastAPI**: Modern web framework
- **librosa**: Audio processing library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/wakebuilder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/wakebuilder/discussions)
- **Documentation**: [Full Documentation](https://wakebuilder.readthedocs.io)

## 🌟 Related Projects

- **WakeEngine**: Companion library for real-time wake word detection
- **Piper TTS**: Local text-to-speech engine
- **ONNX**: Open Neural Network Exchange format

---

**Made with ❤️ by the WakeBuilder Team**

*Democratizing wake word technology, one voice at a time.*
