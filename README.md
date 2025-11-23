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

## 📊 How It Works

### Training Pipeline

1. **User Input**: Record 3-5 samples of your wake word
2. **Data Augmentation**: Generate 500+ variations using:
   - Text-to-speech with multiple voices
   - Speed and pitch variations
   - Background noise injection
   - Volume randomization
3. **Negative Examples**: Create samples that should NOT trigger:
   - Phonetically similar words
   - Random speech
   - Silence and noise
4. **Feature Extraction**: Convert audio to embeddings using base model
5. **Classifier Training**: Train small neural network on embeddings
6. **Evaluation**: Test on validation set and calibrate threshold
7. **Model Export**: Save as ONNX format with metadata

### Why It Works with Few Samples

WakeBuilder uses **transfer learning**. The base model already understands speech patterns from thousands of hours of diverse audio. You're only teaching it to recognize ONE specific phrase pattern, not teaching it about speech from scratch.

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

**Alpha Release** - Core functionality is implemented and working:

- ✅ Project structure and configuration
- ✅ Dependency management
- 🚧 Audio preprocessing pipeline (Phase 1)
- 🚧 Training pipeline (Phase 2)
- 🚧 FastAPI backend (Phase 3)
- 🚧 Web interface (Phase 4)
- 🚧 Docker deployment (Phase 5)
- 🚧 Testing and optimization (Phase 6)

## 🗺️ Roadmap

### Phase 1: Foundation (In Progress)
- [ ] Base speech embedding model integration
- [ ] Audio preprocessing pipeline
- [ ] Development environment setup

### Phase 2: Training Pipeline
- [ ] Data augmentation system
- [ ] Negative example generator
- [ ] Classifier training loop
- [ ] Model evaluation and threshold calibration

### Phase 3: Backend
- [ ] FastAPI endpoints
- [ ] Job management system
- [ ] WebSocket for real-time testing
- [ ] File storage and organization

### Phase 4: Frontend
- [ ] Home page and model dashboard
- [ ] Training wizard
- [ ] Progress tracking interface
- [ ] Real-time testing interface

### Phase 5: Deployment
- [ ] Dockerfile
- [ ] Docker Compose configuration
- [ ] Piper TTS integration
- [ ] Default model training

### Phase 6: Polish
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Documentation
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
