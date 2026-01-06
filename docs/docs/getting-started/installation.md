# Installation

Complete guide to installing WakeBuilder on your system.

---

## System Requirements

### Hardware Requirements

!!! warning "GPU Strongly Recommended"
    Training on CPU is **extremely slow** (potentially several hours). A CUDA-compatible NVIDIA GPU is strongly recommended for practical training times.

| Resource | Minimum | Recommended | Purpose |
|----------|---------|-------------|---------|
| **RAM** | 8GB free | 16GB+ free | Data augmentation and training |
| **GPU VRAM** | 6GB NVIDIA | 8GB+ NVIDIA | CUDA acceleration (**strongly recommended**) |
| **Storage** | 10GB free | 20GB+ free | TTS models and training data |
| **CPU** | Multi-core | 8+ cores | Parallel data processing |

### Expected Training Times

| Hardware | Approximate Training Time |
|----------|---------------------------|
| High-end GPU (RTX 3080+) | 30 min - 1.5 hours |
| Mid-range GPU (RTX 3060) | 1 - 2 hours |
| Entry GPU (GTX 1650) | 2 - 4 hours |
| CPU only | **4+ hours** (not recommended) |

### Software Requirements

| Software | Required Version | Notes |
|----------|-----------------|-------|
| **Python** | 3.12+ | Core runtime |
| **pip or uv** | Latest | Package management |
| **Git** | Any | Repository cloning |
| **Docker** | 20.10+ | Optional, for containerized deployment |
| **CUDA** | 11.8+ | Strongly recommended for GPU acceleration |

---

## Installation Methods

### Method 1: Using run.py (Recommended)

The interactive setup script handles everything automatically:

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Run the setup script
python run.py
```

#### Script Options

| Option | Description |
|--------|-------------|
| `--auto` or `-y` | Automatic mode (no prompts) |
| `--check` | Verify environment only |
| `--install` | Install dependencies only |
| `--download` | Download TTS models only |
| `--run` | Run server only (skip setup) |
| `--cuda VERSION` | Use specific CUDA version (e.g., `12.4`, `12.1`, `cpu`) |
| `--port PORT` | Run server on specified port |
| `--help` | Show all options |

#### Example Commands

```bash
# Full automatic setup
python run.py --auto

# Setup with specific CUDA version
python run.py --cuda 12.4

# CPU-only installation
python run.py --cuda cpu

# Just run the server (after setup)
python run.py --run
```

---

### Method 2: Manual Installation with uv

[uv](https://github.com/astral-sh/uv) is the recommended package manager for faster installs:

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Download TTS models
python -c "from scripts.download_voices import main; main()"

# Run the server
uvicorn src.wakebuilder.backend.main:app --host 0.0.0.0 --port 8000
```

---

### Method 3: Manual Installation with pip

Traditional pip installation:

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install WakeBuilder
pip install -e .

# Download TTS models
python scripts/download_voices.py

# Run the server
uvicorn src.wakebuilder.backend.main:app --host 0.0.0.0 --port 8000
```

!!! note "CUDA Versions"
    Replace `cu128` with your CUDA version:

    - `cu128` - CUDA 12.8
    - `cu126` - CUDA 12.6
    - `cu124` - CUDA 12.4
    - `cu121` - CUDA 12.1
    - `cpu` - CPU-only

---

### Method 4: Docker Installation

For containerized deployment:

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Build and run with Docker Compose
docker-compose up
```

See the [Docker Deployment Guide](../deployment/docker.md) for detailed instructions.

---

## Post-Installation Steps

### 1. Download Negative Data

WakeBuilder requires negative audio samples for training. On first run, you'll see a download panel on the home page:

1. Navigate to `http://localhost:8000`
2. Click **"Download Dataset"** in the Negative Data panel
3. Wait for the UNAC dataset to download (~1.4GB)

Alternatively, download manually:

1. Get the UNAC dataset from [Kaggle](https://www.kaggle.com/datasets/rajichisami/universal-negative-audio-corpus-unac)
2. Extract audio files to `data/negative/`

### 2. Build the Negative Cache

For faster training, build the negative data cache:

1. Click **"Build Cache"** on the home page
2. Wait for preprocessing to complete (~47,000 audio chunks)

This step only needs to be done once.

### 3. Verify Installation

Run the verification command:

```bash
python run.py --check
```

This checks:

- Python version
- Required packages
- CUDA availability
- TTS model availability
- Negative data availability

---

## Environment Variables

Optional environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `WAKEBUILDER_HOST` | `0.0.0.0` | Server host binding |
| `WAKEBUILDER_PORT` | `8000` | Server port |
| `WAKEBUILDER_RELOAD` | `false` | Enable auto-reload |
| `CUDA_VISIBLE_DEVICES` | all | GPU device selection |

Create a `.env` file in the project root:

```bash
WAKEBUILDER_PORT=8080
CUDA_VISIBLE_DEVICES=0
```

---

## Upgrading

To upgrade WakeBuilder:

```bash
cd WakeBuilder

# Pull latest changes
git pull origin main

# Update dependencies
python run.py --install
```

---

## Uninstalling

To completely remove WakeBuilder:

```bash
# Use the cleanup script
python clean.py --all
```

This removes:

- Virtual environment
- TTS voices
- Training data
- Trained models
- Recordings
- Cache directories
