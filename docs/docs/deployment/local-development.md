# Local Development

Setting up WakeBuilder for local development and personal use.

---

## Quick Start

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Run the interactive setup
python run.py
```

---

## Manual Setup

For more control over the setup process:

### 1. Create Virtual Environment

=== "Using uv (Recommended)"

    ```bash
    uv venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

=== "Using venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

### 2. Install Dependencies

=== "Using uv"

    ```bash
    uv sync
    ```

=== "Using pip"

    ```bash
    pip install -e .
    ```

### 3. Install PyTorch with CUDA (Optional)

For GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA version.

### 4. Download TTS Models

```bash
python scripts/download_voices.py
```

This downloads ~5GB of Piper TTS voice models.

### 5. Start the Server

```bash
uvicorn src.wakebuilder.backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Server Options

### Command Line Options

```bash
uvicorn src.wakebuilder.backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
```

| Option | Description |
|--------|-------------|
| `--host` | Bind address (0.0.0.0 for all interfaces) |
| `--port` | Port number (default: 8000) |
| `--reload` | Auto-reload on code changes |
| `--log-level` | Logging verbosity |

### Using run.py

```bash
python run.py --run --port 8080
```

---

## Development Workflow

### Code Changes

With `--reload` enabled, the server automatically restarts when you modify:

- Python files in `src/`
- Configuration files

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_training.py

# With coverage
pytest --cov=src/wakebuilder
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## Directory Setup

WakeBuilder creates these directories on first run:

```
WakeBuilder/
├── models/
│   ├── default/      # Pre-trained models
│   └── custom/       # Your trained models
├── data/
│   ├── negative/     # UNAC dataset
│   └── cache/        # Pre-processed chunks
├── recordings/       # Temporary recordings
└── tts_voices/       # Piper TTS models
```

---

## GPU Configuration

### Checking GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Selecting GPU

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python run.py

# Use CPU only
CUDA_VISIBLE_DEVICES=-1 python run.py
```

---

## Troubleshooting

### Server Issues

??? question "Server won't start"

    1. Check port is not in use:
       ```bash
       netstat -tulpn | grep 8000
       ```
    
    2. Try a different port:
       ```bash
       python run.py --port 8001
       ```
    
    3. Check Python version:
       ```bash
       python --version  # Should be 3.12+
       ```

??? question "Import errors"

    1. Ensure virtual environment is activated
    2. Reinstall dependencies:
       ```bash
       pip install -e .
       ```
    3. Check for conflicting packages

??? question "Out of memory"

    1. Close other applications
    2. Reduce batch size in training
    3. Disable some TTS providers

### Training Issues

??? question "Training is very slow"

    1. Enable GPU acceleration
    2. Reduce target positive samples
    3. Lower hard negative ratio

??? question "Training fails with error"

    1. Check server logs for stack trace
    2. Verify audio files are valid
    3. Ensure sufficient disk space

### Audio Issues

??? question "Audio recording not working"

    1. Check browser microphone permissions
    2. Ensure microphone is not muted
    3. Try a different browser

??? question "TTS generation fails"

    1. Verify TTS voices are downloaded:
       ```bash
       python scripts/download_voices.py
       ```
    2. Check available disk space
    3. Review server logs for errors

---

## Recommended Development Tools

| Tool | Purpose |
|------|---------|
| **VS Code** | IDE with Python support |
| **Postman/Insomnia** | API testing |
| **Chrome DevTools** | Frontend debugging |
| **htop/nvtop** | System monitoring |
