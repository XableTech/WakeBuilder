# Development Setup

Setting up your development environment for contributing to WakeBuilder.

---

## Prerequisites

- Python 3.12 or higher
- Git
- uv or pip package manager

---

## Clone and Setup

### 1. Fork and Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder
```

### 2. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Install with dev dependencies
uv sync --group dev
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Check linting
ruff check src/ tests/
```

---

## Development Tools

### Required Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Black** | Code formatting | `pip install black` |
| **Ruff** | Linting | `pip install ruff` |
| **pytest** | Testing | `pip install pytest` |
| **mypy** | Type checking | `pip install mypy` |

### Recommended Tools

| Tool | Purpose |
|------|---------|
| VS Code | IDE with Python support |
| Pre-commit | Git hooks for formatting |
| htop/nvtop | Resource monitoring |

---

## Project Structure

Key directories for development:

```
WakeBuilder/
├── src/wakebuilder/    # Main source code
│   ├── audio/          # Audio processing
│   ├── backend/        # FastAPI server
│   ├── models/         # ML models
│   └── tts/            # TTS providers
├── tests/              # Test suite
├── frontend/           # Web interface
├── scripts/            # Utilities
└── docs/               # Documentation
```

---

## Running the Server

### Development Mode

```bash
uvicorn src.wakebuilder.backend.main:app --reload --port 8000
```

The `--reload` flag enables auto-restart on code changes.

### Debug Mode

```bash
# With verbose logging
LOG_LEVEL=debug uvicorn src.wakebuilder.backend.main:app --reload
```

---

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

Edit files in `src/wakebuilder/`

### 3. Run Tests

```bash
pytest tests/
```

### 4. Format Code

```bash
black src/ tests/
ruff check src/ tests/ --fix
```

### 5. Commit

```bash
git add .
git commit -m "feat: add my feature"
```

---

## Tips

### Working with Audio

```python
import librosa

# Load audio
audio, sr = librosa.load("test.wav", sr=16000)

# Check properties
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Shape: {audio.shape}")
```

### Working with Models

```python
from wakebuilder.models.classifier import ASTWakeWordModel

# Load model
model = ASTWakeWordModel()
model.load_state_dict(torch.load("model.pt"))
```

### Working with TTS

```python
from wakebuilder.tts import TTSGenerator

gen = TTSGenerator()
audio, sr = gen.synthesize("hello world")
```
