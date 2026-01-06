# Code Style

Coding standards and formatting guidelines for WakeBuilder.

---

## Python Style

### General Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Maximum line length: 100 characters
- Use type hints for all functions
- Write docstrings in Google style

### Formatting with Black

```bash
# Format all code
black src/ tests/

# Check without changing
black --check src/ tests/
```

### Linting with Ruff

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check src/ tests/ --fix
```

---

## Type Hints

All functions should have type hints:

```python
def process_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True
) -> tuple[np.ndarray, int]:
    """Process audio array."""
    ...
```

### Common Types

```python
from typing import Optional, Union
from pathlib import Path
import numpy as np

# Use these patterns
audio: np.ndarray
path: Path
value: Optional[float] = None
items: list[str]
mapping: dict[str, int]
```

---

## Docstrings

Use Google-style docstrings:

```python
def train_model(
    audio_files: list[Path],
    wake_word: str,
    epochs: int = 50,
) -> Path:
    """Train a wake word model.
    
    Args:
        audio_files: List of paths to training audio files.
        wake_word: The wake word text.
        epochs: Number of training epochs. Defaults to 50.
    
    Returns:
        Path to the saved model file.
    
    Raises:
        ValueError: If no audio files provided.
        RuntimeError: If training fails.
    
    Example:
        >>> model_path = train_model(
        ...     audio_files=[Path("rec1.wav"), Path("rec2.wav")],
        ...     wake_word="jarvis"
        ... )
        >>> print(model_path)
        models/custom/jarvis/jarvis.pt
    """
    ...
```

---

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `process_audio()` |
| Variables | snake_case | `sample_rate` |
| Classes | PascalCase | `AudioProcessor` |
| Constants | UPPER_SNAKE | `SAMPLE_RATE` |
| Private | _prefix | `_internal_method()` |

---

## Import Order

```python
# Standard library
import os
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import torch
from fastapi import FastAPI

# Local
from wakebuilder.config import Config
from wakebuilder.audio import preprocess
```

---

## File Structure

```python
"""Module docstring explaining purpose.

This module provides...
"""

# Imports (grouped)
import ...

# Constants
SAMPLE_RATE = 16000

# Classes
class MyClass:
    """Class docstring."""
    ...

# Functions
def my_function():
    """Function docstring."""
    ...

# Entry point (if applicable)
if __name__ == "__main__":
    ...
```

---

## Error Handling

```python
# Use specific exceptions
class AudioProcessingError(Exception):
    """Raised when audio processing fails."""
    pass

# Provide helpful messages
def load_audio(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        audio, sr = librosa.load(path)
    except Exception as e:
        raise AudioProcessingError(f"Failed to load {path}: {e}")
    
    return audio
```

---

## Comments

```python
# Good: Explains WHY
# Use 0.7 threshold based on empirical testing across diverse voices
threshold = 0.7

# Bad: Explains WHAT (obvious from code)
# Set threshold to 0.7
threshold = 0.7
```

---

## Pre-commit Hooks

Install pre-commit for automatic formatting:

```bash
# Install
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
```
