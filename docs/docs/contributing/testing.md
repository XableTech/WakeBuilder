# Testing

Guidelines for writing and running tests.

---

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=src/wakebuilder --cov-report=html
```

### Specific File

```bash
pytest tests/test_training.py
```

### Specific Test

```bash
pytest tests/test_training.py::test_classifier_training
```

---

## Test Structure

### Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_api.py           # API endpoint tests
├── test_ast_model.py     # Model architecture tests
├── test_augmentation.py  # Data augmentation tests
├── test_config.py        # Configuration tests
├── test_foundation.py    # Core functionality tests
├── test_hard_negatives.py # Hard negative generation
├── test_preprocessing.py # Audio preprocessing tests
├── test_training.py      # Training pipeline tests
└── test_training_data.py # Dataset creation tests
```

### Naming Conventions

```python
# Test files: test_*.py
# Test functions: test_*
# Test classes: Test*

def test_audio_preprocessing_normalizes_amplitude():
    """Test that preprocessing normalizes audio to [-1, 1]."""
    ...

class TestASTTrainer:
    """Tests for ASTTrainer class."""
    
    def test_train_creates_model(self):
        ...
```

---

## Writing Tests

### Basic Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_audio_normalization():
    """Test that audio is normalized correctly."""
    # Arrange
    audio = np.random.randn(16000) * 10  # Unnormalized
    
    # Act
    normalized = normalize_audio(audio)
    
    # Assert
    assert normalized.max() <= 1.0
    assert normalized.min() >= -1.0
```

### Using Fixtures

```python
# conftest.py
import pytest
import numpy as np

@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    return np.random.randn(16000).astype(np.float32)

@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 16000

# test_*.py
def test_processing(sample_audio, sample_rate):
    result = process_audio(sample_audio, sample_rate)
    assert result is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("speed", [0.8, 1.0, 1.2, 1.5])
def test_speed_augmentation(sample_audio, speed):
    """Test speed augmentation at various values."""
    result = apply_speed_change(sample_audio, 16000, speed)
    
    expected_length = int(len(sample_audio) / speed)
    assert len(result) == pytest.approx(expected_length, rel=0.1)
```

---

## Test Categories

### Unit Tests

Test individual functions:

```python
def test_phonetically_similar_words():
    """Test hard negative generation for 'jarvis'."""
    words = get_phonetically_similar_words("jarvis")
    
    assert "jarv" in words  # Prefix
    assert "jarvi" in words  # Prefix
    assert "javis" in words  # Edit distance 1
```

### Integration Tests

Test component interactions:

```python
def test_training_pipeline_end_to_end():
    """Test complete training pipeline."""
    trainer = ASTTrainer()
    
    # Prepare minimal data
    positives = [generate_test_audio() for _ in range(10)]
    negatives = [generate_test_audio() for _ in range(20)]
    
    # Train
    train_loader, val_loader = trainer.prepare_data(
        positives, negatives, "test"
    )
    trainer.train(train_loader, val_loader, num_epochs=2)
    
    # Verify
    assert trainer.model is not None
```

### API Tests

Test HTTP endpoints:

```python
from fastapi.testclient import TestClient
from wakebuilder.backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
```

---

## Markers

### Skip GPU Tests

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_gpu_training():
    ...
```

### Slow Tests

```python
@pytest.mark.slow
def test_full_training_run():
    ...
```

Run excluding slow tests:

```bash
pytest -m "not slow"
```

---

## Mocking

### Mock External Services

```python
from unittest.mock import patch, MagicMock

def test_tts_generation(sample_audio):
    """Test TTS without actual synthesis."""
    with patch('wakebuilder.tts.TTSGenerator.synthesize') as mock:
        mock.return_value = (sample_audio, 16000)
        
        gen = TTSGenerator()
        audio, sr = gen.synthesize("test")
        
        mock.assert_called_once_with("test")
        assert sr == 16000
```

---

## Test Data

### Generate Test Audio

```python
def generate_test_audio(
    duration: float = 1.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Generate random audio for testing."""
    samples = int(duration * sample_rate)
    return np.random.randn(samples).astype(np.float32)
```

### Temporary Files

```python
def test_save_and_load(tmp_path):
    """Test saving and loading models."""
    model_path = tmp_path / "test.pt"
    
    # Save
    save_model(model, model_path)
    assert model_path.exists()
    
    # Load
    loaded = load_model(model_path)
    assert loaded is not None
```

---

## Coverage Goals

| Module | Target |
|--------|--------|
| `audio/` | 80%+ |
| `models/` | 70%+ |
| `backend/` | 75%+ |
| `tts/` | 60%+ |
| **Overall** | **75%+** |
