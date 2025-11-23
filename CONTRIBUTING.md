# Contributing to WakeBuilder

Thank you for your interest in contributing to WakeBuilder! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Relevant logs or error messages**
- **Screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description** of the feature
- **Use case** explaining why this would be useful
- **Possible implementation** approach (if you have ideas)
- **Examples** from other projects (if applicable)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure tests pass** and code is properly formatted
6. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- Python 3.12 or higher
- `uv` package manager
- Git

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/wakebuilder.git
cd wakebuilder

# Create and activate virtual environment
uv sync --group dev

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify setup
uv run pytest
```

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters maximum
- **Formatter**: Black
- **Linter**: Ruff
- **Type hints**: Required for all functions
- **Docstrings**: Google style for all public functions and classes

### Running Code Quality Tools

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Fix linting issues automatically
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/
```

### Docstring Example

```python
def process_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Process audio file and return normalized waveform.
    
    Args:
        audio_path: Path to the audio file to process.
        sample_rate: Target sample rate in Hz. Defaults to 16000.
    
    Returns:
        Normalized audio waveform as numpy array.
    
    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If audio file format is not supported.
    
    Example:
        >>> audio = process_audio("wake_word.wav")
        >>> print(audio.shape)
        (16000,)
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Aim for high code coverage (>80%)
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

### Test Structure

```python
def test_audio_preprocessing_normalizes_amplitude():
    """Test that audio preprocessing normalizes amplitude to [-1, 1]."""
    # Arrange
    audio = np.random.randn(16000) * 10  # Unnormalized audio
    
    # Act
    normalized = preprocess_audio(audio)
    
    # Assert
    assert normalized.max() <= 1.0
    assert normalized.min() >= -1.0
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/wakebuilder --cov-report=html

# Run specific test file
uv run pytest tests/test_training.py

# Run specific test
uv run pytest tests/test_training.py::test_classifier_training
```

## Project Structure

Understanding the project structure will help you navigate the codebase:

```
src/wakebuilder/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings
├── training/             # Training pipeline
│   ├── __init__.py
│   ├── augmentation.py   # Data augmentation
│   ├── classifier.py     # Wake word classifier
│   ├── evaluation.py     # Model evaluation
│   └── pipeline.py       # Training orchestration
├── backend/              # FastAPI backend
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── api/             # API endpoints
│   └── websocket.py     # WebSocket handlers
└── frontend/            # Web interface
    ├── static/          # CSS, JS, images
    └── templates/       # HTML templates
```

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(training): add pitch shift augmentation

Implement pitch shifting in data augmentation pipeline to improve
model robustness across different voice pitches.

Closes #123
```

```
fix(backend): handle WebSocket disconnection gracefully

Add proper error handling for WebSocket disconnections during
real-time testing to prevent server crashes.

Fixes #456
```

## Branch Naming

Use descriptive branch names with prefixes:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or updates

Examples:
- `feature/add-noise-augmentation`
- `fix/websocket-connection-error`
- `docs/update-installation-guide`

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features or bug fixes
3. **Run the full test suite** and ensure all tests pass
4. **Format and lint** your code
5. **Update CHANGELOG.md** with your changes
6. **Request review** from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran and their results

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings generated
```

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority

- **Audio Processing**: Improve preprocessing pipeline efficiency
- **Data Augmentation**: Add new augmentation techniques
- **Model Optimization**: Quantization and performance improvements
- **Testing**: Increase test coverage
- **Documentation**: Tutorials and examples

### Medium Priority

- **Web Interface**: UI/UX improvements
- **Multi-language Support**: Add support for non-English wake words
- **Alternative Models**: Integrate different base embedding models
- **Deployment**: Docker optimization and deployment guides

### Low Priority

- **Visualization**: Training progress visualization
- **Analytics**: Model performance analytics
- **Export Formats**: Additional model export formats
- **Integration Examples**: Example projects using WakeBuilder

## Questions?

If you have questions about contributing:

1. Check the [documentation](https://wakebuilder.readthedocs.io)
2. Search [existing issues](https://github.com/yourusername/wakebuilder/issues)
3. Ask in [GitHub Discussions](https://github.com/yourusername/wakebuilder/discussions)
4. Open a new issue with the `question` label

## Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **CHANGELOG.md** for significant contributions
- **GitHub contributors** page

Thank you for contributing to WakeBuilder! 🎙️
