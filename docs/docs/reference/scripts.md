# Scripts Reference

Available utility scripts and their usage.

---

## Overview

WakeBuilder includes utility scripts in the `scripts/` directory for various tasks:

| Script | Purpose |
|--------|---------|
| `download_voices.py` | Download Piper TTS voices |
| `build_negative_cache.py` | Pre-process negative data |
| `diagnose_model.py` | Debug trained models |
| `analyze_model.py` | Analyze model architecture |
| `prepare_noise.py` | Prepare noise samples |
| `convert_mp3_to_wav.py` | Convert audio formats |
| `preview_edge_tts.py` | Test Edge TTS voices |
| `preview_kokoro_tts.py` | Test Kokoro TTS voices |

---

## Voice Download

### download_voices.py

Downloads Piper TTS voice models.

**Usage:**

```bash
python scripts/download_voices.py
```

**Options:**

| Option | Description |
|--------|-------------|
| `--output-dir` | Output directory (default: `tts_voices/`) |
| `--languages` | Languages to download (default: English) |
| `--quality` | Voice quality: low, medium, high |

**Output:**

Downloads voice model files (~5GB total):

```
tts_voices/
├── en_US-amy-low.onnx
├── en_US-amy-low.onnx.json
├── en_US-arctic-medium.onnx
└── ... (85+ voices)
```

---

## Cache Management

### build_negative_cache.py

Pre-processes negative audio data for faster training.

**Usage:**

```bash
python scripts/build_negative_cache.py
```

**Options:**

| Option | Description |
|--------|-------------|
| `--input-dir` | Negative data directory |
| `--output-dir` | Cache output directory |
| `--workers` | Parallel workers (default: 4) |
| `--chunk-duration` | Chunk length in seconds |

**Output:**

Creates chunked audio files:

```
data/cache/negative_chunks/
├── chunk_00001.npy
├── chunk_00002.npy
└── ... (~47,000 chunks)
```

---

## Model Diagnostics

### diagnose_model.py

Diagnoses issues with trained models.

**Usage:**

```bash
python scripts/diagnose_model.py
```

**Features:**

- Lists all available models
- Loads and validates model weights
- Tests inference with sample audio
- Reports any issues found

**Output:**

```
=== Model Diagnostics ===

Scanning models directory: models/custom

Found 2 models:
  - jarvis (created: 2026-01-06)
  - phoenix (created: 2026-01-05)

Testing model: jarvis
  ✓ Model file exists
  ✓ Metadata valid
  ✓ Model loads successfully
  ✓ Inference works
  ✓ ONNX export available

All models healthy!
```

### analyze_model.py

Analyzes model architecture and parameters.

**Usage:**

```bash
python scripts/analyze_model.py --model jarvis
```

**Output:**

```
=== Model Analysis: jarvis ===

Architecture:
  - Base: MIT/ast-finetuned-speech-commands-v2
  - Classifier: WakeWordClassifier
  - Attention: Enabled
  - SE Block: Enabled
  - TCN: Enabled

Parameters:
  - Total: 87,234,567
  - Trainable: 234,567
  - Frozen: 87,000,000

Layers:
  - classifier.attention: 2,359,296 params
  - classifier.se_block: 147,712 params
  - classifier.tcn: 1,180,672 params
  - classifier.fc1: 196,864 params
  - classifier.fc2: 32,896 params
  - classifier.fc3: 258 params
```

---

## Audio Utilities

### convert_mp3_to_wav.py

Converts audio files to WAV format.

**Usage:**

```bash
python scripts/convert_mp3_to_wav.py --input audio.mp3 --output audio.wav
```

**Options:**

| Option | Description |
|--------|-------------|
| `--input` | Input audio file or directory |
| `--output` | Output file or directory |
| `--sample-rate` | Target sample rate (default: 16000) |
| `--mono` | Convert to mono |

### prepare_noise.py

Prepares noise samples for data augmentation.

**Usage:**

```bash
python scripts/prepare_noise.py --input noise_folder/
```

**Features:**

- Normalizes audio levels
- Removes silence
- Splits into chunks
- Validates audio quality

---

## TTS Preview

### preview_edge_tts.py

Tests Edge TTS voice synthesis.

**Usage:**

```bash
python scripts/preview_edge_tts.py --text "hello world" --voice en-US-JennyNeural
```

**Options:**

| Option | Description |
|--------|-------------|
| `--text` | Text to synthesize |
| `--voice` | Voice ID |
| `--output` | Output WAV file |
| `--list` | List available voices |

### preview_kokoro_tts.py

Tests Kokoro TTS voice synthesis.

**Usage:**

```bash
python scripts/preview_kokoro_tts.py --text "hello world" --voice af_heart
```

**Options:**

| Option | Description |
|--------|-------------|
| `--text` | Text to synthesize |
| `--voice` | Voice ID |
| `--output` | Output WAV file |
| `--speed` | Speed multiplier |

---

## Main Scripts

### run.py

Main setup and run script.

**Usage:**

```bash
python run.py [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--auto`, `-y` | Automatic mode (no prompts) |
| `--check` | Verify environment only |
| `--install` | Install dependencies only |
| `--download` | Download TTS models only |
| `--run` | Run server only |
| `--cuda VERSION` | Use specific CUDA version |
| `--port PORT` | Server port |

### clean.py

Cleanup script for removing data and caches.

**Usage:**

```bash
python clean.py [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--all` | Remove everything |
| `--venv` | Remove virtual environment |
| `--voices` | Remove TTS voices |
| `--data` | Remove training data |
| `--models` | Remove trained models |
| `--cache` | Remove cache directories |
| `--temp` | Remove temporary files |
| `--status` | Show disk usage |

---

## Creating Custom Scripts

### Template

```python
#!/usr/bin/env python3
"""
Custom Script Description

Usage:
    python scripts/my_script.py [options]
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="My custom script")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()
    
    # Script logic here
    print("Running custom script...")

if __name__ == "__main__":
    main()
```

### Best Practices

1. Add docstrings with usage examples
2. Use argparse for command-line options
3. Handle errors gracefully
4. Log progress for long operations
5. Support `--help` flag
