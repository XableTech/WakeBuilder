# Audio Settings

Configure audio processing parameters.

---

## Overview

Audio settings control how recordings are processed and augmented.

---

## Sample Rate

**Default**: 16,000 Hz

The AST model is designed for 16kHz audio. All input is resampled.

```python
# In config.py
SAMPLE_RATE = 16000
```

??? warning "Do Not Change"
    Changing the sample rate is not recommended as it requires
    a different pre-trained model.

---

## Duration

**Default**: 1.0 second (16,000 samples)

Audio clips are padded or trimmed to this duration.

| Input Duration | Action |
|----------------|--------|
| < 0.5s | Rejected as too short |
| 0.5s - 1.0s | Zero-padded to 1.0s |
| 1.0s - 2.0s | Trimmed or used as-is |
| > 2.0s | Trimmed to 1.0s |

---

## Mel Spectrogram

Parameters for mel spectrogram extraction:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_mels` | 128 | Number of mel frequency bins |
| `n_fft` | 400 | FFT window size |
| `hop_length` | 160 | Samples between frames |
| `fmin` | 0 | Minimum frequency |
| `fmax` | 8000 | Maximum frequency |

---

## Recording Constraints

### Web UI Settings

| Parameter | Min | Max | Default |
|-----------|-----|-----|---------|
| Recording duration | 0.5s | 3.0s | 1.5s |
| Recordings required | 1 | 5 | 3 |
| Auto-stop on silence | 0.5s | 2.0s | 0.8s |

### Audio Validation

Recordings are validated for:

| Check | Requirement |
|-------|-------------|
| Energy | Above silence threshold |
| Duration | Within allowed range |
| Clipping | No sustained clipping |
| Format | Web Audio compatible |

---

## Noise Settings

Background noise configuration:

| Parameter | Default | Range |
|-----------|---------|-------|
| `noise_snr_min` | -20 dB | -30 to -10 |
| `noise_snr_max` | -5 dB | -10 to 0 |

Lower SNR = more noise relative to signal.

---

## Voice Activity Detection

Simple energy-based VAD:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad_threshold` | 0.02 | RMS energy threshold |
| `vad_min_frames` | 10 | Minimum speech frames |
| `vad_frame_size` | 400 | Samples per frame |

---

## Configuration Example

```python
# src/wakebuilder/config.py

AUDIO_CONFIG = {
    # Core settings
    "sample_rate": 16000,
    "duration": 1.0,
    
    # Mel spectrogram
    "n_mels": 128,
    "n_fft": 400,
    "hop_length": 160,
    "fmin": 0,
    "fmax": 8000,
    
    # Recording constraints
    "min_recording_duration": 0.5,
    "max_recording_duration": 3.0,
    
    # Validation
    "energy_threshold": 0.02,
    "clipping_threshold": 0.99,
}
```
