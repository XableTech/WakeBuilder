# Data Augmentation Settings

Configure TTS and audio augmentation for training data generation.

---

## Overview

Augmentation settings control how training data is generated and varied.

---

## TTS Settings

### Enabling TTS Providers

| Setting | Default | Description |
|---------|---------|-------------|
| `use_tts_positives` | true | Use TTS for positive samples |
| `use_hard_negatives` | true | Generate phonetically similar words |

### Voice Selection

Voices are selected automatically based on:

- Language match
- Quality tier (prefer medium/high)
- Diversity (different genders, accents)

---

## Sample Generation

### Target Counts

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `target_positive_samples` | 5,000 | 3,000-8,000 | Total positive samples |
| `negative_ratio` | 2.0 | 1.0-3.0 | Real negatives per positive |
| `hard_negative_ratio` | 4.0 | 2.0-6.0 | Hard negatives per positive |

### Sample Distribution

With defaults (5,000 positives):

| Type | Samples | Formula |
|------|---------|---------|
| Positives | 5,000 | target_positive_samples |
| Real negatives | 10,000 | positives × negative_ratio |
| Hard negatives | 20,000 | positives × hard_negative_ratio |
| **Total** | **35,000** | |

---

## Speed Augmentation

| Setting | Default | Description |
|---------|---------|-------------|
| `speed_normal_range` | [0.9, 1.1] | Normal variation |
| `speed_fast_range` | [1.3, 1.5] | Fast speech |

### Speed Distribution

| Speed | Applied To | Purpose |
|-------|------------|---------|
| 0.9x | 25% of samples | Slow speakers |
| 1.0x | 25% of samples | Normal |
| 1.1x | 25% of samples | Fast speakers |
| 1.5x | 25% of samples | Very fast |

---

## Pitch Augmentation

| Setting | Default | Description |
|---------|---------|-------------|
| `pitch_range` | [-2, 2] | Semitones shift |

### Pitch Values

| Shift | Purpose |
|-------|---------|
| -2 semitones | Lower voices |
| -1 semitone | Slightly lower |
| 0 | Original |
| +1 semitone | Slightly higher |
| +2 semitones | Higher voices |

---

## Volume Augmentation

| Setting | Default | Description |
|---------|---------|-------------|
| `volume_min` | 0.7 | Quiet speech |
| `volume_max` | 1.3 | Loud speech |

### Volume Levels

| Gain (dB) | Applied To |
|-----------|------------|
| -6 dB | 25% |
| -3 dB | 25% |
| 0 dB | 25% |
| +3 dB | 25% |

---

## Noise Augmentation

| Setting | Default | Description |
|---------|---------|-------------|
| `noise_snr_min` | -20 dB | Heavy noise |
| `noise_snr_max` | -5 dB | Light noise |
| `noise_probability` | 0.5 | % samples with noise |

### Noise Types

| Type | Description |
|------|-------------|
| Background | Office, cafe, street |
| Music | Various genres |
| Crowd | Multiple speakers |
| Environmental | Nature, weather |

---

## Configuration Example

```python
AUGMENTATION_CONFIG = {
    # TTS
    "use_tts_positives": True,
    "use_hard_negatives": True,
    
    # Sample counts
    "target_positive_samples": 5000,
    "negative_ratio": 2.0,
    "hard_negative_ratio": 4.0,
    
    # Speed
    "speed_normal_range": [0.9, 1.1],
    "speed_fast_range": [1.3, 1.5],
    
    # Pitch
    "pitch_range": [-2, 2],
    
    # Volume
    "volume_range": [0.7, 1.3],
    
    # Noise
    "noise_snr_range": [-20, -5],
    "noise_probability": 0.5,
}
```

---

## Trade-offs

### More Augmentation

**Pros**:

- Better generalization
- More robust model
- Handles variations better

**Cons**:

- Longer training time
- More memory usage
- Diminishing returns

### Less Augmentation

**Pros**:

- Faster training
- Lower memory usage
- Simpler data

**Cons**:

- May overfit
- Less robust
- Narrower detection range

---

## Recommendations

### Standard Use

Use defaults. They balance quality and training time.

### Limited Hardware

```python
{
    "target_positive_samples": 3000,
    "negative_ratio": 1.5,
    "hard_negative_ratio": 2.0,
}
```

### Maximum Quality

```python
{
    "target_positive_samples": 8000,
    "negative_ratio": 3.0,
    "hard_negative_ratio": 6.0,
}
```
