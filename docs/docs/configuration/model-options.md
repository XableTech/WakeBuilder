# Model Architecture Options

Configure the classifier architecture and enhancements.

---

## Overview

WakeBuilder's model consists of:

1. **Frozen AST base** - Extracts audio embeddings (not configurable)
2. **Trainable classifier** - Detects wake word (configurable)

---

## Classifier Architecture

### Hidden Dimensions

**Default**: `[256, 128]`

Sizes of fully-connected layers:

| Configuration | Effect |
|---------------|--------|
| `[128]` | Minimal, fastest |
| `[256, 128]` | Default, balanced |
| `[512, 256, 128]` | Larger, more capacity |

More layers = more parameters = slower training.

---

### Dropout Rate

**Default**: 0.5

Applied between layers:

| Rate | Effect |
|------|--------|
| 0.3 | Light regularization |
| 0.5 | Strong regularization |
| 0.7 | Very strong |

Higher = more regularization, prevents overfitting.

---

## Enhancement Modules

### Self-Attention Pooling

**Default**: Enabled

Learns to weight important time frames:

| Setting | Effect |
|---------|--------|
| Enabled | Better temporal focus |
| Disabled | Simple mean pooling |

**Parameters**: ~2.4M

**Recommendation**: Keep enabled for best accuracy.

---

### Squeeze-and-Excitation (SE) Block

**Default**: Enabled

Channel attention mechanism:

| Setting | Effect |
|---------|--------|
| Enabled | Better channel weighting |
| Disabled | No channel attention |

**Parameters**: ~150K

**Recommendation**: Keep enabled.

---

### Temporal Convolutional Network (TCN)

**Default**: Enabled

Captures temporal patterns:

| Setting | Effect |
|---------|--------|
| Enabled | Better temporal modeling |
| Disabled | Simpler temporal processing |

**Parameters**: ~1.2M

**Recommendation**: Keep enabled for complex wake words.

---

## Parameter Count

### With All Enhancements (Default)

| Component | Parameters |
|-----------|------------|
| Self-Attention | 2,359,296 |
| SE Block | 147,712 |
| TCN Block | 1,180,672 |
| FC Layers | 229,890 |
| **Total Trainable** | **~3.9M** |

### Minimal Configuration

Without enhancements:

| Component | Parameters |
|-----------|------------|
| FC Layers | 229,890 |
| **Total Trainable** | **~230K** |

---

## Configuration in Web UI

The Advanced Options panel includes toggles for:

- ✅ Self-Attention Pooling
- ✅ SE Block
- ✅ TCN Block

And fields for:

- Classifier Layers: `256, 128`

---

## Configuration via API

```json
{
  "use_attention": true,
  "use_se_block": true,
  "use_tcn": true,
  "classifier_dims": "256,128"
}
```

---

## Presets

### Minimal (Fast)

```python
{
    "classifier_dims": [128],
    "use_attention": false,
    "use_se_block": false,
    "use_tcn": false,
}
```

**Use when**: Speed is critical, simple wake words.

### Balanced (Default)

```python
{
    "classifier_dims": [256, 128],
    "use_attention": true,
    "use_se_block": true,
    "use_tcn": true,
}
```

**Use when**: Best accuracy/speed trade-off.

### Maximum Capacity

```python
{
    "classifier_dims": [512, 256, 128],
    "use_attention": true,
    "use_se_block": true,
    "use_tcn": true,
}
```

**Use when**: Complex wake words, ample training data.

---

## Trade-offs

### More Capacity

**Pros**:

- Can learn complex patterns
- Higher potential accuracy
- Better discrimination

**Cons**:

- Slower training
- Larger model size
- Risk of overfitting

### Less Capacity

**Pros**:

- Faster training
- Smaller model
- Less likely to overfit

**Cons**:

- May miss subtle patterns
- Lower maximum accuracy
- Less robust

---

## Recommendations

### For Most Users

Use the default configuration. It balances accuracy and performance.

### For Simple Wake Words

Disable TCN if wake word is short (1-2 syllables).

### For Complex Wake Words

Keep all enhancements enabled. Consider larger classifier dims.

### For Resource-Constrained Deployment

Use minimal configuration and quantized ONNX export.
