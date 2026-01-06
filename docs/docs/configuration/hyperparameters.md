# Training Hyperparameters

Configure training parameters for optimal model performance.

---

## Overview

Hyperparameters control how the model learns. The defaults work well for most cases, but tuning can improve results.

---

## Core Parameters

### Batch Size

**Default**: 32

Number of samples processed together:

| Batch Size | Effect |
|------------|--------|
| 8-16 | More stable, slower, higher memory |
| 32 | Recommended balance |
| 64-128 | Faster, may require LR adjustment |

**Adjustment tips**:

- Reduce if out of memory
- Increase for larger datasets
- Adjust learning rate proportionally

---

### Max Epochs

**Default**: 100

Maximum training iterations:

| Epochs | Use Case |
|--------|----------|
| 30-50 | Quick experiments |
| 100 | Default, most cases |
| 150-200 | Maximum accuracy |

Note: Training usually stops earlier due to early stopping.

---

### Early Stopping Patience

**Default**: 8

Epochs without improvement before stopping:

| Patience | Behavior |
|----------|----------|
| 3-5 | Aggressive, fast training |
| 8 | Balanced |
| 12-15 | Thorough, slower |

---

## Optimization Parameters

### Learning Rate

**Default**: 0.0001

Step size for weight updates:

| Learning Rate | Effect |
|---------------|--------|
| 0.00001 | Very slow, stable |
| 0.0001 | Recommended |
| 0.001 | Faster, may overshoot |

Uses OneCycleLR scheduler with warmup.

---

### Dropout

**Default**: 0.5

Regularization to prevent overfitting:

| Dropout | Effect |
|---------|--------|
| 0.2-0.3 | Light regularization |
| 0.5 | Strong regularization (default) |
| 0.6-0.7 | Very strong, may underfit |

Increase if validation loss >> training loss.

---

### Weight Decay

**Default**: 0.01

L2 regularization on weights:

| Weight Decay | Effect |
|--------------|--------|
| 0.001 | Light |
| 0.01 | Moderate |
| 0.1 | Strong |

---

## Loss Function

### Focal Loss (Default)

Better for imbalanced datasets:

| Parameter | Default | Effect |
|-----------|---------|--------|
| **alpha** | 0.5 | Class weight balance |
| **gamma** | 2.0 | Focus on hard examples |

### Alpha Guidelines

| Alpha | Use When |
|-------|----------|
| 0.25 | More negatives than default |
| 0.5 | Balanced (default) |
| 0.75 | More positives than default |

### Gamma Guidelines

| Gamma | Effect |
|-------|--------|
| 0 | Standard cross-entropy |
| 1.0 | Mild focusing |
| 2.0 | Standard focal loss |
| 4.0 | Strong focusing |

---

## Augmentation During Training

### Label Smoothing

**Default**: 0.1

Softens target labels:

| Value | Effect |
|-------|--------|
| 0 | Hard targets (0 or 1) |
| 0.1 | Slight softening |
| 0.2 | Stronger smoothing |

Prevents overconfidence.

---

### Mixup Alpha

**Default**: 0.5

Mixes training examples:

| Alpha | Effect |
|-------|--------|
| 0 | No mixup |
| 0.2 | Light mixing |
| 0.5 | Moderate mixing |
| 1.0 | Heavy mixing |

Creates virtual training samples.

---

## Presets

### Fast Training

```python
{
    "batch_size": 64,
    "max_epochs": 50,
    "early_stopping_patience": 5,
    "learning_rate": 0.0002,
    "dropout": 0.4,
}
```

### Balanced (Default)

```python
{
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 8,
    "learning_rate": 0.0001,
    "dropout": 0.5,
}
```

### Maximum Accuracy

```python
{
    "batch_size": 32,
    "max_epochs": 200,
    "early_stopping_patience": 15,
    "learning_rate": 0.00005,
    "dropout": 0.5,
}
```

---

## Troubleshooting

??? question "Training loss not decreasing"

    - Increase learning rate
    - Check data quality
    - Reduce dropout temporarily

??? question "Validation loss increasing (overfitting)"

    - Increase dropout (0.5-0.6)
    - Increase label smoothing
    - Add more training data
    - Reduce epochs

??? question "Both losses decreasing too slowly"

    - Increase learning rate
    - Increase batch size
    - Check for data issues
