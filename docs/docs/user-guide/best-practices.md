# Tips for Best Results

Expert tips and best practices for creating high-quality wake word models.

---

## Recording Quality

### Environment

| Factor | Recommendation |
|--------|----------------|
| **Background noise** | Minimize - record in a quiet room |
| **Echo** | Reduce - avoid empty rooms with hard walls |
| **Distance** | Consistent - arm's length from microphone |
| **Device** | Same device you'll use for detection |

### Recording Technique

!!! success "Best Practices"

    - Speak at your natural pace and volume
    - Keep recordings 0.5-2 seconds (not too short, not too long)
    - Vary your tone slightly between recordings
    - Include natural variations (morning voice, tired voice)

!!! failure "Common Mistakes"

    - Speaking too slowly or deliberately
    - Whispering or shouting
    - Recording in noisy environments
    - Very different recording conditions than deployment

### Number of Recordings

| Recordings | Quality | Recommendation |
|------------|---------|----------------|
| 1 | Minimum | Works with TTS augmentation |
| 3 | Good | Recommended for most users |
| 5 | Better | Best accuracy |

---

## Wake Word Selection

### Ideal Characteristics

| Characteristic | Good Example | Poor Example |
|----------------|--------------|--------------|
| **Syllables** | 2-3 ("jarvis", "phoenix") | 1 ("hi", "go") |
| **Uniqueness** | Uncommon words | Common words ("computer") |
| **Pronunciation** | Easy, consistent | Hard to say consistently |
| **Phonetic distinction** | Unique sounds | Sounds like many words |

### Recommended Wake Words

```
✓ Good choices:
  - jarvis, phoenix, friday, alfred
  - hey buddy, wake up, listen up
  - Custom names: nimbus, orion, echo
  
✗ Avoid:
  - hi, hello, hey (too common)
  - supercalifragilistic (too long)
  - the, be, okay (appear in speech)
```

---

## Training Settings

### For Best Accuracy

=== "Balanced (Recommended)"

    | Setting | Value |
    |---------|-------|
    | Target Positives | 5,000 |
    | Negative Ratio | 2.0x |
    | Hard Negative Ratio | 4.0x |
    | Batch Size | 32 |
    | Max Epochs | 100 |
    | Early Stopping | 8 |
    | All Enhancements | Enabled |

=== "Maximum Accuracy"

    | Setting | Value |
    |---------|-------|
    | Target Positives | 8,000 |
    | Negative Ratio | 3.0x |
    | Hard Negative Ratio | 6.0x |
    | Batch Size | 32 |
    | Max Epochs | 200 |
    | Early Stopping | 15 |
    | All Enhancements | Enabled |

=== "Fast Training"

    | Setting | Value |
    |---------|-------|
    | Target Positives | 3,000 |
    | Negative Ratio | 1.5x |
    | Hard Negative Ratio | 2.0x |
    | Batch Size | 64 |
    | Max Epochs | 50 |
    | Early Stopping | 5 |
    | Enhancements | Optional |

### Hard Negatives

**Critical for reducing false positives:**

| Ratio | Trade-off |
|-------|-----------|
| 2.0x | Faster training, may have more false positives |
| 4.0x | Good balance (recommended) |
| 6.0x | Best accuracy, slower training |

---

## Reducing False Positives

### During Training

1. **Increase hard negative ratio** to 5.0x or 6.0x
2. **Enable all model enhancements** (attention, SE, TCN)
3. **Use more recordings** (5 if possible)
4. **Increase dropout** to 0.5-0.6

### After Training

1. **Raise the threshold** from 0.65 to 0.75-0.85
2. **Test with similar words** to identify triggers
3. **Retrain** if false positive rate is too high

### Common False Positive Causes

| Cause | Solution |
|-------|----------|
| Similar words trigger | More hard negatives |
| Partial utterances trigger | Add pure prefix negatives |
| Background speech triggers | More real negatives |
| Confident on everything | Higher dropout, label smoothing |

---

## Reducing False Rejections

### During Training

1. **Increase target positive samples** to 6,000-8,000
2. **Lower dropout** to 0.3-0.4
3. **Decrease label smoothing** to 0.05
4. **Increase focal alpha** to 0.6-0.7

### After Training

1. **Lower the threshold** from 0.65 to 0.5-0.6
2. **Test in your actual environment**
3. **Retrain** with more recordings if issues persist

---

## Model Enhancement Options

### When to Enable Each

| Enhancement | Enable When | Disable When |
|-------------|-------------|--------------|
| **Self-Attention** | Complex wake words, similar sounds | Fast training needed |
| **SE Block** | Fine discrimination needed | Memory constrained |
| **TCN Block** | Local patterns important | Simpler wake words |
| **Focal Loss** | Class imbalance, hard examples | Balanced data |

### Default Configuration

All enhancements are **enabled by default** for best results. Only disable if:

- Training is too slow
- Memory issues occur
- Simpler model is preferred

---

## Testing Thoroughly

### Before Deployment

| Test | Purpose |
|------|---------|
| **Normal speech** | Baseline detection |
| **Whisper** | Low volume |
| **Shout** | High volume |
| **Near/far** | Distance variation |
| **With noise** | Background robustness |
| **Similar words** | False positive check |
| **Different people** | Speaker variation |

### Threshold Calibration

1. Start at the recommended threshold (values in metadata)
2. Test with 20+ utterances
3. Count detections and misses
4. Adjust threshold:
   - Too many misses? Lower threshold
   - Too many false triggers? Raise threshold

---

## Troubleshooting

### Low Accuracy

| Issue | Solutions |
|-------|-----------|
| Accuracy < 90% | More recordings, more epochs |
| F1 < 0.85 | Higher hard negative ratio |
| High train/val gap | More dropout, label smoothing |

### High False Positive Rate

| Issue | Solutions |
|-------|-----------|
| FAR > 10% | More hard negatives (5.0x+) |
| Triggers on silence | Add noise samples |
| Triggers on music | Add music negatives |

### High False Rejection Rate

| Issue | Solutions |
|-------|-----------|
| FRR > 10% | Lower threshold |
| Doesn't detect user | More diverse recordings |
| Works for TTS but not user | Record in actual environment |
