# Testing Models

How to test your trained wake word models with real-time audio.

---

## Overview

WakeBuilder provides **real-time testing** through your microphone. Speak your wake word and see instant detection feedback.

---

## Real-time Testing

### Accessing the Test Page

1. Navigate to **Test** in the navigation bar
2. Or click **"Test Model"** on a model card
3. Or click **"Test"** after training completes

### Testing Interface

The real-time testing interface shows:

- **Model selector**: Choose which model to test
- **Detection indicator**: Visual feedback when wake word detected
- **Confidence meter**: Current detection confidence level
- **Threshold slider**: Adjust detection sensitivity
- **Audio visualizer**: Real-time waveform display

### How to Test

1. **Select model**: Choose from the dropdown
2. **Grant microphone access**: Click "Allow" when prompted
3. **Start listening**: Click the "Start" button
4. **Speak**: Say your wake word naturally
5. **Observe**: Watch for the detection indicator

### Detection Feedback

| Indicator | Meaning |
|-----------|---------|
| **Green pulse** | Wake word detected |
| **Confidence bar fills** | Approaching threshold |
| **No response** | Not detected |

### Adjusting Sensitivity

Use the **threshold slider** to fine-tune detection:

| Threshold | Behavior |
|-----------|----------|
| **0.3** | Very sensitive - more triggers, possible false positives |
| **0.5** | Balanced - good for most use cases |
| **0.65** | Default - recommended threshold |
| **0.8** | Conservative - fewer false positives, may miss some |

!!! tip "Finding Your Threshold"
    Start at the default (0.65) and adjust based on your experience:

    - Too many false triggers? Increase threshold
    - Missing detections? Decrease threshold

---

## WebSocket Protocol

### API Endpoint

Real-time testing uses WebSocket:

```
ws://localhost:8000/api/test/ws/{model_id}
```

### Protocol

=== "Client → Server"

    ```json
    {
      "type": "audio",
      "data": "<base64 encoded audio>",
      "sample_rate": 16000
    }
    ```

=== "Server → Client"

    ```json
    {
      "type": "detection",
      "detected": true,
      "confidence": 0.87,
      "threshold": 0.65,
      "model_id": "jarvis",
      "timestamp": "2026-01-06T15:30:00Z"
    }
    ```

### Audio Format Requirements

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Channels | Mono (1) |
| Format | PCM 16-bit or Float32 |
| Chunk size | ~1 second |

---

## Understanding Results

### Confidence Score

The confidence score (0-1) indicates how certain the model is:

| Score | Interpretation |
|-------|----------------|
| **0.9+** | Very confident detection |
| **0.7-0.9** | Likely match |
| **0.5-0.7** | Uncertain |
| **0.3-0.5** | Probably not the wake word |
| **<0.3** | Definitely not the wake word |

### Why Confidence Varies

Normal factors affecting confidence:

- **Speaking style**: Casual vs. clear pronunciation
- **Distance**: Close vs. far from microphone
- **Background noise**: Quiet vs. noisy environment
- **Speaker**: Your voice vs. synthesized voices

---

## Testing Best Practices

### For Accurate Testing

!!! success "Do This"

    - Test in your target environment (where you'll actually use it)
    - Test with natural speech, not exaggerated pronunciation
    - Test with background noise if your environment has it
    - Test with different distances from the microphone

!!! failure "Avoid This"

    - Testing only in perfect silence
    - Over-enunciating the wake word
    - Testing at the same volume every time
    - Ignoring edge cases

### Test Scenarios

| Scenario | Purpose |
|----------|---------|
| **Normal speech** | Baseline performance |
| **Soft speech** | Lower volume detection |
| **With music** | Background noise robustness |
| **From distance** | Microphone sensitivity |
| **Similar words** | False positive rate |
| **Rapid repetition** | Continuous detection |

---

## Troubleshooting

??? question "Model not detecting my wake word"

    1. **Lower the threshold** to 0.4-0.5
    2. **Speak closer** to the microphone
    3. **Check audio levels** aren't too low
    4. If still failing, retrain with more recordings

??? question "Too many false positives"

    1. **Raise the threshold** to 0.75-0.85
    2. **Retrain** with higher hard negative ratio
    3. **Avoid** common words as wake words

??? question "Detection is laggy"

    1. Check **CPU/GPU usage** - may be under load
    2. **Close other applications** using the microphone
    3. On GPU, ensure **CUDA is enabled**

??? question "Microphone not working"

    1. Check **browser permissions** for microphone access
    2. Ensure microphone is **not muted**
    3. Try a different browser (Chrome recommended)
    4. Check **system audio settings**
