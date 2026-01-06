# Creating Your First Model

A complete walkthrough of creating your first custom wake word model.

---

## Before You Begin

Ensure you have:

- [x] WakeBuilder installed and running at `http://localhost:8000`
- [x] Negative data downloaded (check home page status)
- [x] Negative cache built (optional but recommended)
- [x] A working microphone

---

## Step 1: Choose Your Wake Word

### Navigate to Training

1. Open WakeBuilder in your browser: `http://localhost:8000`
2. Click **"Train New Model"** on the home page

### Enter Wake Word

Enter your desired wake word in the input field:

!!! example "Good Wake Word Examples"
    - `jarvis`
    - `phoenix`
    - `hey computer`
    - `listen up`
    - `activate`

### Wake Word Guidelines

| Rule | Description |
|------|-------------|
| **Length** | 4-12 characters |
| **Words** | 1-2 words maximum |
| **Characters** | Letters and spaces only |
| **Uniqueness** | Avoid common words like "the" or "yes" |

!!! warning "Avoid These"
    - Single letters or very short words ("hi", "go")
    - Very common words ("computer", "hello")
    - Words that sound like other common words
    - Long phrases (more than 2 words)

---

## Step 2: Configure Training (Optional)

Click **"Advanced Options"** to customize training parameters:

### Data Generation Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Target Positive Samples** | 5,000 | Total positive training samples |
| **Max Negative Chunks** | 0 (auto) | From cached negative data |
| **Negative Ratio** | 2.0x | Real negatives per positive |
| **Hard Negative Ratio** | 4.0x | Similar-sounding word negatives |

### Training Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| **Batch Size** | 32 | Samples per training step |
| **Max Epochs** | 100 | Maximum training iterations |
| **Early Stopping** | 8 epochs | Patience for convergence |
| **Learning Rate** | 0.0001 | Step size for optimizer |
| **Dropout** | 0.5 | Regularization strength |
| **Label Smoothing** | 0.1 | Confidence calibration |

### Model Enhancements

| Option | Default | Purpose |
|--------|---------|---------|
| **Focal Loss** | ✓ Enabled | Better hard example handling |
| **Self-Attention** | ✓ Enabled | Discriminative feature focus |
| **SE Block** | ✓ Enabled | Channel attention |
| **TCN Block** | ✓ Enabled | Local pattern capture |

!!! tip "For Your First Model"
    Leave all settings at their defaults. They've been tuned for optimal results.

Click **"Continue to Recording"** when ready.

---

## Step 3: Record Your Voice

### Recording Interface

You'll see:

- A **waveform visualizer** showing audio levels
- A **record button** in the center
- A **recordings list** below (initially empty)
- A **recording counter** showing progress

### How to Record

1. **Prepare**: Find a quiet environment with minimal background noise
2. **Position**: Hold your device at normal speaking distance (arm's length)
3. **Record**: Click and **hold** the record button
4. **Speak**: Say your wake word clearly at normal volume
5. **Release**: Let go of the button to stop recording

### Recording Tips

!!! success "Do This"
    - Speak at your natural pace and volume
    - Keep recordings 0.5-2 seconds long
    - Record in the environment where you'll use the model
    - Vary your tone slightly between recordings

!!! failure "Avoid This"
    - Whispering or shouting
    - Very long pauses before or after the word
    - Background noise (TV, music, conversations)
    - Speaking too fast or too slow

### Review and Re-record

After each recording:

1. **Listen**: Click the play button to hear your recording
2. **Evaluate**: Is it clear? At normal volume?
3. **Delete**: Click ✕ to remove a bad recording
4. **Re-record**: Record again if needed

### How Many Recordings?

| Recordings | Result |
|------------|--------|
| 1 | Minimum - works with TTS augmentation |
| 3 | Recommended - good variety |
| 5 | Maximum - best accuracy |

Click **"Start Training"** when you have at least 1 recording.

---

## Step 4: Monitor Training

### Training Phases

Training progresses through several phases:

1. **Preparing data** - Validation and processing
2. **Generating TTS samples** - Creating voice variations
3. **Generating hard negatives** - Similar-sounding words
4. **Loading negative cache** - Background audio samples
5. **Training classifier** - Neural network training
6. **Evaluating model** - Performance assessment

### During Training

You'll see:

- **Progress bar** with percentage complete
- **Current phase** description
- **Training metrics** (loss, accuracy, F1)
- **Live charts** showing training progress
- **Elapsed time** counter

### Training Metrics Explained

| Metric | What It Means |
|--------|--------------|
| **Train Loss** | Error on training data (lower = better) |
| **Val Loss** | Error on validation data (lower = better) |
| **Val Accuracy** | Correct predictions (higher = better) |
| **Val F1** | Balance of precision/recall (higher = better) |

!!! info "Normal Training Behavior"
    - Loss decreases rapidly in early epochs
    - Loss stabilizes after 20-40 epochs
    - Early stopping may end training before max epochs

---

## Step 5: Review Results

### Performance Summary

After training, you'll see:

| Metric | Good Value | Description |
|--------|------------|-------------|
| **Accuracy** | >95% | Overall correctness |
| **F1 Score** | >0.90 | Balanced performance |
| **Threshold** | 0.5-0.7 | Recommended detection sensitivity |

### Threshold Analysis Chart

The FAR/FRR chart shows:

- **FAR (False Accept Rate)**: Wrong triggers on non-wake-words
- **FRR (False Reject Rate)**: Missed detections of wake word

The recommended threshold balances these competing metrics.

### Training Data Summary

Review what was used for training:

- Number of recordings provided
- Total positive samples generated
- Total negative samples used
- Training/validation split

---

## Step 6: Test Your Model

### Immediate Testing

1. Click **"Test Model"** on the results page
2. Grant microphone permission if prompted
3. Start speaking your wake word
4. Watch the detection indicator

### Detection Indicator

- **Green pulse**: Wake word detected
- **No response**: Not detected or below threshold

### Adjusting Sensitivity

Use the **threshold slider** to adjust detection sensitivity:

| Threshold | Behavior |
|-----------|----------|
| Lower (0.3) | More sensitive, may have false triggers |
| Higher (0.8) | Less sensitive, may miss some detections |

---

## Step 7: Download and Deploy

### Download Options

- **ZIP file**: Contains model (.pt) and metadata (.json)
- **ONNX format**: For deployment with WakeEngine

### What's Included

```
your_wake_word_model.zip
├── your_wake_word.pt        # PyTorch model weights
├── your_wake_word.json      # Metadata and threshold
└── your_wake_word.onnx      # ONNX export (if available)
```

---

## Congratulations! 🎉

You've created your first custom wake word model. Next steps:

<div class="grid cards" markdown>

- :material-test-tube:{ .lg .middle } **More Testing**

    ---

    Learn about advanced testing options.

    [:octicons-arrow-right-24: Testing Guide](testing-models.md)

- :material-tune:{ .lg .middle } **Optimization**

    ---

    Improve your model's accuracy.

    [:octicons-arrow-right-24: Best Practices](best-practices.md)

- :material-rocket-launch:{ .lg .middle } **Deployment**

    ---

    Use your model in applications.

    [:octicons-arrow-right-24: Deployment Guide](../deployment/index.md)

</div>
