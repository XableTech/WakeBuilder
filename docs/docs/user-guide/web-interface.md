# Web Interface Guide

Complete walkthrough of the WakeBuilder web interface.

---

## Overview

WakeBuilder's web interface provides a user-friendly way to:

- View and manage wake word models
- Train new custom models
- Test models with live audio
- Monitor training progress

---

## Navigation

The navigation bar at the top provides access to all sections:

| Tab | Purpose |
|-----|---------|
| **Models** | Model dashboard (home page) |
| **Train** | Training wizard |
| **Test** | Model testing |
| **API Docs** | Interactive API documentation |

---

## Models Page (Home)

### Model Dashboard

The home page displays all your wake word models as cards:

- **Model name**: The wake word
- **Type badge**: "Custom" or "Default"
- **Metrics**: Accuracy and F1 score
- **Actions**: Test, Download, Delete

### Negative Data Panel

Shows the status of negative training data:

| Status | Meaning |
|--------|---------|
| **Missing** | Need to download UNAC dataset |
| **Available** | Dataset downloaded |

Click **"Download Dataset"** to get the UNAC corpus (~1.4GB).

### Cache Panel

Shows negative cache status:

| Status | Meaning |
|--------|---------|
| **Not built** | Cache needs to be created |
| **Ready** | Cache available (x chunks) |

Click **"Build Cache"** to preprocess negative audio (~5 min).

---

## Training Wizard

### Step 1: Wake Word

Enter your desired wake word:

1. **Wake word input**: Type your trigger phrase
2. **Model architecture**: Select AST (default)
3. **Advanced options**: Click to expand for configuration

#### Advanced Options

Collapsible sections for customization:

**Data Generation:**

- Target positive samples
- Negative ratio
- Hard negative ratio
- TTS, real negatives, hard negatives toggles

**Training Parameters:**

- Batch size
- Max epochs
- Early stopping patience
- Learning rate
- Dropout
- Label smoothing
- Mixup alpha

**Model Enhancements:**

- Focal loss settings
- Self-attention
- SE block
- TCN block
- Classifier layers

### Step 2: Recording

Record your voice saying the wake word:

1. **Click and hold** the record button
2. **Speak** your wake word
3. **Release** to stop
4. **Review** the recording (play, delete)
5. **Repeat** for 1-5 recordings

The waveform visualizer shows audio levels in real-time.

### Step 3: Training

Monitor training progress:

**Progress Section:**

- Progress bar with percentage
- Current phase description
- Elapsed time

**Training Data Panel:**

- Recordings count
- Positive samples
- Negative samples
- Train/validation split

**Hyperparameters Panel:**

- All configured settings displayed

**Metrics Panel:**

- Current epoch
- Train loss, Val loss
- Val accuracy, Val F1

**Charts:**

- Loss history graph
- Accuracy/F1 history graph

### Step 4: Results

View training results:

**Performance Summary:**

- Accuracy
- F1 Score
- Threshold
- Parameters count

**Training Data Summary:**

- Recordings provided
- Samples generated
- Epochs trained

**Configuration Summary:**

- All hyperparameters used

**Threshold Analysis Chart:**

- FAR/FRR curves
- Optimal threshold line

**Actions:**

- Test model
- Download model
- Train another

---

## Test Page

### Model Selection

Select a model to test from the dropdown menu.

### Real-time Testing

1. **Grant microphone access** when prompted
2. Click **"Start Listening"**
3. **Speak** your wake word
4. Watch the **detection indicator**

**Controls:**

- Start/Stop button
- Threshold slider
- Confidence display

### Detection Feedback

| Indicator | Meaning |
|-----------|---------|
| **Green pulse** | Wake word detected! |
| **Confidence bar** | Current detection confidence |
| **Threshold line** | Detection cutoff |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Hold to record (on recording page) |
| `Escape` | Cancel current action |
| `Enter` | Submit/continue |

---

## Mobile Support

The interface is responsive and works on mobile devices:

- Touch-friendly buttons
- Responsive layout
- Mobile recording support (where browser supports)

---

## API Documentation

Click **"API Docs"** in the navigation to access:

### Swagger UI

Interactive API documentation at `/docs`:

- Try endpoints directly
- View request/response schemas
- Authentication information

### ReDoc

Alternative documentation at `/redoc`:

- Clean, readable format
- Grouped by tags
- Search functionality

---

## Troubleshooting Interface Issues

??? question "Page not loading"

    1. Check the server is running: `python run.py`
    2. Verify the URL: `http://localhost:8000`
    3. Check browser console for errors (F12)

??? question "Recording not working"

    1. Grant microphone permission in browser
    2. Check microphone is not muted
    3. Try a different browser (Chrome recommended)

??? question "Training stuck"

    1. Check server logs for errors
    2. Refresh the page and check job status
    3. Training jobs continue in background

??? question "Charts not displaying"

    1. Ensure JavaScript is enabled
    2. Clear browser cache
    3. Try a different browser
