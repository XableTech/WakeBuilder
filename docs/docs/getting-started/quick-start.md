# Quick Start

Get WakeBuilder installed and start training your first wake word model.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.12+** installed on your system
- **Microphone** for recording wake word samples
- **8GB+ RAM** available (16GB recommended)
- **NVIDIA GPU with CUDA** strongly recommended for reasonable training times

!!! warning "GPU Required for Practical Use"
    Training on CPU can take **many hours**. A CUDA-compatible GPU is strongly recommended and still requires significant time (30 minutes to a few hours depending on hardware).

---

## Installation

### Using the Setup Script (Recommended)

The easiest way to get started is with the interactive setup script:

=== "Windows"

    ```powershell
    # Clone the repository
    git clone https://github.com/XableTech/WakeBuilder.git
    cd WakeBuilder

    # Run the interactive setup
    python run.py
    ```

=== "Linux/macOS"

    ```bash
    # Clone the repository
    git clone https://github.com/XableTech/WakeBuilder.git
    cd WakeBuilder

    # Run the interactive setup
    python run.py
    ```

The `run.py` script will automatically:

1. ✅ Check your Python version
2. ✅ Detect your package manager (uv or pip)
3. ✅ Create a virtual environment
4. ✅ Install dependencies with CUDA support (if available)
5. ✅ Download TTS voice models (~5GB)
6. ✅ Launch the web interface

!!! tip "Automatic Mode"
    Use `python run.py --auto` or `python run.py -y` to skip all prompts and run everything automatically.

---

## Accessing the Web Interface

Once the server starts, open your browser and navigate to:

```
http://localhost:8000
```

You'll see the WakeBuilder home page with the model dashboard.

---

## Training Your First Model

### Step 1: Start a New Training Job

1. Click **"Train New Model"** on the home page
2. Enter your wake word (e.g., "jarvis" or "hey computer")
3. Review the default settings (they work well for most cases)
4. Click **"Continue to Recording"**

### Step 2: Record Your Voice

1. Click and **hold** the record button
2. Say your wake word clearly in your natural voice
3. Release to stop recording
4. Record at least 1 sample (3-5 recommended for best results)
5. Click **"Start Training"**

!!! info "For Users with Speech Differences"
    If you have unique speech patterns due to accent, health conditions, or other factors, your recordings are what the model will learn from. Record naturally—the model will specifically learn **your** voice!

### Step 3: Monitor Training Progress

Training time depends on your hardware. You'll see:

- Real-time progress bar
- Current training phase
- Live loss and accuracy charts
- Training metrics (loss, accuracy, F1 score)

| Hardware | Expected Time |
|----------|---------------|
| High-end GPU | 30 min - 1.5 hours |
| Mid-range GPU | 1 - 2 hours |
| Entry GPU | 2 - 4 hours |
| CPU only | 4+ hours |

### Step 4: Test Your Model

After training completes:

1. View the results summary
2. Click **"Test Model"** to try it live
3. Speak your wake word into the microphone
4. Watch for the detection indicator

---

## What's Next?

<div class="grid cards" markdown>

- :material-book-open-variant:{ .lg .middle } **Learn More**

    ---

    Understand the full training workflow and optimization tips.

    [:octicons-arrow-right-24: Training Workflow](../user-guide/training-workflow.md)

- :material-cog:{ .lg .middle } **Configure**

    ---

    Customize training parameters for better results.

    [:octicons-arrow-right-24: Configuration Guide](../configuration/index.md)

- :material-docker:{ .lg .middle } **Deploy**

    ---

    Run WakeBuilder in Docker for consistent deployment.

    [:octicons-arrow-right-24: Docker Guide](../deployment/docker.md)

</div>

---

## Troubleshooting

### Common Issues

??? question "The server won't start"

    Ensure no other application is using port 8000. Try:
    ```bash
    python run.py --port 8001
    ```

??? question "Training is extremely slow"

    Training on CPU is very slow by design—the AST model is computationally intensive.
    
    **Solutions:**
    
    - Use an NVIDIA GPU with CUDA support
    - Check that CUDA is properly detected: `python -c "import torch; print(torch.cuda.is_available())"`
    - Reduce the target positive samples in Advanced Options

??? question "Model not detecting my wake word"

    Try:
    
    - Record clearer samples with less background noise
    - Lower the detection threshold in testing
    - Increase the number of recordings (up to 5)
    - Increase the hard negative ratio in training options

For more help, see the [full troubleshooting guide](../deployment/local-development.md#troubleshooting).
