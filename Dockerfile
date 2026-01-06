# WakeBuilder Dockerfile
# Multi-stage build for optimal image size with pre-downloaded TTS voices

# ============================================================================
# Stage 1: Base image with system dependencies
# ============================================================================
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Use faster Debian mirror (German mirror - fast for Europe)
RUN sed -i 's|http://deb.debian.org|http://ftp.de.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    libsndfile1 \
    ffmpeg \
    # eSpeak-NG for Coqui TTS phoneme conversion
    espeak-ng \
    libespeak-ng1 \
    # Build tools (needed for some Python packages)
    build-essential \
    # Git for huggingface downloads
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Install Python packages with uv
# ============================================================================
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv pip install --system (no venv needed)
# Increase timeout for large NVIDIA packages
ENV UV_HTTP_TIMEOUT=600
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system -r pyproject.toml

# ============================================================================
# Stage 3: Download TTS voices and models
# ============================================================================
FROM builder AS tts-downloader

# Set working directory
WORKDIR /app

# Copy source code for TTS download scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories for TTS voices
RUN mkdir -p /app/tts_voices

# Download Piper TTS voices (87 voices from piper_tts_voices.json)
# Diverse set of voices in multiple languages for data augmentation
RUN python scripts/download_voices.py || echo "Some Piper voices may have failed, continuing..."

# Download Coqui TTS models (10 models)
# Models: vctk, your_tts, tortoise, ljspeech, german, czech, slovak, slovenian, catalan, portuguese
# Download Kokoro TTS model and voice packs
RUN python scripts/preload_tts_models.py || echo "Some TTS models may have failed, continuing..."

# ============================================================================
# Stage 4: Final runtime image
# ============================================================================
FROM base AS runtime

# Install uv for runtime use
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy TTS voices and cached models from tts-downloader
COPY --from=tts-downloader /app/tts_voices /app/tts_voices
COPY --from=tts-downloader /root/.local/share/tts /root/.local/share/tts
COPY --from=tts-downloader /root/.cache/huggingface /root/.cache/huggingface

# Copy application source
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY main.py ./
COPY pyproject.toml ./

# Create data directories
RUN mkdir -p /app/data/negative /app/data/cache /app/data/temp /app/models /app/recordings

# Set environment variables for the application
ENV PYTHONPATH=/app/src \
    TTS_VOICES_DIR=/app/tts_voices \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command - run the FastAPI server
CMD ["uvicorn", "src.wakebuilder.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
