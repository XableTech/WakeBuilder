# Docker Deployment

Deploy WakeBuilder using Docker for consistent, portable environments.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/XableTech/WakeBuilder.git
cd WakeBuilder

# Build and run
docker-compose up
```

Access WakeBuilder at `http://localhost:8000`.

---

## Prerequisites

- **Docker** 20.10 or later
- **Docker Compose** 2.0 or later
- **NVIDIA Container Toolkit** (for GPU support)

### Installing Docker

=== "Ubuntu/Debian"

    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    ```

=== "Windows"

    Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

=== "macOS"

    Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## Docker Compose

### Basic Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  wakebuilder:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./tts_voices:/app/tts_voices
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### With GPU Support

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  wakebuilder:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./tts_voices:/app/tts_voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Run with GPU:

```bash
docker-compose -f docker-compose.gpu.yml up
```

---

## Dockerfile

### Standard Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/
COPY frontend/ frontend/
COPY scripts/ scripts/

# Create data directories
RUN mkdir -p models/default models/custom data/negative data/cache tts_voices recordings

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.wakebuilder.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Building the Image

```bash
docker build -t wakebuilder:latest .
```

---

## Volume Mounts

### Persistent Data

Mount these directories to persist data:

| Container Path | Purpose | Required |
|----------------|---------|----------|
| `/app/models` | Trained models | Yes |
| `/app/data` | Training data, cache | Yes |
| `/app/tts_voices` | Piper TTS voices | Yes |
| `/app/recordings` | Temporary recordings | Optional |

### Example

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/tts_voices:/app/tts_voices \
  wakebuilder:latest
```

---

## GPU Support

### NVIDIA Container Toolkit

Install the NVIDIA Container Toolkit for GPU access:

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Running with GPU

```bash
docker run --gpus all -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/tts_voices:/app/tts_voices \
  wakebuilder:latest
```

---

## First-Time Setup

### 1. Download TTS Voices

Run inside container:

```bash
docker exec -it wakebuilder python scripts/download_voices.py
```

Or mount pre-downloaded voices:

```bash
# Download locally first
python scripts/download_voices.py

# Mount when running
docker run -v $(pwd)/tts_voices:/app/tts_voices ...
```

### 2. Download Negative Data

Through the web interface:

1. Navigate to `http://localhost:8000`
2. Click "Download Dataset" in the Negative Data panel

Or manually:

```bash
# Download UNAC dataset
# Mount to /app/data/negative
```

### 3. Build Cache

Through web interface or command:

```bash
docker exec -it wakebuilder python scripts/build_negative_cache.py
```

---

## Environment Variables

Configure the container with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WAKEBUILDER_HOST` | `0.0.0.0` | Server bind address |
| `WAKEBUILDER_PORT` | `8000` | Server port |
| `CUDA_VISIBLE_DEVICES` | all | GPU selection |
| `LOG_LEVEL` | `info` | Logging verbosity |

Example:

```bash
docker run -d \
  -e WAKEBUILDER_PORT=8080 \
  -e LOG_LEVEL=debug \
  -p 8080:8080 \
  wakebuilder:latest
```

---

## Health Checks

The Docker image includes health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' wakebuilder
```

---

## Logs

View container logs:

```bash
# Follow logs
docker logs -f wakebuilder

# Last 100 lines
docker logs --tail 100 wakebuilder
```

---

## Troubleshooting

??? question "Container won't start"

    1. Check Docker is running:
       ```bash
       docker info
       ```
    
    2. Check port availability:
       ```bash
       docker ps -a
       netstat -tulpn | grep 8000
       ```
    
    3. View container logs:
       ```bash
       docker logs wakebuilder
       ```

??? question "GPU not detected"

    1. Verify NVIDIA runtime:
       ```bash
       docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
       ```
    
    2. Check NVIDIA Container Toolkit installation
    
    3. Ensure `--gpus all` flag is used

??? question "Permission denied on volumes"

    1. Check volume permissions:
       ```bash
       ls -la models/ data/
       ```
    
    2. Set correct ownership:
       ```bash
       sudo chown -R $USER:$USER models/ data/
       ```
