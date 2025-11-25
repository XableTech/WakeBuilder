# WakeBuilder API Documentation

## Overview

WakeBuilder provides a RESTful API for training custom wake word detection models. The API is built with FastAPI and includes automatic interactive documentation.

**Base URL:** `http://localhost:8000`

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

---

## Authentication

Currently, the API does not require authentication as it's designed for local use. In production deployments, consider adding API key authentication.

---

## Endpoints

### Health & Info

#### GET /health
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### GET /api/info
Get system information.

**Response:**
```json
{
  "name": "WakeBuilder API",
  "version": "0.1.0",
  "description": "Wake word training platform API",
  "docs_url": "/docs",
  "system": {
    "version": "0.1.0",
    "python_version": "3.12.0",
    "torch_version": "2.1.0",
    "cuda_available": false,
    "device": "cpu"
  }
}
```

---

### Training

#### POST /api/train/start
Start a new wake word training job.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wake_word` | string | Yes | Wake word to train (1-2 words, letters and spaces only) |
| `recordings` | file[] | Yes | Audio recordings (3-5 WAV/FLAC/OGG files, 0.5-3s each) |
| `model_type` | string | No | `tc_resnet` (fast) or `bc_resnet` (accurate, default) |
| `batch_size` | int | No | Training batch size (8-256, default: 64) |
| `num_epochs` | int | No | Max training epochs (10-500, default: 100) |
| `learning_rate` | float | No | Initial learning rate (default: 0.001) |

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/train/start" \
  -F "wake_word=Hey Computer" \
  -F "model_type=bc_resnet" \
  -F "recordings=@recording1.wav" \
  -F "recordings=@recording2.wav" \
  -F "recordings=@recording3.wav"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Training job started successfully",
  "wake_word": "Hey Computer",
  "model_type": "bc_resnet"
}
```

**Errors:**
- `400` - Invalid wake word or audio files
- `503` - Server busy (another training job running)

---

#### GET /api/train/status/{job_id}
Get detailed status of a training job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "training",
  "progress_percent": 65.5,
  "current_phase": "Training the model...",
  "message": "Training epoch 45/100 (val_loss: 0.1234, val_acc: 95.2%)",
  "wake_word": "Hey Computer",
  "model_type": "bc_resnet",
  "started_at": "2024-01-15T10:30:00.000Z",
  "updated_at": "2024-01-15T10:35:00.000Z",
  "completed_at": null,
  "error": null,
  "training_progress": {
    "current_epoch": 45,
    "total_epochs": 100,
    "best_val_loss": 0.1234,
    "epochs_without_improvement": 3,
    "epoch_history": [
      {
        "epoch": 0,
        "train_loss": 0.6931,
        "train_accuracy": 0.5,
        "val_loss": 0.6800,
        "val_accuracy": 0.52,
        "val_f1": 0.48,
        "learning_rate": 0.001
      }
    ]
  },
  "hyperparameters": {
    "batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "early_stopping_patience": 20
  }
}
```

**Job Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `validating` | Validating audio recordings |
| `augmenting` | Creating voice variations (TTS + augmentation) |
| `generating_negatives` | Generating negative examples |
| `training` | Training the neural network |
| `calibrating` | Calibrating detection threshold |
| `saving` | Saving model files |
| `completed` | Training finished successfully |
| `failed` | Training failed (see `error` field) |

---

#### GET /api/train/download/{job_id}
Download trained model files as ZIP archive.

**Response:** `application/zip`

Contains:
- `model.pt` - PyTorch model weights
- `metadata.json` - Model configuration and metrics
- `training_history.json` - Epoch-by-epoch training metrics

**Errors:**
- `400` - Job not completed
- `404` - Job not found

---

#### GET /api/train/jobs
List all training jobs.

**Response:**
```json
{
  "jobs": [...],
  "total": 5,
  "active": 1
}
```

---

#### DELETE /api/train/{job_id}
Delete a completed or failed training job.

**Errors:**
- `400` - Cannot delete active job
- `404` - Job not found

---

### Models

#### GET /api/models/list
List all available wake word models.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by `default` or `custom` |

**Response:**
```json
{
  "models": [
    {
      "model_id": "hey_computer",
      "wake_word": "Hey Computer",
      "category": "custom",
      "model_type": "bc_resnet",
      "threshold": 0.65,
      "created_at": "2024-01-15T10:30:00.000Z",
      "parameters": 128000,
      "size_kb": 512.5,
      "metrics": {
        "val_accuracy": 0.952,
        "val_f1": 0.948
      },
      "threshold_analysis": [
        {
          "threshold": 0.5,
          "far": 0.02,
          "frr": 0.08,
          "accuracy": 0.95,
          "precision": 0.96,
          "recall": 0.92,
          "f1": 0.94
        }
      ]
    }
  ],
  "total_count": 5,
  "default_count": 3,
  "custom_count": 2
}
```

---

#### GET /api/models/{model_id}/metadata
Get detailed metadata for a specific model.

**Response:** Same as model object in list response.

---

#### GET /api/models/{model_id}/download
Download model files as ZIP archive.

---

#### DELETE /api/models/{model_id}
Delete a custom model.

**Note:** Default models cannot be deleted.

**Errors:**
- `403` - Cannot delete default model
- `404` - Model not found

---

#### GET /api/models/{model_id}/info
Get model architecture information.

**Response:**
```json
{
  "model_id": "hey_computer",
  "model_type": "bc_resnet",
  "n_mels": 80,
  "total_parameters": 128000,
  "size_mb": 0.49,
  "num_layers": 12,
  "layer_names": ["stem.0.weight", "stem.0.bias", ...]
}
```

---

### Testing

#### POST /api/test/file
Test a model with an uploaded audio file.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | file | Yes | Audio file to test (WAV/FLAC/OGG) |
| `model_id` | string | Yes | Model ID to test with |
| `threshold` | float | No | Custom threshold (0-1, uses model default if not set) |

**Response:**
```json
{
  "detected": true,
  "confidence": 0.92,
  "threshold": 0.65,
  "model_id": "hey_computer",
  "wake_word": "Hey Computer",
  "processing_time_ms": 45.2
}
```

---

#### WebSocket /api/test/realtime
Real-time wake word detection via WebSocket.

**Connection URL:**
```
ws://localhost:8000/api/test/realtime?model_id=hey_computer&threshold=0.65&cooldown_ms=1000
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | Model ID to test with |
| `threshold` | float | No | Detection threshold (0-1) |
| `cooldown_ms` | int | No | Cooldown between detections (default: 1000) |

**Client → Server:**
- Binary: Raw audio data (16-bit PCM, 16kHz, mono)
- JSON: Commands (see below)

**Server → Client:**
```json
// Ready message (on connect)
{
  "type": "ready",
  "model_id": "hey_computer",
  "wake_word": "Hey Computer",
  "threshold": 0.65,
  "cooldown_ms": 1000
}

// Detection event
{
  "type": "detection",
  "detected": true,
  "confidence": 0.92,
  "threshold": 0.65,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "inference_time_ms": 12.5,
  "in_cooldown": false
}

// Error
{
  "type": "error",
  "message": "Error description"
}
```

**Commands (JSON):**
```json
// Update threshold
{"type": "set_threshold", "threshold": 0.7}

// Update cooldown
{"type": "set_cooldown", "cooldown_ms": 500}

// Reset audio buffer
{"type": "reset"}

// Ping
{"type": "ping"}
```

---

#### GET /api/test/models
List models available for testing.

**Response:**
```json
{
  "models": [
    {
      "model_id": "hey_computer",
      "wake_word": "Hey Computer",
      "threshold": 0.65,
      "category": "custom"
    }
  ],
  "total": 5
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "detail": "Additional details (optional)"
}
```

**Common HTTP Status Codes:**
| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid input |
| `403` | Forbidden - Operation not allowed |
| `404` | Not Found - Resource doesn't exist |
| `500` | Internal Server Error |
| `503` | Service Unavailable - Server busy |

---

## Rate Limits

No rate limits are enforced for local use. Only one training job can run at a time.

---

## WebSocket Audio Format

For real-time testing, send audio as:
- **Format:** Raw PCM
- **Bit depth:** 16-bit signed integer
- **Sample rate:** 16,000 Hz
- **Channels:** Mono

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/test/realtime?model_id=hey_computer');

// Get microphone access
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(stream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
  const float32 = e.inputBuffer.getChannelData(0);
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
  }
  ws.send(int16.buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'detection' && data.detected) {
    console.log('Wake word detected!', data.confidence);
  }
};
```

---

## Model Types

| Type | Inference | Size | Accuracy | Use Case |
|------|-----------|------|----------|----------|
| `tc_resnet` | ~0.6ms | ~250KB | Good | Production, real-time |
| `bc_resnet` | ~6ms | ~468KB | Best | When accuracy matters |

---

## Changelog

### v0.1.0
- Initial API release
- Training, model management, and testing endpoints
- WebSocket real-time detection
- OpenAPI documentation
