# API Endpoints

Complete reference for WakeBuilder's REST API and WebSocket endpoints.

---

## Base URL

```
http://localhost:8000
```

All endpoints are relative to this base URL.

---

## Authentication

Currently, WakeBuilder does not require authentication. For production deployment, implement authentication at the reverse proxy level.

---

## Health & Info

### Health Check

```
GET /health
```

Returns the current health status of the API.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-06T15:00:00Z"
}
```

### API Info

```
GET /api/info
```

Returns detailed API and system information.

**Response:**

```json
{
  "name": "WakeBuilder API",
  "version": "0.1.0",
  "description": "Wake word training platform API",
  "docs_url": "/docs",
  "system": {
    "python_version": "3.12.0",
    "torch_version": "2.0.0",
    "cuda_available": true,
    "device": "cuda"
  }
}
```

---

## Training Endpoints

### Start Training

```
POST /api/train/start
```

Starts a new wake word training job.

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wake_word` | string | Yes | Wake word text |
| `audio_files` | file[] | Yes | Voice recordings (1-5) |
| `model_type` | string | No | Model type (default: "ast") |
| `batch_size` | int | No | Batch size (default: 32) |
| `learning_rate` | float | No | Learning rate (default: 0.0001) |
| `num_epochs` | int | No | Max epochs (default: 100) |
| `early_stopping` | int | No | Patience epochs (default: 8) |
| `dropout` | float | No | Dropout rate (default: 0.5) |
| `label_smoothing` | float | No | Label smoothing (default: 0.1) |
| `mixup_alpha` | float | No | Mixup alpha (default: 0.5) |
| `use_focal_loss` | bool | No | Enable focal loss |
| `focal_alpha` | float | No | Focal alpha (default: 0.5) |
| `focal_gamma` | float | No | Focal gamma (default: 2.0) |
| `use_attention` | bool | No | Enable attention |
| `use_se_block` | bool | No | Enable SE block |
| `use_tcn` | bool | No | Enable TCN |
| `use_tts_positives` | bool | No | Generate TTS samples |
| `use_real_negatives` | bool | No | Use cached negatives |
| `use_hard_negatives` | bool | No | Generate hard negatives |
| `target_positive_samples` | int | No | Target sample count |
| `negative_ratio` | float | No | Negative:positive ratio |
| `hard_negative_ratio` | float | No | Hard negative ratio |

**Response:**

```json
{
  "job_id": "abc123",
  "status": "pending",
  "message": "Training job created"
}
```

### Get Training Status

```
GET /api/train/status/{job_id}
```

Returns the current status of a training job.

**Response:**

```json
{
  "job_id": "abc123",
  "status": "training",
  "phase": "Training classifier",
  "progress": 45.5,
  "current_epoch": 23,
  "total_epochs": 50,
  "metrics": {
    "train_loss": 0.234,
    "val_loss": 0.198,
    "val_accuracy": 0.956,
    "val_f1": 0.923
  },
  "data_stats": {
    "recordings": 3,
    "positive_samples": 5000,
    "negative_samples": 10000,
    "train_size": 11250,
    "val_size": 3750
  },
  "elapsed_time": "5m 23s"
}
```

### Download Model

```
GET /api/train/download/{job_id}
```

Downloads the trained model as a ZIP file.

**Response:**

Binary ZIP file containing:

- `{wake_word}.pt` - Model weights
- `{wake_word}.json` - Metadata
- `{wake_word}.onnx` - ONNX export (if available)

### Delete Job

```
DELETE /api/train/{job_id}
```

Deletes a completed or failed training job.

**Response:**

```json
{
  "message": "Job deleted successfully"
}
```

### List Jobs

```
GET /api/train/jobs
```

Lists all training jobs.

**Response:**

```json
{
  "jobs": [
    {
      "job_id": "abc123",
      "wake_word": "jarvis",
      "status": "completed",
      "created_at": "2026-01-06T15:00:00Z"
    }
  ]
}
```

---

## Model Endpoints

### List Models

```
GET /api/models
```

Lists all available wake word models.

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `type` | string | Filter by type: "custom", "default", "all" |

**Response:**

```json
{
  "models": [
    {
      "id": "jarvis",
      "wake_word": "jarvis",
      "type": "custom",
      "created_at": "2026-01-06T15:00:00Z",
      "accuracy": 0.971,
      "f1_score": 0.943,
      "threshold": 0.65
    }
  ],
  "total": 1
}
```

### Get Model Details

```
GET /api/models/{model_id}
```

Returns detailed information about a specific model.

**Response:**

```json
{
  "id": "jarvis",
  "wake_word": "jarvis",
  "type": "custom",
  "created_at": "2026-01-06T15:00:00Z",
  "threshold": 0.65,
  "metrics": {
    "accuracy": 0.971,
    "f1_score": 0.943,
    "precision": 0.952,
    "recall": 0.934
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "dropout": 0.5
  },
  "trainable_params": 234567,
  "has_onnx": true
}
```

### Download Model

```
GET /api/models/{model_id}/download
```

Downloads the model as a ZIP file.

### Delete Model

```
DELETE /api/models/{model_id}
```

Deletes a model.

**Response:**

```json
{
  "message": "Model deleted successfully"
}
```

---

## Testing Endpoints

### Test with File

```
POST /api/test/file
```

Tests a model with an uploaded audio file.

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model to test |
| `audio` | file | Yes | Audio file |
| `threshold` | float | No | Detection threshold |

**Response:**

```json
{
  "detected": true,
  "confidence": 0.892,
  "threshold": 0.65,
  "model_id": "jarvis",
  "processing_time_ms": 156
}
```

### WebSocket Testing

```
WS /api/test/ws/{model_id}
```

Real-time audio testing via WebSocket.

**Client → Server:**

```json
{
  "type": "audio",
  "data": "<base64 encoded audio>",
  "sample_rate": 16000
}
```

**Server → Client:**

```json
{
  "type": "detection",
  "detected": true,
  "confidence": 0.87,
  "threshold": 0.65,
  "timestamp": "2026-01-06T15:30:00Z"
}
```

### Set Device

```
POST /api/test/device
```

Sets the inference device (CPU or GPU).

**Request:**

```json
{
  "device": "cuda"
}
```

### Get Device Info

```
GET /api/test/device
```

Returns current device information.

**Response:**

```json
{
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060"
}
```

---

## Cache Endpoints

### Get Cache Info

```
GET /api/cache/info
```

Returns negative data cache status.

**Response:**

```json
{
  "available": true,
  "chunk_count": 47438,
  "size_mb": 1250,
  "last_built": "2026-01-05T10:00:00Z"
}
```

### Build Cache

```
POST /api/cache/build
```

Starts building the negative data cache.

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `max_workers` | int | Parallel workers (default: 4) |

**Response:**

```json
{
  "message": "Cache build started",
  "estimated_time": "Several minutes (depends on system)"
}
```

### Clear Cache

```
DELETE /api/cache
```

Clears the negative data cache.

**Response:**

```json
{
  "message": "Cache cleared successfully"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": {}
}
```

### Common Error Codes

| Status | Error Type | Description |
|--------|------------|-------------|
| 400 | `validation_error` | Invalid input |
| 404 | `not_found` | Resource not found |
| 409 | `conflict` | Resource already exists |
| 500 | `internal_error` | Server error |

---

## Rate Limits

Currently, no rate limits are enforced. For production, implement rate limiting at the reverse proxy level.
