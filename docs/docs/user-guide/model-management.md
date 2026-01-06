# Model Management

Managing, downloading, and organizing your trained wake word models.

---

## Model Dashboard

The home page displays all your wake word models:

### Model Cards

Each model card shows:

- **Wake word name**: The trigger phrase
- **Type badge**: "Custom" or "Default"
- **Creation date**: When the model was trained
- **Accuracy**: Validation accuracy
- **F1 Score**: Balanced performance metric
- **Action buttons**: Test, Download, Delete

### Filtering Models

Use the filter buttons to view:

| Filter | Shows |
|--------|-------|
| **All** | All available models |
| **Custom** | User-trained models |
| **Default** | Pre-trained models |

---

## Model Storage

### Directory Structure

Models are stored in the `models/` directory:

```
models/
├── default/              # Pre-trained default models
│   └── assistant/
│       ├── assistant.pt
│       └── assistant.json
└── custom/               # User-trained models
    └── jarvis/
        ├── jarvis.pt
        ├── jarvis.json
        └── jarvis.onnx
```

### Model Files

Each model consists of:

| File | Description |
|------|-------------|
| `{name}.pt` | PyTorch model weights |
| `{name}.json` | Metadata (threshold, metrics, config) |
| `{name}.onnx` | ONNX export (optional) |

### Metadata JSON

Example model metadata:

```json
{
  "wake_word": "jarvis",
  "model_type": "ast",
  "created_at": "2026-01-06T15:00:00Z",
  "threshold": 0.65,
  "accuracy": 0.971,
  "f1_score": 0.943,
  "precision": 0.952,
  "recall": 0.934,
  "trainable_params": 234567,
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "dropout": 0.5,
    "epochs_trained": 45,
    "positive_samples": 5000,
    "negative_samples": 10000
  },
  "recordings": [
    "recording_001.wav",
    "recording_002.wav",
    "recording_003.wav"
  ]
}
```

---

## Downloading Models

### From the Dashboard

1. Click the **Download** button on a model card
2. A ZIP file will be downloaded

### Download Contents

```
jarvis_model.zip
├── jarvis.pt        # PyTorch weights
├── jarvis.json      # Metadata
└── jarvis.onnx      # ONNX model (if available)
```

### Via API

```bash
# Download model as ZIP
curl -O http://localhost:8000/api/models/jarvis/download

# Get model file directly
curl -O http://localhost:8000/api/models/jarvis/file
```

---

## Deleting Models

### From the Dashboard

1. Click the **Delete** button (trash icon) on a model card
2. Confirm the deletion in the dialog

!!! warning "Deletion is Permanent"
    Deleted models cannot be recovered. Download a backup first if needed.

### Via API

```bash
curl -X DELETE http://localhost:8000/api/models/jarvis
```

---

## Model API Reference

### List All Models

```
GET /api/models
```

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
  "training_config": { ... },
  "has_onnx": true
}
```

### Download Model

```
GET /api/models/{model_id}/download
```

Returns a ZIP file containing all model files.

### Delete Model

```
DELETE /api/models/{model_id}
```

---

## ONNX Export

### What is ONNX?

**ONNX (Open Neural Network Exchange)** is a cross-platform format for deploying ML models. Benefits include:

- Platform-independent (Windows, Linux, macOS, mobile)
- Optimized runtime performance
- Compatible with WakeEngine

### Export Methods

#### Automatic Export

ONNX models are automatically created after training if the export succeeds.

#### Manual Export

```bash
python scripts/export_onnx.py --model jarvis
```

#### Via API

```bash
curl -X POST http://localhost:8000/api/models/jarvis/export-onnx
```

### Export Status

Check if ONNX export is available:

```json
{
  "has_onnx": true,
  "onnx_size_bytes": 2450000
}
```

---

## Importing Models

### Manual Import

1. Copy model files to `models/custom/{wake_word}/`
2. Ensure both `.pt` and `.json` files are present
3. Refresh the dashboard

### Required Files

| File | Required | Description |
|------|----------|-------------|
| `{name}.pt` | Yes | Model weights |
| `{name}.json` | Yes | Metadata |
| `{name}.onnx` | No | ONNX export |

---

## Model Versioning

### Naming Convention

WakeBuilder uses the wake word as the model name. To create multiple versions:

- Train with different parameters
- Rename the model folder manually
- Keep backups of previous versions

### Manual Backup

```bash
# Backup a model
cp -r models/custom/jarvis models/custom/jarvis_v1

# Restore from backup
cp -r models/custom/jarvis_v1 models/custom/jarvis
```

---

## Troubleshooting

??? question "Model not appearing in dashboard"

    1. Check files exist in `models/custom/{name}/`
    2. Ensure both `.pt` and `.json` files are present
    3. Refresh the page
    4. Check server logs for errors

??? question "Download failing"

    1. Check the model files exist
    2. Ensure sufficient disk space
    3. Try downloading via API directly

??? question "ONNX export not available"

    1. Export may have failed during training
    2. Try manual export: `python scripts/export_onnx.py --model {name}`
    3. Check ONNX dependencies are installed
