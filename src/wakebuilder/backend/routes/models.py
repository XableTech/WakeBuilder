"""
Model management endpoints for WakeBuilder API.

This module provides endpoints for:
- Listing all available models (default and custom)
- Getting model metadata
- Downloading model files
- Deleting custom models
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...config import Config
from ..schemas import (
    ErrorResponse,
    ModelCategory,
    ModelDeleteResponse,
    ModelListResponse,
    ModelMetadata,
    ThresholdMetrics,
)

router = APIRouter()


def get_model_id(model_dir: Path) -> str:
    """Generate a model ID from the directory name."""
    return model_dir.name


def load_model_metadata(
    model_dir: Path, category: ModelCategory
) -> Optional[ModelMetadata]:
    """
    Load metadata for a model from its directory.

    Args:
        model_dir: Path to the model directory
        category: Whether this is a default or custom model

    Returns:
        ModelMetadata if valid, None otherwise
    """
    metadata_path = model_dir / "metadata.json"
    model_path = model_dir / "model.pt"

    if not metadata_path.exists() or not model_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            meta = json.load(f)

        # Get total package size (including ONNX files if present)
        size_kb = model_path.stat().st_size / 1024
        
        # Add ONNX files size if they exist
        onnx_path = model_dir / "model.onnx"
        onnx_data_path = model_dir / "model.onnx.data"
        if onnx_path.exists():
            size_kb += onnx_path.stat().st_size / 1024
        if onnx_data_path.exists():
            size_kb += onnx_data_path.stat().st_size / 1024

        # Parse creation time from metadata or file
        created_at = datetime.now()
        if "created_at" in meta:
            try:
                created_at = datetime.fromisoformat(meta["created_at"])
            except (ValueError, TypeError):
                pass
        else:
            # Use file modification time
            created_at = datetime.fromtimestamp(model_path.stat().st_mtime)

        # Parse threshold analysis if available
        threshold_analysis = None
        if "threshold_analysis" in meta:
            threshold_analysis = [
                ThresholdMetrics(
                    threshold=t["threshold"],
                    far=t["far"],
                    frr=t["frr"],
                    accuracy=t["accuracy"],
                    precision=t["precision"],
                    recall=t["recall"],
                    f1=t["f1"],
                )
                for t in meta["threshold_analysis"]
            ]

        return ModelMetadata(
            model_id=get_model_id(model_dir),
            wake_word=meta.get("wake_word", model_dir.name.replace("_", " ").title()),
            category=category,
            model_type=meta.get("model_type", "unknown"),
            threshold=meta.get("threshold", 0.5),
            created_at=created_at,
            parameters=meta.get("parameters", 0),
            size_kb=size_kb,
            metrics=meta.get("metrics"),
            threshold_analysis=threshold_analysis,
            training_config=meta.get("training_config"),
            data_stats=meta.get("data_stats"),
            training_time_seconds=meta.get("training_time_seconds"),
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Log error but don't fail
        print(f"Warning: Failed to load metadata for {model_dir}: {e}")
        return None


def generate_chart_images(model_dir: Path) -> dict[str, bytes]:
    """
    Generate chart images (loss, accuracy, threshold) for a model.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Dictionary mapping filename to PNG image bytes
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    charts = {}
    
    # Load training history
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history_raw = json.load(f)
        
        # Convert list format to dict format if needed
        if isinstance(history_raw, list):
            history = {
                "train_loss": [e.get("train_loss") for e in history_raw],
                "val_loss": [e.get("val_loss") for e in history_raw],
                "val_accuracy": [e.get("val_acc") for e in history_raw],
                "val_f1": [e.get("val_f1") for e in history_raw],
            }
        else:
            history = history_raw
        
        # Generate Loss Chart
        if "train_loss" in history and "val_loss" in history:
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(history["train_loss"]) + 1)
            ax.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            buf.seek(0)
            charts['loss_history.png'] = buf.read()
            plt.close(fig)
        
        # Generate Accuracy/F1 Chart
        if "val_accuracy" in history:
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(history["val_accuracy"]) + 1)
            ax.plot(epochs, [v * 100 for v in history["val_accuracy"]], 'g-', label='Val Accuracy', linewidth=2)
            if "val_f1" in history:
                ax.plot(epochs, [v * 100 for v in history["val_f1"]], 'orange', label='Val F1', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Accuracy & F1 History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            buf.seek(0)
            charts['accuracy_f1_history.png'] = buf.read()
            plt.close(fig)
    
    # Load metadata for threshold analysis
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        
        threshold_analysis = meta.get("threshold_analysis")
        if threshold_analysis:
            # Handle both list format and dict format
            if isinstance(threshold_analysis, list):
                thresholds = [t.get("threshold") for t in threshold_analysis]
                far = [t.get("far") for t in threshold_analysis]
                frr = [t.get("frr") for t in threshold_analysis]
            else:
                thresholds = threshold_analysis.get("thresholds", [])
                far = threshold_analysis.get("far", [])
                frr = threshold_analysis.get("frr", [])
            
            fig, ax = plt.subplots(figsize=(8, 5))
            optimal = meta.get("threshold", 0.5)
            
            if thresholds and far and frr:
                ax.plot(thresholds, [v * 100 for v in far], 'r-', label='FAR (False Accept)', linewidth=2)
                ax.plot(thresholds, [v * 100 for v in frr], 'orange', label='FRR (False Reject)', linewidth=2)
                ax.axvline(x=optimal, color='cyan', linestyle='--', label=f'Optimal: {optimal:.2f}', linewidth=2)
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Rate (%)')
                ax.set_title('Threshold Analysis (FAR/FRR)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 100)
                ax.set_facecolor('#1a1a2e')
                fig.patch.set_facecolor('#1a1a2e')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
                buf.seek(0)
                charts['threshold_analysis.png'] = buf.read()
                plt.close(fig)
    
    return charts


def scan_models_directory(
    base_dir: Path, category: ModelCategory
) -> list[ModelMetadata]:
    """
    Scan a directory for model subdirectories.

    Args:
        base_dir: Base directory to scan
        category: Category to assign to found models

    Returns:
        List of ModelMetadata for valid models
    """
    models: list[ModelMetadata] = []

    if not base_dir.exists():
        return models

    for model_dir in base_dir.iterdir():
        if model_dir.is_dir():
            metadata = load_model_metadata(model_dir, category)
            if metadata:
                models.append(metadata)

    return models


@router.get(
    "/list",
    response_model=ModelListResponse,
    summary="List All Models",
    description="""
List all available wake word models.

Returns both default (pre-trained) and custom (user-trained) models
with their metadata including wake word, threshold, and performance metrics.

**Filtering:**
- Use `category` parameter to filter by model type
- Results are sorted by creation date (newest first)
""",
)
async def list_models(
    category: Optional[ModelCategory] = Query(
        None, description="Filter by model category"
    ),
) -> ModelListResponse:
    """
    List all available models.

    Scans both default and custom model directories.
    """
    all_models: list[ModelMetadata] = []

    # Scan default models
    if category is None or category == ModelCategory.DEFAULT:
        default_models = scan_models_directory(
            Config.DEFAULT_MODELS_DIR, ModelCategory.DEFAULT
        )
        all_models.extend(default_models)

    # Scan custom models
    if category is None or category == ModelCategory.CUSTOM:
        custom_models = scan_models_directory(
            Config.CUSTOM_MODELS_DIR, ModelCategory.CUSTOM
        )
        all_models.extend(custom_models)

    # Sort by creation date (newest first)
    all_models.sort(key=lambda m: m.created_at, reverse=True)

    # Count by category
    default_count = sum(1 for m in all_models if m.category == ModelCategory.DEFAULT)
    custom_count = sum(1 for m in all_models if m.category == ModelCategory.CUSTOM)

    return ModelListResponse(
        models=all_models,
        total_count=len(all_models),
        default_count=default_count,
        custom_count=custom_count,
    )


@router.get(
    "/{model_id}/metadata",
    response_model=ModelMetadata,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
    summary="Get Model Metadata",
    description="""
Get detailed metadata for a specific model.

Returns comprehensive information including:
- Wake word and model architecture
- Recommended detection threshold
- Training metrics (accuracy, F1, etc.)
- FAR/FRR analysis at different thresholds
""",
)
async def get_model_metadata(model_id: str) -> ModelMetadata:
    """
    Get metadata for a specific model.

    Searches both default and custom model directories.
    """
    # Search in custom models first
    model_dir = Config.CUSTOM_MODELS_DIR / model_id
    if model_dir.exists():
        metadata = load_model_metadata(model_dir, ModelCategory.CUSTOM)
        if metadata:
            return metadata

    # Search in default models
    model_dir = Config.DEFAULT_MODELS_DIR / model_id
    if model_dir.exists():
        metadata = load_model_metadata(model_dir, ModelCategory.DEFAULT)
        if metadata:
            return metadata

    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


@router.get(
    "/{model_id}/training-history",
    responses={
        404: {"model": ErrorResponse, "description": "Model or training history not found"},
    },
    summary="Get Training History",
    description="Get training history (loss, accuracy, F1 per epoch) for a model.",
)
async def get_training_history(model_id: str) -> dict:
    """
    Get training history for a model.
    
    Returns loss and accuracy metrics per epoch.
    """
    # Search in custom models first
    model_dir = Config.CUSTOM_MODELS_DIR / model_id
    if not model_dir.exists():
        model_dir = Config.DEFAULT_MODELS_DIR / model_id
    
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    history_path = model_dir / "training_history.json"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail=f"Training history not found for model: {model_id}")
    
    with open(history_path) as f:
        history = json.load(f)
    
    # If history is a list of epoch objects, convert to dict format
    if isinstance(history, list):
        converted = {
            "train_loss": [e.get("train_loss") for e in history],
            "val_loss": [e.get("val_loss") for e in history],
            "train_accuracy": [e.get("train_acc") for e in history],
            "val_accuracy": [e.get("val_acc") for e in history],
            "val_f1": [e.get("val_f1") for e in history],
            "learning_rate": [e.get("lr") for e in history],
        }
        return converted
    
    return history


@router.get(
    "/{model_id}/download",
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Model files as ZIP archive",
        },
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
    summary="Download Model",
    description="""
Download model files as a ZIP archive.

The archive contains:
- `model.pt`: PyTorch model weights and architecture info
- `metadata.json`: Model configuration and metrics
- `training_history.json`: Training metrics per epoch (if available)
- `recordings/`: User recordings used for training (if available)
""",
)
async def download_model(model_id: str) -> StreamingResponse:
    """
    Download a model as a ZIP archive.

    Works for both default and custom models.
    Includes associated recordings if available.
    """
    # Find model directory
    model_dir = None

    custom_dir = Config.CUSTOM_MODELS_DIR / model_id
    if custom_dir.exists():
        model_dir = custom_dir

    default_dir = Config.DEFAULT_MODELS_DIR / model_id
    if default_dir.exists():
        model_dir = default_dir

    if not model_dir:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Create ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add model files
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)
        
        # Generate and add chart images
        chart_images = generate_chart_images(model_dir)
        for chart_name, chart_data in chart_images.items():
            zf.writestr(f"charts/{chart_name}", chart_data)
        
        # Add recordings if they exist
        recordings_dir = Config.RECORDINGS_DIR / model_id
        if recordings_dir.exists():
            for rec_file in recordings_dir.iterdir():
                if rec_file.is_file():
                    zf.write(rec_file, f"recordings/{rec_file.name}")

    zip_buffer.seek(0)

    # Get wake word for filename
    metadata_path = model_dir / "metadata.json"
    wake_word = model_id
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
                wake_word = meta.get("wake_word", model_id).lower().replace(" ", "_")
        except (json.JSONDecodeError, KeyError):
            pass

    filename = f"{wake_word}_model.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete(
    "/{model_id}",
    response_model=ModelDeleteResponse,
    responses={
        404: {"description": "Model not found"},
    },
    summary="Delete Model",
    description="""
Delete a wake word model (custom or default).

Also deletes associated recordings if they exist.
""",
)
async def delete_model(model_id: str) -> ModelDeleteResponse:
    """
    Delete a model (custom or default).

    Also deletes associated recordings.
    """
    import shutil

    # Check custom models first, then default
    model_dir = None
    if (Config.CUSTOM_MODELS_DIR / model_id).exists():
        model_dir = Config.CUSTOM_MODELS_DIR / model_id
    elif (Config.DEFAULT_MODELS_DIR / model_id).exists():
        model_dir = Config.DEFAULT_MODELS_DIR / model_id
    
    if model_dir is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Delete the model directory
    try:
        shutil.rmtree(model_dir)
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete model: {e}"
        ) from e
    
    # Delete associated recordings if they exist
    recordings_dir = Config.RECORDINGS_DIR / model_id
    recordings_deleted = False
    if recordings_dir.exists():
        try:
            shutil.rmtree(recordings_dir)
            recordings_deleted = True
        except OSError:
            pass  # Non-critical, just log

    return ModelDeleteResponse(
        success=True,
        message=f"Model deleted successfully{' (including recordings)' if recordings_deleted else ''}",
        model_id=model_id,
    )


@router.post(
    "/{model_id}/move-to-default",
    summary="Move Model to Default",
    description="Move a custom model to the default models folder.",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        400: {"model": ErrorResponse, "description": "Model already in default folder"},
    },
)
async def move_to_default(model_id: str) -> dict:
    """
    Move a custom model to the default models folder.
    
    This makes the model a "default" model that cannot be deleted through the UI.
    """
    import shutil
    
    # Check if already in default folder
    default_dir = Config.DEFAULT_MODELS_DIR / model_id
    if default_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Model is already in the default folder.",
        )
    
    # Check if custom model exists
    custom_dir = Config.CUSTOM_MODELS_DIR / model_id
    if not custom_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    # Move the model directory
    try:
        shutil.move(str(custom_dir), str(default_dir))
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to move model: {e}"
        ) from e
    
    return {
        "success": True,
        "message": f"Model '{model_id}' moved to default folder",
        "model_id": model_id,
        "new_category": "default",
    }


@router.get(
    "/{model_id}/recordings",
    summary="List Model Recordings",
    description="Get list of user recordings associated with a model.",
)
async def list_model_recordings(model_id: str) -> dict:
    """
    List recordings associated with a model.
    
    Returns list of recording files with their metadata.
    """
    recordings_dir = Config.RECORDINGS_DIR / model_id
    
    if not recordings_dir.exists():
        return {
            "model_id": model_id,
            "recordings": [],
            "count": 0,
        }
    
    recordings = []
    for rec_file in sorted(recordings_dir.iterdir()):
        if rec_file.is_file() and rec_file.suffix.lower() in [".wav", ".flac", ".ogg", ".mp3"]:
            recordings.append({
                "filename": rec_file.name,
                "size_kb": rec_file.stat().st_size / 1024,
                "url": f"/api/models/{model_id}/recordings/{rec_file.name}",
            })
    
    return {
        "model_id": model_id,
        "recordings": recordings,
        "count": len(recordings),
    }


@router.get(
    "/{model_id}/recordings/{filename}",
    summary="Download Recording",
    description="Download a specific recording file.",
)
async def download_recording(model_id: str, filename: str) -> StreamingResponse:
    """
    Download a specific recording file.
    """
    recordings_dir = Config.RECORDINGS_DIR / model_id
    file_path = recordings_dir / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Recording not found: {filename}")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return StreamingResponse(
        open(file_path, "rb"),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={filename}"},
    )


@router.get(
    "/{model_id}/info",
    summary="Get Model Architecture Info",
    description="Get detailed information about the model architecture and parameters.",
)
async def get_model_info(model_id: str) -> dict:
    """
    Get detailed model architecture information.

    Loads the model and returns parameter counts and layer information.
    """
    # Find model directory
    model_path = None

    custom_path = Config.CUSTOM_MODELS_DIR / model_id / "model.pt"
    if custom_path.exists():
        model_path = custom_path

    default_path = Config.DEFAULT_MODELS_DIR / model_id / "model.pt"
    if default_path.exists():
        model_path = default_path

    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        model_type = checkpoint.get("model_type", "unknown")
        n_mels = checkpoint.get("n_mels", 80)

        # Count parameters from state dict
        state_dict = checkpoint.get("model_state_dict", {})
        total_params = sum(
            p.numel() for p in [torch.tensor(v) for v in state_dict.values()]
        )

        # Get layer names
        layer_names = list(state_dict.keys())

        return {
            "model_id": model_id,
            "model_type": model_type,
            "n_mels": n_mels,
            "total_parameters": total_params,
            "size_mb": total_params * 4 / (1024 * 1024),
            "num_layers": len(set(name.split(".")[0] for name in layer_names)),
            "layer_names": layer_names[:20],  # First 20 layers
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load model info: {e}"
        ) from e


@router.post(
    "/{model_id}/export-onnx",
    summary="Export Model to ONNX",
    description="""
Export a trained model to ONNX format for deployment.

The ONNX model includes the full pipeline (AST base + classifier).
Uses torch.onnx.export for conversion.
""",
)
async def export_to_onnx(model_id: str) -> dict:
    """
    Export a model to ONNX format.
    
    Creates model.onnx in the model directory.
    """
    import tempfile
    
    # Find model directory
    model_dir = None
    model_path = None
    
    custom_path = Config.CUSTOM_MODELS_DIR / model_id / "model.pt"
    if custom_path.exists():
        model_dir = Config.CUSTOM_MODELS_DIR / model_id
        model_path = custom_path
    
    default_path = Config.DEFAULT_MODELS_DIR / model_id / "model.pt"
    if default_path.exists():
        model_dir = Config.DEFAULT_MODELS_DIR / model_id
        model_path = default_path
    
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    try:
        from ...models.classifier import ASTWakeWordModel
        from transformers import AutoFeatureExtractor
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Create model
        model = ASTWakeWordModel(
            freeze_base=True,
            classifier_hidden_dims=checkpoint.get("classifier_hidden_dims", [256, 128]),
            classifier_dropout=checkpoint.get("classifier_dropout", 0.3),
            use_attention=checkpoint.get("use_attention", False),
            use_se_block=checkpoint.get("use_se_block", False),
            use_tcn=checkpoint.get("use_tcn", False),
        )
        
        # Load classifier weights
        if "classifier_state_dict" in checkpoint:
            model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        
        model.eval()
        
        # Create dummy input for ONNX export
        # AST expects input of shape (batch, max_length, num_mel_bins)
        # The feature extractor produces this shape
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-speech-commands-v2"
        )
        
        # Create dummy audio (1 second at 16kHz)
        import numpy as np
        dummy_audio = np.random.randn(16000).astype(np.float32)
        inputs = feature_extractor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        dummy_input = inputs["input_values"]
        
        # Export to ONNX
        onnx_path = model_dir / "model.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        
        # Get file size
        onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        
        return {
            "success": True,
            "model_id": model_id,
            "onnx_path": str(onnx_path),
            "onnx_size_mb": round(onnx_size_mb, 2),
            "message": f"Model exported to ONNX format ({onnx_size_mb:.2f} MB)",
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export model to ONNX: {e}\n{traceback.format_exc()}"
        ) from e


@router.get(
    "/{model_id}/onnx-status",
    summary="Check ONNX Export Status",
    description="Check if a model has been exported to ONNX format.",
)
async def get_onnx_status(model_id: str) -> dict:
    """
    Check if ONNX export exists for a model.
    """
    # Check custom models
    custom_onnx = Config.CUSTOM_MODELS_DIR / model_id / "model.onnx"
    if custom_onnx.exists():
        return {
            "model_id": model_id,
            "onnx_available": True,
            "onnx_path": str(custom_onnx),
            "onnx_size_mb": round(custom_onnx.stat().st_size / (1024 * 1024), 2),
        }
    
    # Check default models
    default_onnx = Config.DEFAULT_MODELS_DIR / model_id / "model.onnx"
    if default_onnx.exists():
        return {
            "model_id": model_id,
            "onnx_available": True,
            "onnx_path": str(default_onnx),
            "onnx_size_mb": round(default_onnx.stat().st_size / (1024 * 1024), 2),
        }
    
    return {
        "model_id": model_id,
        "onnx_available": False,
        "message": "ONNX export not found. Use POST /api/models/{model_id}/export-onnx to create it.",
    }


@router.delete(
    "/{model_id}/onnx",
    summary="Delete ONNX Export",
    description="Delete the ONNX export for a model.",
)
async def delete_onnx_export(model_id: str) -> dict:
    """
    Delete ONNX export for a model.
    """
    # Check custom models
    custom_onnx = Config.CUSTOM_MODELS_DIR / model_id / "model.onnx"
    if custom_onnx.exists():
        custom_onnx.unlink()
        return {"success": True, "message": "ONNX export deleted"}
    
    # Check default models
    default_onnx = Config.DEFAULT_MODELS_DIR / model_id / "model.onnx"
    if default_onnx.exists():
        default_onnx.unlink()
        return {"success": True, "message": "ONNX export deleted"}
    
    raise HTTPException(status_code=404, detail="ONNX export not found")


@router.delete(
    "/recordings/orphaned",
    summary="Clean Orphaned Recordings",
    description="Delete recordings that don't have an associated model.",
)
async def clean_orphaned_recordings() -> dict:
    """
    Delete recordings directories that don't have a corresponding model.
    
    This cleans up recordings from models that were deleted before
    the automatic recording cleanup was implemented.
    """
    import shutil
    
    if not Config.RECORDINGS_DIR.exists():
        return {
            "success": True,
            "message": "No recordings directory found",
            "deleted": [],
            "count": 0,
        }
    
    # Get all model IDs
    model_ids = set()
    if Config.CUSTOM_MODELS_DIR.exists():
        for d in Config.CUSTOM_MODELS_DIR.iterdir():
            if d.is_dir():
                model_ids.add(d.name)
    if Config.DEFAULT_MODELS_DIR.exists():
        for d in Config.DEFAULT_MODELS_DIR.iterdir():
            if d.is_dir():
                model_ids.add(d.name)
    
    # Find and delete orphaned recordings
    deleted = []
    for recordings_dir in Config.RECORDINGS_DIR.iterdir():
        if recordings_dir.is_dir() and recordings_dir.name not in model_ids:
            try:
                shutil.rmtree(recordings_dir)
                deleted.append(recordings_dir.name)
            except OSError:
                pass  # Skip if can't delete
    
    return {
        "success": True,
        "message": f"Deleted {len(deleted)} orphaned recording(s)",
        "deleted": deleted,
        "count": len(deleted),
    }


@router.get(
    "/recordings/orphaned",
    summary="List Orphaned Recordings",
    description="List recordings that don't have an associated model.",
)
async def list_orphaned_recordings() -> dict:
    """
    List recordings directories that don't have a corresponding model.
    """
    if not Config.RECORDINGS_DIR.exists():
        return {
            "orphaned": [],
            "count": 0,
        }
    
    # Get all model IDs
    model_ids = set()
    if Config.CUSTOM_MODELS_DIR.exists():
        for d in Config.CUSTOM_MODELS_DIR.iterdir():
            if d.is_dir():
                model_ids.add(d.name)
    if Config.DEFAULT_MODELS_DIR.exists():
        for d in Config.DEFAULT_MODELS_DIR.iterdir():
            if d.is_dir():
                model_ids.add(d.name)
    
    # Find orphaned recordings
    orphaned = []
    for recordings_dir in Config.RECORDINGS_DIR.iterdir():
        if recordings_dir.is_dir() and recordings_dir.name not in model_ids:
            # Count files in directory
            file_count = sum(1 for f in recordings_dir.iterdir() if f.is_file())
            orphaned.append({
                "name": recordings_dir.name,
                "file_count": file_count,
            })
    
    return {
        "orphaned": orphaned,
        "count": len(orphaned),
    }
