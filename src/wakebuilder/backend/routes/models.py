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

        # Get file size
        size_kb = model_path.stat().st_size / 1024

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
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Log error but don't fail
        print(f"Warning: Failed to load metadata for {model_dir}: {e}")
        return None


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
""",
)
async def download_model(model_id: str) -> StreamingResponse:
    """
    Download a model as a ZIP archive.

    Works for both default and custom models.
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
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

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
        404: {"model": ErrorResponse, "description": "Model not found"},
        403: {"model": ErrorResponse, "description": "Cannot delete default model"},
    },
    summary="Delete Model",
    description="""
Delete a custom wake word model.

**Note:** Default (pre-trained) models cannot be deleted.
""",
)
async def delete_model(model_id: str) -> ModelDeleteResponse:
    """
    Delete a custom model.

    Only custom models can be deleted. Default models are protected.
    """
    import shutil

    # Check if it's a default model
    default_dir = Config.DEFAULT_MODELS_DIR / model_id
    if default_dir.exists():
        raise HTTPException(
            status_code=403,
            detail="Cannot delete default models. Only custom models can be deleted.",
        )

    # Check if custom model exists
    custom_dir = Config.CUSTOM_MODELS_DIR / model_id
    if not custom_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # Delete the model directory
    try:
        shutil.rmtree(custom_dir)
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete model: {e}"
        ) from e

    return ModelDeleteResponse(
        success=True,
        message="Model deleted successfully",
        model_id=model_id,
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
