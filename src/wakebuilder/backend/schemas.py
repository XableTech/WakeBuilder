"""
Pydantic schemas for WakeBuilder API.

This module defines all request/response models for the API endpoints,
providing validation, serialization, and automatic documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================


class JobStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"
    VALIDATING = "validating"
    AUGMENTING = "augmenting"
    GENERATING_NEGATIVES = "generating_negatives"
    TRAINING = "training"
    CALIBRATING = "calibrating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, Enum):
    """Type of wake word model architecture."""

    AST = "ast"  # Audio Spectrogram Transformer (default)
    # Legacy types for backward compatibility
    TC_RESNET = "tc_resnet"
    BC_RESNET = "bc_resnet"


class ModelCategory(str, Enum):
    """Category of model (default or custom)."""

    DEFAULT = "default"
    CUSTOM = "custom"


# ============================================================================
# Training Schemas
# ============================================================================


class TrainingRequest(BaseModel):
    """Request to start a new training job."""

    wake_word: str = Field(
        ...,
        min_length=2,
        max_length=30,
        description="The wake word to train (1-2 words, letters and spaces only)",
        examples=["Hey Computer", "Phoenix"],
    )
    model_type: ModelType = Field(
        default=ModelType.AST,
        description="Model architecture: ast (Audio Spectrogram Transformer, recommended)",
    )
    # Audio recordings will be sent as base64 or multipart form data
    # This is handled separately in the endpoint

    @field_validator("wake_word")
    @classmethod
    def validate_wake_word(cls, v: str) -> str:
        """Validate wake word format."""
        # Strip and normalize whitespace
        v = " ".join(v.split())

        # Check allowed characters
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        if not all(c in allowed for c in v):
            raise ValueError("Wake word must contain only letters and spaces")

        # Check word count
        words = v.split()
        if len(words) > 2:
            raise ValueError("Wake word must be 1 or 2 words")

        return v


class TrainingHyperparameters(BaseModel):
    """Hyperparameters for training (optional customization)."""

    batch_size: int = Field(default=32, ge=8, le=256, description="Training batch size")
    num_epochs: int = Field(
        default=100, ge=10, le=500, description="Maximum training epochs"
    )
    learning_rate: float = Field(
        default=5e-4, gt=0, le=0.1, description="Initial learning rate"
    )
    dropout: float = Field(default=0.2, ge=0, le=0.7, description="Dropout probability")
    early_stopping_patience: int = Field(
        default=15, ge=5, le=50, description="Epochs to wait before early stopping"
    )
    # Regularization parameters - balanced for good learning
    weight_decay: float = Field(
        default=1e-4, ge=0, le=0.5, description="L2 regularization strength"
    )
    label_smoothing: float = Field(
        default=0.05,
        ge=0,
        le=0.3,
        description="Label smoothing factor (lower = more confident)",
    )
    spec_augment: bool = Field(
        default=True, description="Enable SpecAugment (time/frequency masking)"
    )
    mixup_alpha: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Mixup augmentation strength (lower = preserve patterns)",
    )
    # Data generation settings
    target_positive_samples: int = Field(
        default=6000,
        ge=100,
        le=20000,
        description="Target number of positive samples to generate from recordings",
    )
    use_tts_positives: bool = Field(
        default=True, description="Generate additional positive samples using TTS"
    )
    use_real_negatives: bool = Field(
        default=True, description="Use real negative data from data/negative/ directory"
    )
    max_real_negatives: int = Field(
        default=0,
        ge=0,
        le=100000,
        description="Maximum real negative samples (0 = no limit, use all)",
    )


class TrainingStartResponse(BaseModel):
    """Response after starting a training job."""

    job_id: str = Field(..., description="Unique identifier for the training job")
    message: str = Field(..., description="Status message")
    wake_word: str = Field(..., description="The wake word being trained")
    model_type: str = Field(..., description="Model architecture being used")


class EpochMetrics(BaseModel):
    """Metrics for a single training epoch."""

    epoch: int = Field(..., description="Epoch number (0-indexed)")
    train_loss: float = Field(..., description="Training loss")
    train_accuracy: float = Field(..., description="Training accuracy (0-1)")
    val_loss: float = Field(..., description="Validation loss")
    val_accuracy: float = Field(..., description="Validation accuracy (0-1)")
    val_f1: float = Field(..., description="Validation F1 score")
    learning_rate: float = Field(..., description="Current learning rate")


class DataStats(BaseModel):
    """Statistics about training data."""

    num_recordings: int = Field(0, description="Number of original recordings")
    num_positive_samples: int = Field(
        0, description="Total positive samples after augmentation"
    )
    num_negative_samples: int = Field(0, description="Total negative samples generated")
    num_train_samples: int = Field(0, description="Training set size")
    num_val_samples: int = Field(0, description="Validation set size")


class TrainingProgress(BaseModel):
    """Detailed training progress information."""

    current_epoch: int = Field(..., description="Current epoch number")
    total_epochs: int = Field(..., description="Total planned epochs")
    best_val_loss: float = Field(..., description="Best validation loss so far")
    epochs_without_improvement: int = Field(
        ..., description="Epochs since last improvement"
    )
    epoch_history: list[EpochMetrics] = Field(
        default_factory=list, description="History of all epoch metrics"
    )
    data_stats: Optional[DataStats] = Field(
        None, description="Training data statistics"
    )


class TrainingStatusResponse(BaseModel):
    """Response for training job status."""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress_percent: float = Field(
        ..., ge=0, le=100, description="Overall progress percentage"
    )
    current_phase: str = Field(..., description="Human-readable current phase")
    message: str = Field(..., description="Status message")
    wake_word: str = Field(..., description="Wake word being trained")
    model_type: str = Field(..., description="Model architecture")
    started_at: datetime = Field(..., description="Job start time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion time (if completed)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")
    training_progress: Optional[TrainingProgress] = Field(
        None, description="Detailed training progress (during training phase)"
    )
    hyperparameters: Optional[TrainingHyperparameters] = Field(
        None, description="Training hyperparameters used"
    )


# ============================================================================
# Model Management Schemas
# ============================================================================


class ThresholdMetrics(BaseModel):
    """Metrics at a specific detection threshold."""

    threshold: float = Field(..., description="Detection threshold value")
    far: float = Field(..., description="False Acceptance Rate")
    frr: float = Field(..., description="False Rejection Rate")
    accuracy: float = Field(..., description="Overall accuracy")
    precision: float = Field(..., description="Precision for wake word class")
    recall: float = Field(..., description="Recall for wake word class")
    f1: float = Field(..., description="F1 score")


class ModelMetadata(BaseModel):
    """Metadata for a trained wake word model."""

    model_id: str = Field(..., description="Unique model identifier")
    wake_word: str = Field(..., description="The wake word this model detects")
    category: ModelCategory = Field(..., description="Model category (default/custom)")
    model_type: str = Field(..., description="Model architecture")
    threshold: float = Field(..., description="Recommended detection threshold")
    created_at: datetime = Field(..., description="Model creation timestamp")
    parameters: int = Field(..., description="Number of model parameters")
    size_kb: float = Field(..., description="Model file size in KB")
    metrics: Optional[dict[str, Any]] = Field(
        None, description="Training metrics (accuracy, F1, etc.)"
    )
    threshold_analysis: Optional[list[ThresholdMetrics]] = Field(
        None, description="FAR/FRR at different thresholds"
    )
    training_config: Optional[dict[str, Any]] = Field(
        None, description="Training configuration used"
    )
    data_stats: Optional[dict[str, Any]] = Field(
        None, description="Training data statistics"
    )
    training_time_seconds: Optional[float] = Field(
        None, description="Training duration in seconds"
    )


class ModelListResponse(BaseModel):
    """Response for listing all models."""

    models: list[ModelMetadata] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")
    default_count: int = Field(..., description="Number of default models")
    custom_count: int = Field(..., description="Number of custom models")


class ModelDeleteResponse(BaseModel):
    """Response after deleting a model."""

    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")
    model_id: str = Field(..., description="ID of deleted model")


# ============================================================================
# Testing Schemas
# ============================================================================


class DetectionEvent(BaseModel):
    """A wake word detection event."""

    detected: bool = Field(..., description="Whether wake word was detected")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence score")
    threshold: float = Field(..., description="Threshold used for detection")
    timestamp: datetime = Field(..., description="Detection timestamp")
    audio_duration_ms: float = Field(..., description="Duration of audio processed")


class TestSessionConfig(BaseModel):
    """Configuration for a real-time testing session."""

    model_id: str = Field(..., description="Model to test with")
    threshold: Optional[float] = Field(
        None, ge=0, le=1, description="Custom threshold (uses model default if not set)"
    )
    cooldown_ms: int = Field(
        default=1000, ge=0, le=5000, description="Cooldown between detections in ms"
    )


class TestFileRequest(BaseModel):
    """Request to test a model with an audio file."""

    model_id: str = Field(..., description="Model to test with")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Custom threshold")


class TestFileResponse(BaseModel):
    """Response from testing with an audio file."""

    detected: bool = Field(..., description="Whether wake word was detected")
    confidence: float = Field(..., description="Detection confidence")
    threshold: float = Field(..., description="Threshold used")
    model_id: str = Field(..., description="Model used for testing")
    wake_word: str = Field(..., description="Wake word the model detects")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


# ============================================================================
# Health & Info Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server time")


class SystemInfo(BaseModel):
    """System information."""

    version: str = Field(..., description="WakeBuilder version")
    python_version: str = Field(..., description="Python version")
    torch_version: str = Field(..., description="PyTorch version")
    cuda_available: bool = Field(..., description="Whether CUDA is available")
    device: str = Field(..., description="Device being used for inference")


class APIInfo(BaseModel):
    """API information response."""

    name: str = Field(default="WakeBuilder API", description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    docs_url: str = Field(..., description="URL to API documentation")
    system: SystemInfo = Field(..., description="System information")


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class ValidationErrorDetail(BaseModel):
    """Detail for a validation error."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")


class ValidationErrorResponse(BaseModel):
    """Response for validation errors."""

    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(
        default="Request validation failed", description="Error message"
    )
    details: list[ValidationErrorDetail] = Field(
        ..., description="List of validation errors"
    )
