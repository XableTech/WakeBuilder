"""
Training endpoints for WakeBuilder API.

This module provides endpoints for:
- Starting new training jobs
- Checking training status and progress
- Downloading trained models
"""

import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ...config import Config
from ...audio.real_data_loader import RealNegativeDataLoader
from ...models.trainer import ASTTrainer, TrainingConfig, calibrate_threshold
from ..jobs import (
    PHASE_DESCRIPTIONS,
    JobInfo,
    JobStatus,
    TrainingProgress,
    get_job_manager,
)
from ..schemas import (
    ErrorResponse,
    JobStatus as SchemaJobStatus,
    ModelType,
    TrainingHyperparameters,
    TrainingStartResponse,
    TrainingStatusResponse,
)

router = APIRouter()


def validate_wake_word(wake_word: str) -> str:
    """Validate and normalize wake word."""
    # Strip and normalize whitespace
    wake_word = " ".join(wake_word.split())

    if len(wake_word) < 2:
        raise ValueError("Wake word must be at least 2 characters")
    if len(wake_word) > 30:
        raise ValueError("Wake word must be at most 30 characters")

    # Check allowed characters
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    if not all(c in allowed for c in wake_word):
        raise ValueError("Wake word must contain only letters and spaces")

    # Check word count
    words = wake_word.split()
    if len(words) > 2:
        raise ValueError("Wake word must be 1 or 2 words")

    return wake_word


def validate_audio_file(audio_data: bytes, filename: str) -> tuple[np.ndarray, int]:
    """
    Validate and load an audio file.

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_data)
        audio, sr = sf.read(audio_io)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Check duration
        duration = len(audio) / sr
        if duration < Config.MIN_DURATION:
            raise ValueError(
                f"Recording too short ({duration:.1f}s). "
                f"Minimum is {Config.MIN_DURATION}s"
            )
        if duration > Config.MAX_DURATION:
            raise ValueError(
                f"Recording too long ({duration:.1f}s). "
                f"Maximum is {Config.MAX_DURATION}s"
            )

        return audio.astype(np.float32), sr

    except Exception as e:
        if "Recording too" in str(e):
            raise
        raise ValueError(f"Invalid audio file '{filename}': {e}") from e


def run_training_job(
    job: JobInfo,
    audio_files: list[tuple[np.ndarray, int]],
    temp_dir: Path,
) -> None:
    """
    Execute the training job using AST-based transfer learning.

    This function runs in a background thread and updates the job status
    as it progresses through the training pipeline.
    """
    try:
        # Phase 1: Validation
        job.update_status(JobStatus.VALIDATING, "Validating audio recordings...")

        if len(audio_files) < Config.MIN_RECORDINGS:
            raise ValueError(
                f"Need at least {Config.MIN_RECORDINGS} recordings, "
                f"got {len(audio_files)}"
            )

        # Phase 2: Setup AST trainer
        hyperparams = job.hyperparameters or {}
        config = TrainingConfig(
            # Classifier settings
            classifier_hidden_dims=hyperparams.get(
                "classifier_hidden_dims", [256, 128]
            ),
            classifier_dropout=hyperparams.get("dropout", 0.5),
            freeze_base=True,  # Always freeze AST base model
            # Training settings
            batch_size=hyperparams.get("batch_size", 32),
            num_epochs=hyperparams.get("num_epochs", 100),
            learning_rate=hyperparams.get("learning_rate", 1e-4),
            weight_decay=hyperparams.get("weight_decay", 1e-3),
            patience=hyperparams.get(
                "patience", hyperparams.get("early_stopping_patience", 8)
            ),
            # Regularization (higher values prevent false positives)
            label_smoothing=hyperparams.get("label_smoothing", 0.25),
            mixup_alpha=hyperparams.get("mixup_alpha", 0.5),
            # Focal loss for hard example mining
            use_focal_loss=hyperparams.get("use_focal_loss", True),
            focal_alpha=hyperparams.get("focal_alpha", 0.25),
            focal_gamma=hyperparams.get("focal_gamma", 2.0),
            # Classifier architecture enhancements
            use_attention=hyperparams.get("use_attention", False),
            use_se_block=hyperparams.get("use_se_block", False),
            use_tcn=hyperparams.get("use_tcn", False),
            # Data settings
            target_positive_samples=hyperparams.get("target_positive_samples", 4000),
            use_tts_positives=hyperparams.get("use_tts_positives", True),
            use_real_negatives=hyperparams.get("use_real_negatives", True),
            max_real_negatives=hyperparams.get("max_real_negatives", 0),
            use_hard_negatives=hyperparams.get(
                "use_hard_negatives", True
            ),  # Critical for accuracy
            # Negative ratios (when max_real_negatives=0) - defaults match UI
            negative_ratio=float(hyperparams.get("negative_ratio", 2.0)),
            hard_negative_ratio=float(hyperparams.get("hard_negative_ratio", 4.0)),
        )

        output_dir = Config.CUSTOM_MODELS_DIR
        trainer = ASTTrainer(config=config, output_dir=output_dir)

        # Phase 3: Data augmentation - Loading AST model
        job.update_status(
            JobStatus.AUGMENTING, "Loading AST model and feature extractor..."
        )

        # Phase 4: Generate negatives (handled in prepare_data)
        # The prepare_data function will update progress via callback
        def progress_callback(message: str, progress_percent: float = 0):
            """Callback to update progress during data preparation."""
            # progress_percent is the direct percentage (0-100) to show
            # Update message and progress directly to avoid update_status overwriting progress
            job.message = message
            job.progress_percent = progress_percent
            job.updated_at = datetime.now()

        job.update_status(JobStatus.GENERATING_NEGATIVES, "Starting data generation...")

        # Prepare data with progress callback
        train_loader, val_loader = trainer.prepare_data(
            positive_audio=audio_files,
            negative_audio=[],  # Will be generated
            wake_word=job.wake_word,
            augment_positive=True,
            progress_callback=progress_callback,
        )

        # Store data stats in job for UI
        if hasattr(trainer, "data_stats") and trainer.data_stats:
            from ..jobs import DataStats

            job.training_progress = job.training_progress or TrainingProgress()
            job.training_progress.data_stats = DataStats(
                num_recordings=trainer.data_stats.get("num_recordings", 0),
                num_positive_samples=trainer.data_stats.get("num_positive_samples", 0),
                num_negative_samples=trainer.data_stats.get("num_negative_samples", 0),
                num_train_samples=trainer.data_stats.get("num_train_samples", 0),
                num_val_samples=trainer.data_stats.get("num_val_samples", 0),
            )
            print(f"[DEBUG] Data stats set: {job.training_progress.data_stats}")

        # Phase 5: Training
        job.update_status(
            JobStatus.TRAINING, "Training classifier on AST embeddings..."
        )

        # Create model and setup training
        trainer.model = trainer.create_model()
        trainer.setup_training(trainer.model, len(train_loader))

        # Custom training loop with progress updates
        import time

        best_model_state = None
        start_time = time.time()

        for epoch in range(config.num_epochs):
            # Train epoch
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = trainer.validate(val_loader)

            # Update trainer metrics
            trainer.metrics.epoch = epoch
            trainer.metrics.train_loss = train_loss
            trainer.metrics.train_acc = train_acc
            trainer.metrics.val_loss = val_metrics["loss"]
            trainer.metrics.val_acc = val_metrics["accuracy"]
            trainer.metrics.val_f1 = val_metrics["f1"]
            trainer.metrics.learning_rate = trainer.optimizer.param_groups[0]["lr"]

            # Save history
            trainer.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "lr": trainer.metrics.learning_rate,
                }
            )

            # Check for improvement
            if val_metrics["loss"] < trainer.metrics.best_val_loss - config.min_delta:
                trainer.metrics.best_val_loss = val_metrics["loss"]
                trainer.metrics.epochs_without_improvement = 0
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in trainer.model.classifier.state_dict().items()
                }
            else:
                trainer.metrics.epochs_without_improvement += 1

            # Update job progress
            job.update_training_progress(
                epoch=epoch,
                total_epochs=config.num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_f1=val_metrics["f1"],
                lr=trainer.metrics.learning_rate,
                best_val_loss=trainer.metrics.best_val_loss,
                epochs_without_improvement=trainer.metrics.epochs_without_improvement,
            )

            # Early stopping
            if trainer.metrics.epochs_without_improvement >= config.patience:
                break

        # Restore best model
        if best_model_state is not None:
            trainer.model.classifier.load_state_dict(best_model_state)

        # Phase 6: Calibration
        job.update_status(JobStatus.CALIBRATING, "Calibrating detection threshold...")

        optimal_threshold, threshold_metrics = calibrate_threshold(
            trainer.model,
            val_loader,
            trainer.device,
        )

        # Phase 7: Save model
        job.update_status(JobStatus.SAVING, "Saving trained model...")

        # Prepare threshold analysis for metadata
        threshold_analysis = [
            {
                "threshold": float(m.threshold),
                "far": float(m.far),
                "frr": float(m.frr),
                "accuracy": float(m.accuracy),
                "precision": float(m.precision),
                "recall": float(m.recall),
                "f1": float(m.f1),
            }
            for m in threshold_metrics[::10]  # Sample every 10th threshold
        ]

        # Build training config for metadata
        training_config = {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "dropout": config.classifier_dropout,
            "label_smoothing": config.label_smoothing,
            "mixup_alpha": config.mixup_alpha,
            "use_focal_loss": config.use_focal_loss,
            "focal_gamma": config.focal_gamma,
            "use_attention": config.use_attention,
            "use_se_block": config.use_se_block,
            "use_tcn": config.use_tcn,
            "classifier_hidden_dims": config.classifier_hidden_dims,
            "weight_decay": config.weight_decay,
        }

        model_dir = trainer.save_model(
            wake_word=job.wake_word,
            threshold=optimal_threshold,
            metadata={
                "threshold_analysis": threshold_analysis,
                "training_time_seconds": time.time() - start_time,
                "base_model": "MIT/ast-finetuned-speech-commands-v2",
                "training_config": training_config,
            },
        )

        # Mark completed
        job.mark_completed(
            model_path=model_dir,
            metadata={
                "threshold": optimal_threshold,
                "val_accuracy": float(trainer.metrics.val_acc),
                "val_f1": float(trainer.metrics.val_f1),
                "epochs_trained": trainer.metrics.epoch + 1,
                "base_model": "MIT/ast-finetuned-speech-commands-v2",
            },
        )

    except Exception as e:
        import traceback

        job.mark_failed(str(e), traceback.format_exc())

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Negative Data Cache Endpoints
# ============================================================================


@router.get(
    "/negative-cache/info",
    summary="Get Negative Data Cache Info",
    description="Get information about cached negative audio chunks.",
)
async def get_negative_cache_info():
    """Get cache status and chunk count."""
    loader = RealNegativeDataLoader()
    cache_info = loader.get_cache_info()
    file_counts = loader.get_file_count()

    return {
        "cached": cache_info["cached"],
        "chunk_count": cache_info["chunk_count"],
        "source_files": file_counts["total"],
        "file_counts": file_counts,
        "created_at": cache_info.get("created_at"),
    }


@router.post(
    "/negative-cache/build",
    summary="Build Negative Data Cache",
    description="Pre-process all negative audio files and cache chunks for fast loading.",
)
async def build_negative_cache(
    max_workers: int = 4,
):
    """Build the negative data cache."""
    import threading

    loader = RealNegativeDataLoader()

    if not loader.available:
        raise HTTPException(
            status_code=404, detail="No negative data found in data/negative/ directory"
        )

    # Run in background thread
    def build():
        loader.build_cache(max_workers=max_workers)

    thread = threading.Thread(target=build, daemon=True)
    thread.start()

    file_counts = loader.get_file_count()
    return {
        "message": "Cache building started in background",
        "source_files": file_counts["total"],
        "estimated_chunks": file_counts["total"] * 5,  # ~5 chunks per file
    }


@router.delete(
    "/negative-cache",
    summary="Clear Negative Data Cache",
    description="Delete all cached negative audio chunks.",
)
async def clear_negative_cache():
    """Clear the cache."""
    loader = RealNegativeDataLoader()
    loader.clear_cache()
    return {"message": "Cache cleared"}


@router.post(
    "/start",
    response_model=TrainingStartResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Server busy"},
    },
    summary="Start Training Job",
    description="""
Start a new wake word training job.

Upload 3-5 audio recordings of the wake word along with the wake word text
and model configuration. The training will run in the background.

**Audio Requirements:**
- Format: WAV, FLAC, or OGG
- Duration: 0.5-3.0 seconds per recording
- Minimum recordings: 3
- Maximum recordings: 5

**Model:**
Uses Audio Spectrogram Transformer (AST) with transfer learning from
MIT/ast-finetuned-speech-commands-v2. The base model is frozen and only
the classifier head is trained.
""",
)
async def start_training(
    wake_word: Annotated[str, Form(description="The wake word to train (1-2 words)")],
    recordings: Annotated[
        list[UploadFile], File(description="Audio recordings of the wake word (3-5)")
    ],
    model_type: Annotated[
        ModelType, Form(description="Model architecture (AST recommended)")
    ] = ModelType.AST,
    batch_size: Annotated[
        Optional[int], Form(description="Training batch size", ge=8, le=256)
    ] = None,
    num_epochs: Annotated[
        Optional[int], Form(description="Maximum training epochs", ge=10, le=500)
    ] = None,
    learning_rate: Annotated[
        Optional[float], Form(description="Initial learning rate", gt=0, le=0.1)
    ] = None,
    # Regularization hyperparameters (balanced defaults)
    dropout: Annotated[
        Optional[float],
        Form(description="Dropout probability (default: 0.5)", ge=0, le=0.7),
    ] = None,
    label_smoothing: Annotated[
        Optional[float],
        Form(description="Label smoothing factor (default: 0.25)", ge=0, le=0.4),
    ] = None,
    weight_decay: Annotated[
        Optional[float],
        Form(description="L2 regularization strength (default: 0.001)", ge=0, le=0.5),
    ] = None,
    mixup_alpha: Annotated[
        Optional[float],
        Form(description="Mixup augmentation strength (default: 0.5)", ge=0, le=1.0),
    ] = None,
    # Model enhancements
    use_focal_loss: Annotated[
        Optional[bool],
        Form(description="Use focal loss for hard example mining (default: true)"),
    ] = None,
    focal_alpha: Annotated[
        Optional[float],
        Form(
            description="Focal loss alpha - weight for positive class (default: 0.25)",
            ge=0.1,
            le=0.9,
        ),
    ] = None,
    focal_gamma: Annotated[
        Optional[float],
        Form(description="Focal loss gamma (default: 2.0)", ge=0.5, le=5.0),
    ] = None,
    use_attention: Annotated[
        Optional[bool],
        Form(description="Use self-attention in classifier (default: false)"),
    ] = None,
    use_se_block: Annotated[
        Optional[bool],
        Form(
            description="Use Squeeze-and-Excitation block for channel attention (default: false)"
        ),
    ] = None,
    use_tcn: Annotated[
        Optional[bool],
        Form(description="Use Temporal Convolutional Network block (default: false)"),
    ] = None,
    classifier_hidden_dims: Annotated[
        Optional[str],
        Form(description="Classifier hidden dims as JSON array (default: [256, 128])"),
    ] = None,
    # Data generation settings
    target_positive_samples: Annotated[
        Optional[int],
        Form(description="Target positive samples (default: 6000)", ge=100, le=20000),
    ] = None,
    use_tts_positives: Annotated[
        Optional[bool],
        Form(description="Generate TTS positive samples (default: true)"),
    ] = None,
    use_real_negatives: Annotated[
        Optional[bool],
        Form(description="Use real negative data from data/negative/ (default: true)"),
    ] = None,
    max_real_negatives: Annotated[
        Optional[int],
        Form(description="Max real negative samples (0 = no limit)", ge=0, le=100000),
    ] = None,
    use_hard_negatives: Annotated[
        Optional[bool],
        Form(
            description="Generate hard negatives from similar-sounding words (default: true)"
        ),
    ] = None,
    negative_ratio: Annotated[
        Optional[float],
        Form(
            description="Negative:Positive ratio for real negatives (default: 2.0)",
            ge=0.5,
            le=10.0,
        ),
    ] = None,
    hard_negative_ratio: Annotated[
        Optional[float],
        Form(
            description="Hard negative:Positive ratio (default: 4.0)", ge=0.5, le=10.0
        ),
    ] = None,
) -> TrainingStartResponse:
    """
    Start a new training job.

    This endpoint accepts the wake word text and audio recordings,
    validates them, and starts a background training job.
    """
    job_manager = get_job_manager()

    # Check if we can start a new job
    if not job_manager.can_start_job():
        raise HTTPException(
            status_code=503,
            detail="Server is busy with another training job. Please try again later.",
        )

    # Validate wake word
    try:
        wake_word = validate_wake_word(wake_word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Validate number of recordings
    if len(recordings) < Config.MIN_RECORDINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {Config.MIN_RECORDINGS} recordings, got {len(recordings)}",
        )
    if len(recordings) > Config.MAX_RECORDINGS:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {Config.MAX_RECORDINGS} recordings allowed, got {len(recordings)}",
        )

    # Create temp directory for audio files
    temp_dir = Path(tempfile.mkdtemp(prefix="wakebuilder_"))

    # Load and validate audio files
    audio_files: list[tuple[np.ndarray, int]] = []
    raw_recordings: list[tuple[bytes, str]] = []  # Store raw bytes for saving
    try:
        for i, recording in enumerate(recordings):
            content = await recording.read()
            filename = recording.filename or f"recording_{i}.wav"

            try:
                audio, sr = validate_audio_file(content, filename)
                audio_files.append((audio, sr))
                raw_recordings.append((content, filename))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    # Save recordings to model-specific directory
    model_id = wake_word.lower().replace(" ", "_")
    recordings_dir = Config.RECORDINGS_DIR / model_id
    recordings_dir.mkdir(parents=True, exist_ok=True)

    for i, (content, filename) in enumerate(raw_recordings):
        # Use consistent naming: recording_001.wav, recording_002.wav, etc.
        ext = Path(filename).suffix or ".wav"
        save_path = recordings_dir / f"recording_{i+1:03d}{ext}"
        with open(save_path, "wb") as f:
            f.write(content)

    # Build hyperparameters with balanced defaults
    hyperparameters = {}
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if num_epochs is not None:
        hyperparameters["num_epochs"] = num_epochs
    if learning_rate is not None:
        hyperparameters["learning_rate"] = learning_rate
    # Regularization hyperparameters
    if dropout is not None:
        hyperparameters["dropout"] = dropout
    if label_smoothing is not None:
        hyperparameters["label_smoothing"] = label_smoothing
    if weight_decay is not None:
        hyperparameters["weight_decay"] = weight_decay
    if mixup_alpha is not None:
        hyperparameters["mixup_alpha"] = mixup_alpha
    # Model enhancements
    if use_focal_loss is not None:
        hyperparameters["use_focal_loss"] = use_focal_loss
    if focal_alpha is not None:
        hyperparameters["focal_alpha"] = focal_alpha
    if focal_gamma is not None:
        hyperparameters["focal_gamma"] = focal_gamma
    if use_attention is not None:
        hyperparameters["use_attention"] = use_attention
    if use_se_block is not None:
        hyperparameters["use_se_block"] = use_se_block
    if use_tcn is not None:
        hyperparameters["use_tcn"] = use_tcn
    if classifier_hidden_dims is not None:
        try:
            hyperparameters["classifier_hidden_dims"] = json.loads(
                classifier_hidden_dims
            )
        except json.JSONDecodeError:
            pass  # Use default if parsing fails
    # Data generation settings
    if target_positive_samples is not None:
        hyperparameters["target_positive_samples"] = target_positive_samples
    if use_tts_positives is not None:
        hyperparameters["use_tts_positives"] = use_tts_positives
    if use_real_negatives is not None:
        hyperparameters["use_real_negatives"] = use_real_negatives
    if max_real_negatives is not None:
        hyperparameters["max_real_negatives"] = max_real_negatives
    if use_hard_negatives is not None:
        hyperparameters["use_hard_negatives"] = use_hard_negatives
    if negative_ratio is not None:
        hyperparameters["negative_ratio"] = negative_ratio
    if hard_negative_ratio is not None:
        hyperparameters["hard_negative_ratio"] = hard_negative_ratio

    # Create job
    job = job_manager.create_job(
        wake_word=wake_word,
        model_type=model_type.value,
        hyperparameters=hyperparameters if hyperparameters else None,
    )

    # Start training in background
    def training_task(job: JobInfo) -> None:
        run_training_job(job, audio_files, temp_dir)

    job_manager.start_job(job.job_id, training_task)

    return TrainingStartResponse(
        job_id=job.job_id,
        message="Training job started successfully",
        wake_word=wake_word,
        model_type=model_type.value,
    )


@router.get(
    "/status/{job_id}",
    response_model=TrainingStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
    summary="Get Training Status",
    description="""
Get the current status of a training job.

Returns detailed progress information including:
- Current phase and progress percentage
- Training metrics (loss, accuracy, F1) per epoch
- Hyperparameters used
- Error information if failed
""",
)
async def get_training_status(job_id: str) -> TrainingStatusResponse:
    """
    Get the status of a training job.

    Poll this endpoint to track training progress.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Convert job to response
    from ..schemas import EpochMetrics as EpochMetricsSchema
    from ..schemas import TrainingProgress as TrainingProgressSchema
    from ..schemas import DataStats as DataStatsSchema

    training_progress = None
    if job.training_progress:
        data_stats = None
        if job.training_progress.data_stats:
            ds = job.training_progress.data_stats
            data_stats = DataStatsSchema(
                num_recordings=ds.num_recordings,
                num_positive_samples=ds.num_positive_samples,
                num_negative_samples=ds.num_negative_samples,
                num_train_samples=ds.num_train_samples,
                num_val_samples=ds.num_val_samples,
            )
        else:
            print("[DEBUG] No data_stats in training_progress")

        # Handle inf values that can't be JSON serialized
        best_val_loss = job.training_progress.best_val_loss
        if (
            best_val_loss == float("inf") or best_val_loss != best_val_loss
        ):  # Check for inf or nan
            best_val_loss = 999.0  # Use a large but valid number

        training_progress = TrainingProgressSchema(
            current_epoch=job.training_progress.current_epoch,
            total_epochs=job.training_progress.total_epochs,
            best_val_loss=best_val_loss,
            epochs_without_improvement=job.training_progress.epochs_without_improvement,
            epoch_history=[
                EpochMetricsSchema(
                    epoch=m.epoch,
                    train_loss=m.train_loss,
                    train_accuracy=m.train_accuracy,
                    val_loss=m.val_loss,
                    val_accuracy=m.val_accuracy,
                    val_f1=m.val_f1,
                    learning_rate=m.learning_rate,
                )
                for m in job.training_progress.epoch_history
            ],
            data_stats=data_stats,
        )

    hyperparameters = None
    if job.hyperparameters:
        hyperparameters = TrainingHyperparameters(**job.hyperparameters)

    # Convert job status to schema status (they have same values)
    schema_status = SchemaJobStatus(job.status.value)

    # Use dynamic message if available, otherwise fall back to static phase description
    current_phase = (
        job.message
        if job.message
        else PHASE_DESCRIPTIONS.get(job.status, str(job.status))
    )

    return TrainingStatusResponse(
        job_id=job.job_id,
        status=schema_status,
        progress_percent=job.progress_percent,
        current_phase=current_phase,
        message=job.message,
        wake_word=job.wake_word,
        model_type=job.model_type,
        started_at=job.started_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        error=job.error,
        training_progress=training_progress,
        hyperparameters=hyperparameters,
    )


@router.get(
    "/download/{job_id}",
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "Model files as ZIP archive",
        },
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"},
    },
    summary="Download Trained Model",
    description="""
Download the trained model files for a completed job.

Returns a ZIP archive containing:
- `model.pt`: PyTorch model weights
- `metadata.json`: Model metadata and configuration
- `training_history.json`: Training metrics history
""",
)
async def download_model(job_id: str) -> StreamingResponse:
    """
    Download the trained model for a completed job.

    Returns a ZIP file containing the model and metadata.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job.status.value}",
        )

    if not job.model_path or not job.model_path.exists():
        raise HTTPException(status_code=404, detail="Model files not found")

    # Create ZIP archive
    import zipfile

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in job.model_path.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)

    # Generate filename
    model_name = job.wake_word.lower().replace(" ", "_")
    filename = f"{model_name}_model.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete(
    "/{job_id}",
    responses={
        200: {"description": "Job deleted successfully"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Cannot delete active job"},
    },
    summary="Delete Training Job",
    description="Delete a completed or failed training job from the job list.",
)
async def delete_job(job_id: str) -> dict:
    """
    Delete a training job.

    Only completed or failed jobs can be deleted.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an active job. Wait for it to complete or fail.",
        )

    job_manager.delete_job(job_id)

    return {"message": "Job deleted successfully", "job_id": job_id}


@router.get(
    "/jobs",
    summary="List All Jobs",
    description="Get a list of all training jobs (active and completed).",
)
async def list_jobs() -> dict:
    """
    List all training jobs.

    Returns both active and completed jobs.
    """
    job_manager = get_job_manager()
    jobs = job_manager.get_all_jobs()

    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": len(jobs),
        "active": len(
            [j for j in jobs if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED)]
        ),
    }
