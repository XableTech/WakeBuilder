"""
Training endpoints for WakeBuilder API.

This module provides endpoints for:
- Starting new training jobs
- Checking training status and progress
- Downloading trained models
"""

import io
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ...config import Config
from ...models.trainer import Trainer, TrainingConfig, calibrate_threshold
from ..jobs import PHASE_DESCRIPTIONS, JobInfo, JobStatus, TrainingProgress, get_job_manager
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
    Execute the training job.

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

        # Phase 2: Setup trainer with improved defaults
        hyperparams = job.hyperparameters or {}
        config = TrainingConfig(
            model_type=job.model_type,
            batch_size=hyperparams.get("batch_size", 32),
            num_epochs=hyperparams.get("num_epochs", 150),
            learning_rate=hyperparams.get("learning_rate", 3e-4),
            dropout=hyperparams.get("dropout", 0.4),
            patience=hyperparams.get("early_stopping_patience", 25),
            # New parameters for improved training
            weight_decay=hyperparams.get("weight_decay", 1e-2),
            negative_class_weight=hyperparams.get("negative_class_weight", 2.0),
            spec_augment=hyperparams.get("spec_augment", True),
            mixup_alpha=hyperparams.get("mixup_alpha", 0.4),
        )

        output_dir = Config.CUSTOM_MODELS_DIR
        trainer = Trainer(config=config, output_dir=output_dir)

        # Phase 3: Data augmentation
        job.update_status(JobStatus.AUGMENTING, "Creating voice variations...")

        # Phase 4: Generate negatives (handled in prepare_data)
        job.update_status(
            JobStatus.GENERATING_NEGATIVES, "Generating negative examples..."
        )

        # Prepare data
        train_loader, val_loader = trainer.prepare_data(
            positive_audio=audio_files,
            negative_audio=[],  # Will be generated
            wake_word=job.wake_word,
            augment_positive=True,
            generate_negatives=True,
        )

        # Store data stats in job for UI
        if hasattr(trainer, 'data_stats') and trainer.data_stats:
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
        job.update_status(JobStatus.TRAINING, "Training the model...")

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
                best_model_state = trainer.model.state_dict().copy()
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
            trainer.model.load_state_dict(best_model_state)

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
                "threshold": m.threshold,
                "far": m.far,
                "frr": m.frr,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
            }
            for m in threshold_metrics[::10]  # Sample every 10th threshold
        ]

        model_dir = trainer.save_model(
            wake_word=job.wake_word,
            threshold=optimal_threshold,
            metadata={
                "threshold_analysis": threshold_analysis,
                "training_time_seconds": time.time() - start_time,
            },
        )

        # Mark completed
        job.mark_completed(
            model_path=model_dir,
            metadata={
                "threshold": optimal_threshold,
                "val_accuracy": trainer.metrics.val_acc,
                "val_f1": trainer.metrics.val_f1,
                "epochs_trained": trainer.metrics.epoch + 1,
            },
        )

    except Exception as e:
        import traceback

        job.mark_failed(str(e), traceback.format_exc())

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


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

**Model Types:**
- `tc_resnet`: Fast inference (~0.6ms), good for production
- `bc_resnet`: Higher accuracy, recommended for best results
""",
)
async def start_training(
    wake_word: Annotated[str, Form(description="The wake word to train (1-2 words)")],
    recordings: Annotated[
        list[UploadFile], File(description="Audio recordings of the wake word (3-5)")
    ],
    model_type: Annotated[
        ModelType, Form(description="Model architecture to use")
    ] = ModelType.BC_RESNET,
    batch_size: Annotated[
        Optional[int], Form(description="Training batch size", ge=8, le=256)
    ] = None,
    num_epochs: Annotated[
        Optional[int], Form(description="Maximum training epochs", ge=10, le=500)
    ] = None,
    learning_rate: Annotated[
        Optional[float], Form(description="Initial learning rate", gt=0, le=0.1)
    ] = None,
    # New hyperparameters for improved training
    dropout: Annotated[
        Optional[float], Form(description="Dropout probability", ge=0, le=0.7)
    ] = None,
    negative_class_weight: Annotated[
        Optional[float], Form(description="Weight for negative class (higher = fewer false positives)", ge=1.0, le=5.0)
    ] = None,
    spec_augment: Annotated[
        Optional[bool], Form(description="Enable SpecAugment (time/frequency masking)")
    ] = None,
    weight_decay: Annotated[
        Optional[float], Form(description="L2 regularization strength", ge=0, le=0.5)
    ] = None,
    mixup_alpha: Annotated[
        Optional[float], Form(description="Mixup augmentation strength", ge=0, le=1.0)
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
    try:
        for i, recording in enumerate(recordings):
            content = await recording.read()
            filename = recording.filename or f"recording_{i}.wav"

            try:
                audio, sr = validate_audio_file(content, filename)
                audio_files.append((audio, sr))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    # Build hyperparameters
    hyperparameters = {}
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if num_epochs is not None:
        hyperparameters["num_epochs"] = num_epochs
    if learning_rate is not None:
        hyperparameters["learning_rate"] = learning_rate
    # New hyperparameters
    if dropout is not None:
        hyperparameters["dropout"] = dropout
    if negative_class_weight is not None:
        hyperparameters["negative_class_weight"] = negative_class_weight
    if spec_augment is not None:
        hyperparameters["spec_augment"] = spec_augment
    if weight_decay is not None:
        hyperparameters["weight_decay"] = weight_decay
    if mixup_alpha is not None:
        hyperparameters["mixup_alpha"] = mixup_alpha

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
            print(f"[DEBUG] No data_stats in training_progress")
        
        # Handle inf values that can't be JSON serialized
        best_val_loss = job.training_progress.best_val_loss
        if best_val_loss == float('inf') or best_val_loss != best_val_loss:  # Check for inf or nan
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

    return TrainingStatusResponse(
        job_id=job.job_id,
        status=schema_status,
        progress_percent=job.progress_percent,
        current_phase=PHASE_DESCRIPTIONS.get(job.status, str(job.status)),
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
