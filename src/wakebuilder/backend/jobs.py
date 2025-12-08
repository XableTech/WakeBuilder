"""
Background job management for WakeBuilder.

This module provides a threading-based job queue for managing long-running
training tasks. Jobs are tracked in memory with status updates that can
be polled by the frontend.
"""

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class JobStatus(str, Enum):
    """Status of a background job."""

    PENDING = "pending"
    VALIDATING = "validating"
    AUGMENTING = "augmenting"
    GENERATING_NEGATIVES = "generating_negatives"
    TRAINING = "training"
    CALIBRATING = "calibrating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


# Human-readable phase descriptions for the UI
PHASE_DESCRIPTIONS = {
    JobStatus.PENDING: "Waiting to start...",
    JobStatus.VALIDATING: "Validating audio recordings...",
    JobStatus.AUGMENTING: "Creating voice variations...",
    JobStatus.GENERATING_NEGATIVES: "Generating and Loading data...",
    JobStatus.TRAINING: "Training the model...",
    JobStatus.CALIBRATING: "Calibrating detection threshold...",
    JobStatus.SAVING: "Saving trained model...",
    JobStatus.COMPLETED: "Training complete!",
    JobStatus.FAILED: "Training failed",
}

# Progress percentages for each phase
PHASE_PROGRESS = {
    JobStatus.PENDING: 0,
    JobStatus.VALIDATING: 5,
    JobStatus.AUGMENTING: 15,
    JobStatus.GENERATING_NEGATIVES: 30,
    JobStatus.TRAINING: 45,  # Training goes from 45-90%
    JobStatus.CALIBRATING: 92,
    JobStatus.SAVING: 97,
    JobStatus.COMPLETED: 100,
    JobStatus.FAILED: 0,
}


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    learning_rate: float


@dataclass
class DataStats:
    """Statistics about training data."""
    
    num_recordings: int = 0
    num_positive_samples: int = 0
    num_negative_samples: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0


@dataclass
class TrainingProgress:
    """Detailed training progress information."""

    current_epoch: int = 0
    total_epochs: int = 100
    best_val_loss: float = 999.0  # Use large but JSON-serializable value instead of inf
    epochs_without_improvement: int = 0
    epoch_history: list[EpochMetrics] = field(default_factory=list)
    data_stats: Optional[DataStats] = None


@dataclass
class JobInfo:
    """Information about a training job."""

    job_id: str
    wake_word: str
    model_type: str
    status: JobStatus = JobStatus.PENDING
    progress_percent: float = 0.0
    message: str = "Job created"
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    model_path: Optional[Path] = None
    training_progress: Optional[TrainingProgress] = None
    hyperparameters: Optional[dict[str, Any]] = None
    result_metadata: Optional[dict[str, Any]] = None

    def update_status(
        self,
        status: JobStatus,
        message: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Update job status with optional message and progress."""
        self.status = status
        self.updated_at = datetime.now()

        if message:
            self.message = message
        else:
            self.message = PHASE_DESCRIPTIONS.get(status, str(status))

        if progress is not None:
            self.progress_percent = progress
        else:
            self.progress_percent = PHASE_PROGRESS.get(status, 0)

    def update_training_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        lr: float,
        best_val_loss: float,
        epochs_without_improvement: int,
    ) -> None:
        """Update detailed training progress."""
        # Preserve existing data_stats if training_progress exists
        existing_data_stats = None
        if self.training_progress is not None:
            existing_data_stats = self.training_progress.data_stats
            if epoch == 0:
                print(f"[DEBUG] Epoch 0: existing data_stats = {existing_data_stats}")
        
        if self.training_progress is None:
            self.training_progress = TrainingProgress()
            print(f"[DEBUG] Created new TrainingProgress")
        
        # Restore data_stats if it was set before
        if existing_data_stats is not None:
            self.training_progress.data_stats = existing_data_stats

        self.training_progress.current_epoch = epoch
        self.training_progress.total_epochs = total_epochs
        self.training_progress.best_val_loss = best_val_loss
        self.training_progress.epochs_without_improvement = epochs_without_improvement

        # Add epoch metrics to history
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_f1=val_f1,
            learning_rate=lr,
        )
        self.training_progress.epoch_history.append(epoch_metrics)

        # Update overall progress (training phase is 45-90%)
        training_progress = (epoch + 1) / total_epochs
        self.progress_percent = 45 + (training_progress * 45)
        self.message = (
            f"Training epoch {epoch + 1}/{total_epochs} "
            f"(val_loss: {val_loss:.4f}, val_acc: {val_acc:.1%})"
        )
        self.updated_at = datetime.now()

    def mark_completed(
        self,
        model_path: Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.progress_percent = 100
        self.message = "Training complete!"
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.model_path = model_path
        self.result_metadata = metadata

    def mark_failed(self, error: str, traceback_str: Optional[str] = None) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.message = f"Training failed: {error}"
        self.error = error
        self.error_traceback = traceback_str
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "job_id": self.job_id,
            "wake_word": self.wake_word,
            "model_type": self.model_type,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "current_phase": PHASE_DESCRIPTIONS.get(self.status, str(self.status)),
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error": self.error,
            "hyperparameters": self.hyperparameters,
        }

        if self.training_progress:
            result["training_progress"] = {
                "current_epoch": self.training_progress.current_epoch,
                "total_epochs": self.training_progress.total_epochs,
                "best_val_loss": self.training_progress.best_val_loss,
                "epochs_without_improvement": self.training_progress.epochs_without_improvement,
                "epoch_history": [
                    {
                        "epoch": m.epoch,
                        "train_loss": m.train_loss,
                        "train_accuracy": m.train_accuracy,
                        "val_loss": m.val_loss,
                        "val_accuracy": m.val_accuracy,
                        "val_f1": m.val_f1,
                        "learning_rate": m.learning_rate,
                    }
                    for m in self.training_progress.epoch_history
                ],
            }

        return result


class JobManager:
    """
    Thread-safe manager for background training jobs.

    Provides methods to create, track, and manage training jobs.
    Jobs are stored in memory and will be lost on restart.
    """

    def __init__(self, max_concurrent_jobs: int = 1) -> None:
        """
        Initialize the job manager.

        Args:
            max_concurrent_jobs: Maximum number of concurrent training jobs
        """
        self._jobs: dict[str, JobInfo] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent_jobs

    def create_job(
        self,
        wake_word: str,
        model_type: str,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> JobInfo:
        """
        Create a new training job.

        Args:
            wake_word: The wake word to train
            model_type: Model architecture to use
            hyperparameters: Optional training hyperparameters

        Returns:
            JobInfo for the created job
        """
        job_id = str(uuid.uuid4())

        job = JobInfo(
            job_id=job_id,
            wake_word=wake_word,
            model_type=model_type,
            hyperparameters=hyperparameters,
        )

        with self._lock:
            self._jobs[job_id] = job

        return job

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> list[JobInfo]:
        """Get all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_active_jobs(self) -> list[JobInfo]:
        """Get all active (non-completed, non-failed) jobs."""
        with self._lock:
            return [
                job
                for job in self._jobs.values()
                if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]

    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        active = self.get_active_jobs()
        return len(active) < self._max_concurrent

    def start_job(
        self,
        job_id: str,
        task_func: Callable[[JobInfo], None],
    ) -> bool:
        """
        Start a job in a background thread.

        Args:
            job_id: ID of the job to start
            task_func: Function to execute (receives JobInfo as argument)

        Returns:
            True if job was started, False otherwise
        """
        job = self.get_job(job_id)
        if not job:
            return False

        # Check if this job is already running
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return False

        def run_task() -> None:
            try:
                task_func(job)
            except Exception as e:
                job.mark_failed(str(e), traceback.format_exc())

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

        with self._lock:
            self._threads[job_id] = thread

        return True

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job (only if completed or failed).

        Args:
            job_id: ID of the job to delete

        Returns:
            True if job was deleted, False otherwise
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                return False

            del self._jobs[job_id]
            self._threads.pop(job_id, None)

        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove completed/failed jobs older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs removed
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0

        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if job.completed_at and job.completed_at < cutoff:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                self._threads.pop(job_id, None)
                removed += 1

        return removed


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
