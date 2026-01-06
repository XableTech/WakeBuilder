"""
Tests for WakeBuilder FastAPI backend.

This module tests:
- Health and info endpoints
- Training endpoints (start, status, download)
- Model management endpoints (list, metadata, delete)
- File-based testing endpoint
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from wakebuilder import __version__
from wakebuilder.backend.jobs import JobManager, JobStatus
from wakebuilder.backend.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def job_manager() -> JobManager:
    """Get a fresh job manager for testing."""
    manager = JobManager()
    return manager


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate sample audio as bytes for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Generate a simple sine wave
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Convert to bytes (WAV format)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory with test model using AST."""
    import torch

    from wakebuilder.models.classifier import WakeWordClassifier

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # Create a simple classifier (not the full AST model for faster tests)
    classifier = WakeWordClassifier(
        embedding_dim=768,
        hidden_dims=[256, 128],
        dropout=0.3,
    )

    # Save classifier state dict in the format used by the app
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "wake_word": "Test Word",
            "threshold": 0.5,
            "embedding_dim": 768,
            "classifier_hidden_dims": [256, 128],
            "model_version": "2.0",
        },
        model_dir / "model.pt",
    )

    # Save metadata
    metadata = {
        "wake_word": "Test Word",
        "threshold": 0.5,
        "model_type": "ast",
        "parameters": 64000,
        "metrics": {"val_accuracy": 0.95, "val_f1": 0.93},
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return model_dir


# ============================================================================
# Health and Info Endpoint Tests
# ============================================================================


class TestHealthEndpoints:
    """Tests for health and info endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == __version__
        assert "timestamp" in data

    def test_api_info(self, client: TestClient) -> None:
        """Test API info endpoint returns system information."""
        response = client.get("/api/info")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "WakeBuilder API"
        assert data["version"] == __version__
        assert "system" in data
        assert "python_version" in data["system"]
        assert "torch_version" in data["system"]
        assert "cuda_available" in data["system"]

    def test_root_serves_frontend_or_docs(self, client: TestClient) -> None:
        """Test root endpoint serves frontend (200) or redirects to docs (307)."""
        response = client.get("/", follow_redirects=False)
        # Returns 200 if frontend exists, 307 redirect if not
        assert response.status_code in [200, 307]
        if response.status_code == 307:
            assert response.headers["location"] == "/docs"

    def test_openapi_schema_available(self, client: TestClient) -> None:
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "WakeBuilder API"


# ============================================================================
# Job Manager Tests
# ============================================================================


class TestJobManager:
    """Tests for the background job manager."""

    def test_create_job(self, job_manager: JobManager) -> None:
        """Test creating a new job."""
        job = job_manager.create_job(
            wake_word="Test Word",
            model_type="bc_resnet",
        )

        assert job.job_id is not None
        assert job.wake_word == "Test Word"
        assert job.model_type == "bc_resnet"
        assert job.status == JobStatus.PENDING

    def test_get_job(self, job_manager: JobManager) -> None:
        """Test retrieving a job by ID."""
        job = job_manager.create_job("Test", "tc_resnet")

        retrieved = job_manager.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_job(self, job_manager: JobManager) -> None:
        """Test retrieving a non-existent job returns None."""
        result = job_manager.get_job("nonexistent-id")
        assert result is None

    def test_job_status_update(self, job_manager: JobManager) -> None:
        """Test updating job status."""
        job = job_manager.create_job("Test", "bc_resnet")

        job.update_status(JobStatus.TRAINING, "Training in progress...")
        assert job.status == JobStatus.TRAINING
        assert job.message == "Training in progress..."

    def test_job_completion(self, job_manager: JobManager, tmp_path: Path) -> None:
        """Test marking a job as completed."""
        job = job_manager.create_job("Test", "bc_resnet")
        model_path = tmp_path / "model"
        model_path.mkdir()

        job.mark_completed(model_path, {"accuracy": 0.95})

        assert job.status == JobStatus.COMPLETED
        assert job.model_path == model_path
        assert job.completed_at is not None

    def test_job_failure(self, job_manager: JobManager) -> None:
        """Test marking a job as failed."""
        job = job_manager.create_job("Test", "bc_resnet")

        job.mark_failed("Test error", "Traceback...")

        assert job.status == JobStatus.FAILED
        assert job.error == "Test error"
        assert job.completed_at is not None

    def test_can_start_job(self, job_manager: JobManager) -> None:
        """Test checking if a new job can be started."""
        assert job_manager.can_start_job() is True

    def test_delete_completed_job(self, job_manager: JobManager) -> None:
        """Test deleting a completed job."""
        job = job_manager.create_job("Test", "bc_resnet")
        job.mark_failed("Test error")

        result = job_manager.delete_job(job.job_id)
        assert result is True
        assert job_manager.get_job(job.job_id) is None

    def test_cannot_delete_active_job(self, job_manager: JobManager) -> None:
        """Test that active jobs cannot be deleted."""
        job = job_manager.create_job("Test", "bc_resnet")
        job.update_status(JobStatus.TRAINING)

        result = job_manager.delete_job(job.job_id)
        assert result is False
        assert job_manager.get_job(job.job_id) is not None


# ============================================================================
# Training Endpoint Tests
# ============================================================================


class TestTrainingEndpoints:
    """Tests for training endpoints."""

    def test_start_training_validation_error_short_wake_word(
        self, client: TestClient, sample_audio_bytes: bytes
    ) -> None:
        """Test that short wake words are rejected."""
        files = [
            ("recordings", ("audio1.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio2.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio3.wav", sample_audio_bytes, "audio/wav")),
        ]

        response = client.post(
            "/api/train/start",
            data={"wake_word": "A"},  # Too short
            files=files,
        )

        assert response.status_code == 400
        assert "at least 2 characters" in response.json()["detail"]

    def test_start_training_validation_error_invalid_chars(
        self, client: TestClient, sample_audio_bytes: bytes
    ) -> None:
        """Test that wake words with invalid characters are rejected."""
        files = [
            ("recordings", ("audio1.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio2.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio3.wav", sample_audio_bytes, "audio/wav")),
        ]

        response = client.post(
            "/api/train/start",
            data={"wake_word": "Test123!"},  # Invalid characters
            files=files,
        )

        assert response.status_code == 400
        assert "only letters and spaces" in response.json()["detail"]

    def test_start_training_one_recording_allowed(
        self, client: TestClient, sample_audio_bytes: bytes
    ) -> None:
        """Test that one recording is now allowed (min_recordings=1)."""
        files = [
            ("recordings", ("audio1.wav", sample_audio_bytes, "audio/wav")),
        ]

        response = client.post(
            "/api/train/start",
            data={"wake_word": "Test Word"},
            files=files,
        )

        # Should accept the request (200) or fail due to missing TTS/dependencies (500/503)
        # but NOT reject due to too few recordings (400)
        assert response.status_code in [200, 500, 503]

    def test_get_status_not_found(self, client: TestClient) -> None:
        """Test getting status for non-existent job."""
        response = client.get("/api/train/status/nonexistent-job-id")
        assert response.status_code == 404

    def test_list_jobs(self, client: TestClient) -> None:
        """Test listing all jobs."""
        response = client.get("/api/train/jobs")
        assert response.status_code == 200

        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert "active" in data


# ============================================================================
# Model Management Endpoint Tests
# ============================================================================


class TestModelEndpoints:
    """Tests for model management endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        """Test listing all models."""
        response = client.get("/api/models/list")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total_count" in data
        assert "default_count" in data
        assert "custom_count" in data

    def test_list_models_filter_by_category(self, client: TestClient) -> None:
        """Test filtering models by category."""
        response = client.get("/api/models/list?category=custom")
        assert response.status_code == 200

        data = response.json()
        # All returned models should be custom
        for model in data["models"]:
            assert model["category"] == "custom"

    def test_get_model_metadata_not_found(self, client: TestClient) -> None:
        """Test getting metadata for non-existent model."""
        response = client.get("/api/models/nonexistent-model/metadata")
        assert response.status_code == 404

    def test_delete_model_not_found(self, client: TestClient) -> None:
        """Test deleting non-existent model."""
        response = client.delete("/api/models/nonexistent-model")
        assert response.status_code == 404

    def test_download_model_not_found(self, client: TestClient) -> None:
        """Test downloading non-existent model."""
        response = client.get("/api/models/nonexistent-model/download")
        assert response.status_code == 404


# ============================================================================
# Testing Endpoint Tests
# ============================================================================


class TestTestingEndpoints:
    """Tests for model testing endpoints."""

    def test_list_testable_models(self, client: TestClient) -> None:
        """Test listing models available for testing."""
        response = client.get("/api/test/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total" in data

    def test_test_file_model_not_found(
        self, client: TestClient, sample_audio_bytes: bytes
    ) -> None:
        """Test file testing with non-existent model."""
        response = client.post(
            "/api/test/file",
            data={"model_id": "nonexistent-model"},
            files={"audio_file": ("test.wav", sample_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 404


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the API."""

    def test_full_workflow_validation(
        self, client: TestClient, sample_audio_bytes: bytes
    ) -> None:
        """Test the validation part of the training workflow."""
        # Step 1: Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # Step 2: List existing models
        models_response = client.get("/api/models/list")
        assert models_response.status_code == 200

        # Step 3: Attempt to start training with valid data
        # (This will start but we won't wait for completion)
        files = [
            ("recordings", ("audio1.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio2.wav", sample_audio_bytes, "audio/wav")),
            ("recordings", ("audio3.wav", sample_audio_bytes, "audio/wav")),
        ]

        # Note: This test validates the endpoint accepts the request
        # Full training would take too long for unit tests
        response = client.post(
            "/api/train/start",
            data={
                "wake_word": "Test Word",
                # Note: model_type is no longer used, AST is always used
            },
            files=files,
        )

        # Should either succeed (200) or fail due to missing dependencies
        # in test environment (which is acceptable)
        # 503 = Service Unavailable (TTS not loaded)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
            assert data["wake_word"] == "Test Word"

            # Step 4: Check job status
            status_response = client.get(f"/api/train/status/{data['job_id']}")
            assert status_response.status_code == 200


# ============================================================================
# Schema Validation Tests
# ============================================================================


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_training_request_wake_word_normalization(self) -> None:
        """Test that wake word whitespace is normalized."""
        from wakebuilder.backend.schemas import TrainingRequest

        # Multiple spaces should be normalized
        request = TrainingRequest(wake_word="Hello   World")
        assert request.wake_word == "Hello World"

    def test_training_request_invalid_wake_word(self) -> None:
        """Test that invalid wake words are rejected."""
        from pydantic import ValidationError

        from wakebuilder.backend.schemas import TrainingRequest

        with pytest.raises(ValidationError):
            TrainingRequest(wake_word="Test123")  # Numbers not allowed

    def test_training_request_too_many_words(self) -> None:
        """Test that wake words with more than 2 words are rejected."""
        from pydantic import ValidationError

        from wakebuilder.backend.schemas import TrainingRequest

        with pytest.raises(ValidationError):
            TrainingRequest(wake_word="One Two Three")  # Too many words

    def test_hyperparameters_validation(self) -> None:
        """Test hyperparameter validation."""
        from wakebuilder.backend.schemas import TrainingHyperparameters

        # Valid hyperparameters
        params = TrainingHyperparameters(
            batch_size=32,
            num_epochs=50,
            learning_rate=0.001,
        )
        assert params.batch_size == 32

    def test_hyperparameters_out_of_range(self) -> None:
        """Test that out-of-range hyperparameters are rejected."""
        from pydantic import ValidationError

        from wakebuilder.backend.schemas import TrainingHyperparameters

        with pytest.raises(ValidationError):
            TrainingHyperparameters(batch_size=1)  # Too small

        with pytest.raises(ValidationError):
            TrainingHyperparameters(learning_rate=1.0)  # Too large
