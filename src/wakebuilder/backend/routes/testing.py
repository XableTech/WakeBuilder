"""
Testing endpoints for WakeBuilder API.

This module provides endpoints for:
- File-based model testing (upload audio file)
- Real-time WebSocket testing (stream audio)
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from ...audio import AudioPreprocessor
from ...config import Config
from ...models.classifier import create_model
from ..schemas import (
    ErrorResponse,
    TestFileResponse,
)

router = APIRouter()


class ModelLoader:
    """
    Cached model loader for efficient inference.

    Keeps recently used models in memory to avoid repeated loading.
    """

    def __init__(self, max_cache_size: int = 3):
        self._cache: dict[str, tuple[torch.nn.Module, dict, float]] = {}
        self._max_cache_size = max_cache_size
        self._preprocessor = AudioPreprocessor()

    def _find_model_path(self, model_id: str) -> Optional[Path]:
        """Find the model file path."""
        # Check custom models first
        custom_path = Config.CUSTOM_MODELS_DIR / model_id / "model.pt"
        if custom_path.exists():
            return custom_path

        # Check default models
        default_path = Config.DEFAULT_MODELS_DIR / model_id / "model.pt"
        if default_path.exists():
            return default_path

        return None

    def _load_metadata(self, model_id: str) -> dict:
        """Load model metadata."""
        # Check custom models first
        custom_meta = Config.CUSTOM_MODELS_DIR / model_id / "metadata.json"
        if custom_meta.exists():
            with open(custom_meta) as f:
                return json.load(f)

        # Check default models
        default_meta = Config.DEFAULT_MODELS_DIR / model_id / "metadata.json"
        if default_meta.exists():
            with open(default_meta) as f:
                return json.load(f)

        return {}

    def load_model(self, model_id: str) -> tuple[torch.nn.Module, dict]:
        """
        Load a model by ID, using cache if available.

        Returns:
            Tuple of (model, metadata)
        """
        # Check cache
        if model_id in self._cache:
            model, metadata, _ = self._cache[model_id]
            # Update access time
            self._cache[model_id] = (model, metadata, time.time())
            return model, metadata

        # Find model path
        model_path = self._find_model_path(model_id)
        if not model_path:
            raise FileNotFoundError(f"Model not found: {model_id}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Create model
        model_type = checkpoint.get("model_type", "bc_resnet")
        n_mels = checkpoint.get("n_mels", 80)

        kwargs = {}
        if model_type == "bc_resnet":
            kwargs["base_channels"] = checkpoint.get("base_channels", 16)
            kwargs["scale"] = checkpoint.get("scale", 1.0)
        elif model_type == "tc_resnet":
            kwargs["width_mult"] = checkpoint.get("scale", 1.0)

        model = create_model(
            model_type=model_type,
            num_classes=2,
            n_mels=n_mels,
            **kwargs,
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load metadata
        metadata = self._load_metadata(model_id)

        # Add to cache
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_id = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[oldest_id]

        self._cache[model_id] = (model, metadata, time.time())

        return model, metadata

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache.clear()


# Global model loader
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def run_inference(
    model: torch.nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    preprocessor: AudioPreprocessor,
) -> float:
    """
    Run inference on audio data.

    Args:
        model: The wake word model
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        preprocessor: Audio preprocessor

    Returns:
        Confidence score (0-1)
    """
    # Check for silence/very low energy audio - return 0 confidence
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.005:  # Very quiet audio threshold
        return 0.0
    
    # Preprocess audio to mel spectrogram
    spec = preprocessor.process_audio(audio, sample_rate)

    # Convert to tensor
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence = probs[0, 1].item()  # Probability of wake word class
    
    # Scale confidence by audio energy to reduce false positives on quiet audio
    # This helps prevent high confidence on near-silence
    energy_scale = min(1.0, rms / 0.05)  # Full confidence at RMS >= 0.05
    confidence = confidence * (0.3 + 0.7 * energy_scale)  # Keep 30% base, scale 70%

    return confidence


@router.post(
    "/file",
    response_model=TestFileResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio file"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
    summary="Test with Audio File",
    description="""
Test a wake word model with an uploaded audio file.

**Audio Requirements:**
- Format: WAV, FLAC, or OGG
- Duration: 0.5-3.0 seconds
- Sample rate: Any (will be resampled to 16kHz)

Returns detection result with confidence score.
""",
)
async def test_with_file(
    audio_file: UploadFile = File(..., description="Audio file to test"),
    model_id: str = Form(..., description="Model ID to test with"),
    threshold: Optional[float] = Form(
        None, ge=0, le=1, description="Custom threshold (uses model default if not set)"
    ),
) -> TestFileResponse:
    """
    Test a model with an uploaded audio file.

    Returns whether the wake word was detected and the confidence score.
    """
    start_time = time.time()

    # Load model
    loader = get_model_loader()
    try:
        model, metadata = loader.load_model(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # Get threshold
    if threshold is None:
        threshold = metadata.get("threshold", 0.5)

    # Load audio file
    try:
        content = await audio_file.read()
        audio_io = io.BytesIO(content)
        audio, sr = sf.read(audio_io)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}") from e

    # Check duration
    duration = len(audio) / sr
    if duration < 0.1:
        raise HTTPException(
            status_code=400, detail="Audio file too short (minimum 0.1 seconds)"
        )
    if duration > 10.0:
        raise HTTPException(
            status_code=400, detail="Audio file too long (maximum 10 seconds)"
        )

    # Run inference
    preprocessor = AudioPreprocessor()
    confidence = run_inference(model, audio, sr, preprocessor)

    # Determine detection
    detected = confidence >= threshold

    processing_time = (time.time() - start_time) * 1000

    return TestFileResponse(
        detected=detected,
        confidence=confidence,
        threshold=threshold,
        model_id=model_id,
        wake_word=metadata.get("wake_word", "unknown"),
        processing_time_ms=processing_time,
    )


@router.websocket("/realtime")
async def realtime_testing(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time wake word detection.

    **Connection:**
    Connect with query parameters:
    - `model_id`: ID of the model to test
    - `threshold`: Optional custom threshold (0-1)
    - `cooldown_ms`: Cooldown between detections (default: 1000)

    **Protocol:**
    1. Client sends audio chunks as binary data (16-bit PCM, 16kHz, mono)
    2. Server responds with JSON detection events

    **Detection Event Format:**
    ```json
    {
        "type": "detection",
        "detected": true,
        "confidence": 0.95,
        "threshold": 0.5,
        "timestamp": "2024-01-15T10:30:00.000Z"
    }
    ```

    **Error Event Format:**
    ```json
    {
        "type": "error",
        "message": "Error description"
    }
    ```
    """
    # Get query parameters
    model_id = websocket.query_params.get("model_id")
    threshold_str = websocket.query_params.get("threshold")
    cooldown_str = websocket.query_params.get("cooldown_ms", "1000")

    if not model_id:
        await websocket.close(code=4000, reason="model_id parameter required")
        return

    # Parse parameters
    threshold: Optional[float] = None
    if threshold_str:
        try:
            threshold = float(threshold_str)
            if not 0 <= threshold <= 1:
                await websocket.close(
                    code=4001, reason="threshold must be between 0 and 1"
                )
                return
        except ValueError:
            await websocket.close(code=4001, reason="Invalid threshold value")
            return

    try:
        cooldown_ms = int(cooldown_str)
    except ValueError:
        cooldown_ms = 1000

    # Accept connection
    await websocket.accept()

    # Load model
    loader = get_model_loader()
    try:
        model, metadata = loader.load_model(model_id)
    except FileNotFoundError:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Model not found: {model_id}",
            }
        )
        await websocket.close(code=4004)
        return

    # Get threshold from metadata if not provided
    if threshold is None:
        threshold = metadata.get("threshold", 0.5)

    # Send ready message
    await websocket.send_json(
        {
            "type": "ready",
            "model_id": model_id,
            "wake_word": metadata.get("wake_word", "unknown"),
            "threshold": threshold,
            "cooldown_ms": cooldown_ms,
        }
    )

    # Setup
    preprocessor = AudioPreprocessor()
    sample_rate = 16000
    last_detection_time = 0.0
    audio_buffer = np.array([], dtype=np.float32)
    chunk_duration = 1.0  # Process 1 second chunks
    chunk_samples = int(sample_rate * chunk_duration)

    try:
        while True:
            # Receive audio data
            data = await websocket.receive()

            if "bytes" in data:
                # Binary audio data (16-bit PCM)
                audio_bytes = data["bytes"]

                # Convert to numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(
                    np.float32
                )
                audio_chunk = audio_chunk / 32768.0  # Normalize to [-1, 1]

                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                # Process when we have enough samples
                while len(audio_buffer) >= chunk_samples:
                    # Extract chunk
                    chunk = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[chunk_samples // 2 :]  # 50% overlap

                    # Run inference
                    start_time = time.time()
                    confidence = run_inference(model, chunk, sample_rate, preprocessor)
                    inference_time = (time.time() - start_time) * 1000

                    # Check for detection with cooldown
                    current_time = time.time()
                    detected = bool(confidence >= threshold)  # Ensure Python bool
                    in_cooldown = bool((current_time - last_detection_time) < (
                        cooldown_ms / 1000
                    ))

                    if detected and not in_cooldown:
                        last_detection_time = current_time

                    # Send detection event - ensure all values are JSON serializable
                    await websocket.send_json(
                        {
                            "type": "detection",
                            "detected": bool(detected and not in_cooldown),
                            "confidence": float(confidence),
                            "threshold": float(threshold),
                            "timestamp": datetime.now().isoformat(),
                            "inference_time_ms": float(inference_time),
                            "in_cooldown": bool(in_cooldown),
                        }
                    )

            elif "text" in data:
                # JSON command
                try:
                    command = json.loads(data["text"])
                    cmd_type = command.get("type")

                    if cmd_type == "set_threshold":
                        new_threshold = command.get("threshold")
                        if new_threshold is not None and 0 <= new_threshold <= 1:
                            threshold = new_threshold
                            await websocket.send_json(
                                {
                                    "type": "threshold_updated",
                                    "threshold": threshold,
                                }
                            )

                    elif cmd_type == "set_cooldown":
                        new_cooldown = command.get("cooldown_ms")
                        if new_cooldown is not None and new_cooldown >= 0:
                            cooldown_ms = new_cooldown
                            await websocket.send_json(
                                {
                                    "type": "cooldown_updated",
                                    "cooldown_ms": cooldown_ms,
                                }
                            )

                    elif cmd_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif cmd_type == "reset":
                        audio_buffer = np.array([], dtype=np.float32)
                        last_detection_time = 0.0
                        await websocket.send_json({"type": "reset_complete"})

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid JSON command",
                        }
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
        except Exception:
            pass


@router.get(
    "/models",
    summary="List Testable Models",
    description="Get a list of models available for testing with their thresholds.",
)
async def list_testable_models() -> dict:
    """
    List all models available for testing.

    Returns model IDs, wake words, and recommended thresholds.
    """
    models = []

    # Scan custom models
    if Config.CUSTOM_MODELS_DIR.exists():
        for model_dir in Config.CUSTOM_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            meta = json.load(f)
                        models.append(
                            {
                                "model_id": model_dir.name,
                                "wake_word": meta.get("wake_word", model_dir.name),
                                "threshold": meta.get("threshold", 0.5),
                                "category": "custom",
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Scan default models
    if Config.DEFAULT_MODELS_DIR.exists():
        for model_dir in Config.DEFAULT_MODELS_DIR.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            meta = json.load(f)
                        models.append(
                            {
                                "model_id": model_dir.name,
                                "wake_word": meta.get("wake_word", model_dir.name),
                                "threshold": meta.get("threshold", 0.5),
                                "category": "default",
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass

    return {
        "models": models,
        "total": len(models),
    }
