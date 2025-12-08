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
from ...models.classifier import ASTWakeWordModel, AST_MODEL_CHECKPOINT
from ..schemas import (
    ErrorResponse,
    TestFileResponse,
)

router = APIRouter()


class ModelLoader:
    """
    Cached model loader for efficient inference.

    Keeps recently used models in memory to avoid repeated loading.
    Supports both AST-based models (new) and legacy models.
    """

    def __init__(self, max_cache_size: int = 3):
        self._cache: dict[str, tuple[torch.nn.Module, dict, float, str]] = {}
        self._max_cache_size = max_cache_size
        self._current_device = "cpu"
        self._preprocessor = AudioPreprocessor()
        self._ast_feature_extractor = None  # Lazy loaded

    def _get_ast_feature_extractor(self):
        """Get or create AST feature extractor."""
        if self._ast_feature_extractor is None:
            from transformers import AutoFeatureExtractor
            self._ast_feature_extractor = AutoFeatureExtractor.from_pretrained(
                AST_MODEL_CHECKPOINT
            )
        return self._ast_feature_extractor

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

    def set_device(self, device: str) -> None:
        """Set the device for inference (cpu or cuda)."""
        if device not in ["cpu", "cuda"]:
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        if device != self._current_device:
            self._current_device = device
            # Clear cache to reload models on new device
            self.clear_cache()
    
    def get_device(self) -> str:
        """Get current device."""
        return self._current_device
    
    def get_device_info(self) -> dict:
        """Get device availability info."""
        cuda_available = torch.cuda.is_available()
        return {
            "current_device": self._current_device,
            "cuda_available": cuda_available,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }

    def load_model(self, model_id: str) -> tuple[torch.nn.Module, dict]:
        """
        Load a model by ID, using cache if available.

        Returns:
            Tuple of (model, metadata)
        """
        # Check cache
        if model_id in self._cache:
            model, metadata, _, cached_device = self._cache[model_id]
            if cached_device == self._current_device:
                # Update access time
                self._cache[model_id] = (model, metadata, time.time(), cached_device)
                return model, metadata
            # Device changed, need to reload
            del self._cache[model_id]

        # Find model path
        model_path = self._find_model_path(model_id)
        if not model_path:
            raise FileNotFoundError(f"Model not found: {model_id}")

        # Load checkpoint
        device = self._current_device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check if this is an AST model (new format) or legacy model
        is_ast_model = (
            "classifier_state_dict" in checkpoint or 
            checkpoint.get("base_model", "").startswith("MIT/ast")
        )

        if is_ast_model:
            # Load AST-based model
            model = ASTWakeWordModel(
                freeze_base=True,
                classifier_hidden_dims=checkpoint.get("classifier_hidden_dims", [256, 128]),
                classifier_dropout=checkpoint.get("classifier_dropout", 0.3),
                use_attention=checkpoint.get("use_attention", False),
            )
            
            # Load classifier weights
            if "classifier_state_dict" in checkpoint:
                model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            
            model = model.to(device)
            model.eval()
        else:
            # Legacy model format - not supported in new version
            raise ValueError(
                f"Legacy model format not supported. Model {model_id} needs to be retrained "
                "with the new AST-based architecture."
            )

        # Load metadata
        metadata = self._load_metadata(model_id)
        metadata["is_ast_model"] = is_ast_model

        # Add to cache
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_id = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[oldest_id]

        self._cache[model_id] = (model, metadata, time.time(), device)

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


def spectral_gate_denoise(audio: np.ndarray, sample_rate: int, threshold_db: float = -30) -> np.ndarray:
    """
    Apply spectral gating for noise reduction using proper STFT with overlap-add.
    
    This technique:
    1. Computes the STFT of the audio with proper windowing
    2. Estimates noise floor from quiet parts
    3. Attenuates frequency bins below the noise threshold
    4. Reconstructs using overlap-add
    """
    # STFT parameters
    n_fft = 512
    hop_length = 128
    window = np.hanning(n_fft)
    
    # Pad audio to ensure we can process all samples
    pad_length = n_fft - (len(audio) % hop_length)
    audio_padded = np.pad(audio, (0, pad_length), mode='constant')
    
    # Compute number of frames
    num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    
    # Extract frames with windowing
    frames = np.zeros((num_frames, n_fft))
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = audio_padded[start:start + n_fft] * window
    
    # Compute FFT for each frame
    stft = np.fft.rfft(frames, axis=1)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise floor from the quietest 20% of frames (by total energy)
    frame_energies = np.sum(magnitude ** 2, axis=1)
    noise_percentile = 20
    noise_threshold_idx = int(len(frame_energies) * noise_percentile / 100)
    quiet_frame_indices = np.argsort(frame_energies)[:max(1, noise_threshold_idx)]
    noise_floor = np.mean(magnitude[quiet_frame_indices], axis=0)
    
    # Apply soft threshold (spectral subtraction with over-subtraction factor)
    over_subtraction = 2.0  # More aggressive noise removal
    threshold_linear = 10 ** (threshold_db / 20)
    noise_estimate = noise_floor * over_subtraction * threshold_linear
    
    # Wiener-like soft mask
    mask = np.maximum(0, 1 - (noise_estimate / (magnitude + 1e-10)) ** 2)
    mask = np.sqrt(mask)  # Smooth the mask
    
    # Apply mask and reconstruct
    cleaned_stft = magnitude * mask * np.exp(1j * phase)
    cleaned_frames = np.fft.irfft(cleaned_stft, n=n_fft, axis=1)
    
    # Overlap-add reconstruction
    output_length = len(audio_padded)
    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)
    
    for i in range(num_frames):
        start = i * hop_length
        output[start:start + n_fft] += cleaned_frames[i] * window
        window_sum[start:start + n_fft] += window ** 2
    
    # Normalize by window sum (avoid division by zero)
    window_sum = np.maximum(window_sum, 1e-10)
    output = output / window_sum
    
    # Trim to original length
    cleaned = output[:len(audio)]
    
    return cleaned.astype(np.float32)


def compute_spectral_features(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Compute advanced spectral features to distinguish speech from non-speech sounds.
    
    Key insight: Speech has MULTIPLE formant peaks and temporal variation,
    while screams/laughs/whistles are more tonal (single peak) or impulsive.
    """
    n_fft = min(2048, len(audio))
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    
    # Basic features
    spectrum_sum = np.sum(spectrum) + 1e-10
    centroid = np.sum(freqs * spectrum) / spectrum_sum
    
    # Spectral flatness
    log_spectrum = np.log(spectrum + 1e-10)
    geometric_mean = np.exp(np.mean(log_spectrum))
    arithmetic_mean = np.mean(spectrum) + 1e-10
    flatness = geometric_mean / arithmetic_mean
    
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
    zcr = zero_crossings / len(audio)
    
    # Spectral rolloff
    cumsum = np.cumsum(spectrum ** 2)
    rolloff_threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
    
    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectrum_sum)
    
    # === NEW: Spectral complexity features ===
    
    # 1. Count significant spectral peaks (speech has 3-5 formants)
    # Normalize spectrum for peak detection
    spectrum_norm = spectrum / (np.max(spectrum) + 1e-10)
    # Find peaks above 20% of max
    peak_threshold = 0.2
    peaks = []
    for i in range(1, len(spectrum_norm) - 1):
        if (spectrum_norm[i] > spectrum_norm[i-1] and 
            spectrum_norm[i] > spectrum_norm[i+1] and
            spectrum_norm[i] > peak_threshold):
            peaks.append(i)
    num_peaks = len(peaks)
    
    # 2. Spectral entropy (speech has moderate entropy, pure tones have low)
    spectrum_prob = spectrum / spectrum_sum
    spectral_entropy = -np.sum(spectrum_prob * np.log(spectrum_prob + 1e-10))
    max_entropy = np.log(len(spectrum))
    normalized_entropy = spectral_entropy / max_entropy
    
    # 3. Peak concentration - ratio of energy in top peak vs total
    # Pure tones have high concentration, speech has distributed energy
    sorted_spectrum = np.sort(spectrum)[::-1]
    top_energy = np.sum(sorted_spectrum[:10])  # Top 10 bins
    total_energy = np.sum(spectrum) + 1e-10
    peak_concentration = top_energy / total_energy
    
    # 4. Temporal variation - compute spectrum in 4 segments and measure variance
    segment_len = len(audio) // 4
    segment_centroids = []
    for i in range(4):
        seg = audio[i * segment_len:(i + 1) * segment_len]
        seg_spec = np.abs(np.fft.rfft(seg, n=n_fft // 2))
        seg_freqs = np.fft.rfftfreq(n_fft // 2, 1.0 / sample_rate)
        seg_sum = np.sum(seg_spec) + 1e-10
        seg_centroid = np.sum(seg_freqs * seg_spec) / seg_sum
        segment_centroids.append(seg_centroid)
    temporal_variation = np.std(segment_centroids) / (np.mean(segment_centroids) + 1e-10)
    
    # 5. Formant-like structure: check for energy in typical formant regions
    # F1: 200-900 Hz, F2: 900-2500 Hz, F3: 2500-3500 Hz
    f1_mask = (freqs >= 200) & (freqs <= 900)
    f2_mask = (freqs >= 900) & (freqs <= 2500)
    f3_mask = (freqs >= 2500) & (freqs <= 3500)
    
    f1_energy = np.sum(spectrum[f1_mask]) / total_energy if np.any(f1_mask) else 0
    f2_energy = np.sum(spectrum[f2_mask]) / total_energy if np.any(f2_mask) else 0
    f3_energy = np.sum(spectrum[f3_mask]) / total_energy if np.any(f3_mask) else 0
    
    # Speech typically has energy distributed across F1, F2, F3
    formant_balance = min(f1_energy, f2_energy) / (max(f1_energy, f2_energy) + 1e-10)
    
    return {
        "centroid": centroid,
        "flatness": flatness,
        "zcr": zcr,
        "rolloff": rolloff,
        "bandwidth": bandwidth,
        "num_peaks": num_peaks,
        "spectral_entropy": normalized_entropy,
        "peak_concentration": peak_concentration,
        "temporal_variation": temporal_variation,
        "formant_balance": formant_balance,
        "f1_energy": f1_energy,
        "f2_energy": f2_energy,
        "f3_energy": f3_energy,
    }


def is_speech_like(audio: np.ndarray, sample_rate: int) -> tuple[bool, float]:
    """
    Determine if audio has speech-like characteristics using advanced analysis.
    
    Key discriminators:
    - Speech has MULTIPLE spectral peaks (formants) - screams/whistles have 1-2
    - Speech has temporal variation in spectrum - pure tones are constant
    - Speech has energy distributed across formant regions
    - Speech has moderate spectral entropy - not too tonal, not noise
    
    Returns:
        Tuple of (is_speech_like, speech_score)
    """
    features = compute_spectral_features(audio, sample_rate)
    
    score = 0.0
    penalties = 0.0
    
    # === Primary discriminators (most important) ===
    
    # 1. Number of spectral peaks (CRITICAL: speech has 3+ formants)
    num_peaks = features["num_peaks"]
    if num_peaks >= 4:
        score += 0.25  # Strong speech indicator
    elif num_peaks >= 3:
        score += 0.15
    elif num_peaks <= 1:
        penalties += 0.3  # Single peak = likely tonal (scream, whistle, hum)
    
    # 2. Peak concentration (CRITICAL: speech energy is distributed)
    peak_conc = features["peak_concentration"]
    if peak_conc > 0.8:
        penalties += 0.3  # Energy too concentrated = tonal sound
    elif peak_conc > 0.6:
        penalties += 0.15
    elif peak_conc < 0.4:
        score += 0.15  # Well distributed = speech-like
    
    # 3. Temporal variation (speech changes over time)
    temp_var = features["temporal_variation"]
    if temp_var > 0.15:
        score += 0.2  # Good temporal variation
    elif temp_var > 0.08:
        score += 0.1
    elif temp_var < 0.03:
        penalties += 0.15  # Too constant = sustained tone
    
    # 4. Formant balance (speech has energy in multiple formant regions)
    formant_bal = features["formant_balance"]
    if formant_bal > 0.3:
        score += 0.15  # Good balance between F1 and F2
    elif formant_bal < 0.1:
        penalties += 0.1  # Energy too concentrated in one region
    
    # 5. Spectral entropy (speech has moderate complexity)
    entropy = features["spectral_entropy"]
    if 0.4 <= entropy <= 0.75:
        score += 0.15  # Moderate entropy = speech-like
    elif entropy < 0.3:
        penalties += 0.2  # Too tonal
    elif entropy > 0.85:
        penalties += 0.1  # Too noisy
    
    # === Secondary features ===
    
    # Spectral flatness (speech is not too flat, not too tonal)
    flatness = features["flatness"]
    if 0.15 <= flatness <= 0.4:
        score += 0.1  # Moderate flatness
    elif flatness < 0.1:
        penalties += 0.1  # Too tonal
    elif flatness > 0.7:
        penalties += 0.1  # Too noisy
    
    # ZCR check
    zcr = features["zcr"]
    if 0.03 <= zcr <= 0.12:
        score += 0.05
    elif zcr > 0.35:
        penalties += 0.1  # Too high = noise
    
    # Compute final score
    final_score = max(0.0, min(1.0, score - penalties))
    
    # Higher threshold for speech detection (0.5 instead of 0.4)
    is_speech = final_score >= 0.5
    
    return is_speech, final_score


def run_inference(
    model: torch.nn.Module,
    audio: np.ndarray,
    sample_rate: int,
    preprocessor: AudioPreprocessor,
    feature_extractor=None,
    is_ast_model: bool = True,
) -> float:
    """
    Run inference on audio data.

    For AST models, we use a simplified pipeline since the AST model
    already has strong audio understanding from pre-training.

    Args:
        model: The wake word model (ASTWakeWordModel or legacy)
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        preprocessor: Audio preprocessor (for legacy models)
        feature_extractor: AST feature extractor (for AST models)
        is_ast_model: Whether this is an AST-based model

    Returns:
        Confidence score (0-1)
    """
    # Quick silence check - return 0 for very quiet audio
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.005:
        return 0.0
    
    # Normalize audio to consistent volume level
    max_val = np.abs(audio).max()
    if max_val > 0.01:
        audio = audio / max_val * 0.9

    # Get device from model
    device = next(model.parameters()).device

    if is_ast_model and feature_extractor is not None:
        # AST model inference - simplified pipeline
        # AST is pre-trained on speech commands, so it already handles
        # speech vs non-speech discrimination well
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Process through AST feature extractor
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        # Run inference
        with torch.inference_mode():
            outputs = model(input_values)
            probs = F.softmax(outputs, dim=1)
            confidence = probs[0, 1].item()
    else:
        # Legacy model inference (mel spectrogram based)
        # Keep the speech detection for legacy models
        is_speech, speech_score = is_speech_like(audio, sample_rate)
        if not is_speech:
            return 0.0
        
        spec = preprocessor.process_audio(audio, sample_rate)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(spec_tensor)
            probs = F.softmax(outputs, dim=1)
            raw_confidence = probs[0, 1].item()
        
        # Apply speech score modifier for legacy models
        confidence = raw_confidence * (0.7 + 0.3 * speech_score)

    return confidence


@router.get(
    "/device",
    summary="Get Device Info",
    description="Get current inference device and availability info.",
)
async def get_device_info():
    """Get current device info for inference."""
    loader = get_model_loader()
    return loader.get_device_info()


@router.post(
    "/device",
    summary="Set Inference Device",
    description="Set the device for inference (cpu or cuda).",
)
async def set_device(device: str = Form(..., description="Device to use: 'cpu' or 'cuda'")):
    """Set the device for inference."""
    loader = get_model_loader()
    loader.set_device(device)
    return loader.get_device_info()


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
    is_ast = metadata.get("is_ast_model", True)
    feature_extractor = loader._get_ast_feature_extractor() if is_ast else None
    confidence = run_inference(
        model, audio, sr, preprocessor,
        feature_extractor=feature_extractor,
        is_ast_model=is_ast,
    )

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
    noise_reduction_str = websocket.query_params.get("noise_reduction", "false")

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

    # Parse noise reduction flag
    use_noise_reduction = noise_reduction_str.lower() in ("true", "1", "yes")
    
    # Import noisereduce if needed
    nr_reduce_noise = None
    if use_noise_reduction:
        try:
            import noisereduce as nr
            nr_reduce_noise = nr.reduce_noise
        except ImportError:
            logger.warning("noisereduce not installed, noise reduction disabled")
            use_noise_reduction = False

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
            "noise_reduction": use_noise_reduction,
        }
    )

    # Setup
    preprocessor = AudioPreprocessor()
    is_ast = metadata.get("is_ast_model", True)
    feature_extractor = loader._get_ast_feature_extractor() if is_ast else None
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

                    # Apply noise reduction if enabled
                    if use_noise_reduction and nr_reduce_noise is not None:
                        try:
                            chunk = nr_reduce_noise(
                                y=chunk,
                                sr=sample_rate,
                                stationary=True,
                                prop_decrease=0.8,
                            )
                        except Exception as e:
                            logger.warning(f"Noise reduction failed: {e}")

                    # Run inference
                    start_time = time.time()
                    confidence = run_inference(
                        model, chunk, sample_rate, preprocessor,
                        feature_extractor=feature_extractor,
                        is_ast_model=is_ast,
                    )
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
