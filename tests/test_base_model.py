"""
Tests for base model loader and inference.
"""

import numpy as np
import pytest
import torch

from wakebuilder.models.base_model import BaseModelLoader, SpeechEmbeddingModel


class TestSpeechEmbeddingModel:
    """Tests for SpeechEmbeddingModel."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = SpeechEmbeddingModel(embedding_dim=96)
        
        assert model.embedding_dim == 96
        assert isinstance(model, torch.nn.Module)
        
    def test_model_forward_single(self):
        """Test forward pass with single sample."""
        model = SpeechEmbeddingModel(embedding_dim=96)
        model.eval()
        
        # Create input (time=96, freq=80)
        x = torch.randn(1, 96, 80)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (1, 96)
        assert torch.isfinite(output).all()
        
    def test_model_forward_batch(self):
        """Test forward pass with batch."""
        model = SpeechEmbeddingModel(embedding_dim=96)
        model.eval()
        
        # Create batch input
        x = torch.randn(4, 96, 80)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (4, 96)
        assert torch.isfinite(output).all()
        
    def test_model_different_embedding_dims(self):
        """Test model with different embedding dimensions."""
        for dim in [64, 96, 128, 256]:
            model = SpeechEmbeddingModel(embedding_dim=dim)
            x = torch.randn(2, 96, 80)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, dim)


class TestBaseModelLoader:
    """Tests for BaseModelLoader."""

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = BaseModelLoader()
        
        assert loader.device in ["cpu", "cuda"]
        assert not loader.is_loaded
        
    def test_loader_device_selection(self):
        """Test device selection."""
        loader = BaseModelLoader(device="cpu")
        assert loader.device == "cpu"
        
    def test_loader_not_loaded_error(self):
        """Test that methods raise error when model not loaded."""
        loader = BaseModelLoader()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.get_embeddings(np.random.randn(96, 80))
            
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.get_embedding_dim()
            
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.get_metadata()


class TestBaseModelIntegration:
    """Integration tests for base model (requires downloaded model)."""

    @pytest.fixture
    def model_exists(self):
        """Check if model file exists."""
        from pathlib import Path
        from wakebuilder.config import Config
        
        config = Config()
        model_path = Path(config.BASE_MODEL_PATH)
        
        if not model_path.exists():
            pytest.skip("Model not downloaded. Run: uv run python scripts/download_base_model.py")
        
        return model_path

    def test_load_model(self, model_exists):
        """Test loading the model."""
        loader = BaseModelLoader()
        loader.load()
        
        assert loader.is_loaded
        assert loader.model is not None
        assert loader.metadata is not None
        
    def test_get_embeddings_single(self, model_exists):
        """Test getting embeddings for single sample."""
        loader = BaseModelLoader()
        loader.load()
        
        # Create sample mel spectrogram
        mel_spec = np.random.randn(96, 80).astype(np.float32)
        
        # Get embeddings
        embeddings = loader.get_embeddings(mel_spec)
        
        # Check output
        assert embeddings.shape == (96,)  # Single sample, no batch dim
        assert np.isfinite(embeddings).all()
        
    def test_get_embeddings_batch(self, model_exists):
        """Test getting embeddings for batch."""
        loader = BaseModelLoader()
        loader.load()
        
        # Create batch of mel spectrograms
        mel_specs = np.random.randn(4, 96, 80).astype(np.float32)
        
        # Get embeddings
        embeddings = loader.get_embeddings(mel_specs)
        
        # Check output
        assert embeddings.shape == (4, 96)
        assert np.isfinite(embeddings).all()
        
    def test_get_embedding_dim(self, model_exists):
        """Test getting embedding dimension."""
        loader = BaseModelLoader()
        loader.load()
        
        dim = loader.get_embedding_dim()
        
        assert isinstance(dim, int)
        assert dim > 0
        
    def test_get_metadata(self, model_exists):
        """Test getting metadata."""
        loader = BaseModelLoader()
        loader.load()
        
        metadata = loader.get_metadata()
        
        assert isinstance(metadata, dict)
        assert "embedding_dim" in metadata
        assert "model_name" in metadata


class TestEndToEnd:
    """End-to-end tests combining preprocessing and model inference."""

    @pytest.fixture
    def model_exists(self):
        """Check if model file exists."""
        from pathlib import Path
        from wakebuilder.config import Config
        
        config = Config()
        model_path = Path(config.BASE_MODEL_PATH)
        
        if not model_path.exists():
            pytest.skip("Model not downloaded. Run: uv run python scripts/download_base_model.py")
        
        return model_path

    def test_audio_to_embeddings(self, model_exists):
        """Test complete pipeline from audio to embeddings."""
        from wakebuilder.audio.preprocessing import AudioPreprocessor
        
        # Create synthetic audio
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Preprocess
        preprocessor = AudioPreprocessor.from_config()
        mel_spec = preprocessor.process_audio(audio, sample_rate=sample_rate)
        
        # Get embeddings
        loader = BaseModelLoader()
        loader.load()
        embeddings = loader.get_embeddings(mel_spec)
        
        # Verify
        assert embeddings.shape == (96,)
        assert np.isfinite(embeddings).all()
        assert embeddings.std() > 0  # Should have variation
        
    def test_batch_audio_to_embeddings(self, model_exists):
        """Test batch processing from audio to embeddings."""
        from wakebuilder.audio.preprocessing import AudioPreprocessor
        
        # Create batch of synthetic audio
        sample_rate = 16000
        audio_list = [
            np.random.randn(16000) * 0.1,
            np.random.randn(16000) * 0.1,
            np.random.randn(16000) * 0.1,
        ]
        
        # Preprocess batch
        preprocessor = AudioPreprocessor.from_config()
        mel_specs = preprocessor.process_batch(
            audio_list,
            sample_rates=[sample_rate] * 3,
        )
        
        # Get embeddings
        loader = BaseModelLoader()
        loader.load()
        embeddings = loader.get_embeddings(mel_specs)
        
        # Verify
        assert embeddings.shape == (3, 96)
        assert np.isfinite(embeddings).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
