"""
Base speech embedding model loader and inference.

This module provides functionality to load the pre-trained Google Speech Embedding
model and extract embeddings from mel spectrograms for wake word detection.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import Config


class SpeechEmbeddingModel(nn.Module):
    """
    PyTorch implementation of Google Speech Embedding model.

    This model takes mel spectrograms as input and produces fixed-dimensional
    embeddings optimized for speech representation tasks.
    """

    def __init__(self, embedding_dim: int = 96):
        """
        Initialize the speech embedding model.

        Args:
            embedding_dim: Dimension of output embeddings (default: 96)
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Placeholder architecture - will be loaded from checkpoint
        self.conv_layers = nn.Sequential()
        self.embedding_layer = nn.Linear(1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding model.

        Args:
            x: Input mel spectrogram tensor of shape (batch, time, freq)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Simple pooling for now - will be replaced with actual model
        pooled = x.mean(dim=(1, 2), keepdim=True)
        embeddings = self.embedding_layer(pooled).squeeze()

        # Handle single sample case
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings


class BaseModelLoader:
    """
    Loader for the pre-trained base speech embedding model.

    This class handles loading the model from disk, managing device placement,
    and providing a clean interface for embedding extraction.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the base model loader.

        Args:
            model_path: Path to the saved model checkpoint. If None, uses default from config.
            device: Device to load model on ('cpu', 'cuda'). If None, auto-detects.
        """
        self.config = Config()
        self.model_path = (
            Path(model_path) if model_path else Path(self.config.BASE_MODEL_PATH)
        )

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[SpeechEmbeddingModel] = None
        self.metadata: Optional[dict] = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load the model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please run: uv run python scripts/download_base_model.py"
            )

        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )

            # Extract metadata
            self.metadata = checkpoint.get("metadata", {})
            embedding_dim = self.metadata.get("embedding_dim", 96)

            # Create model
            self.model = SpeechEmbeddingModel(embedding_dim=embedding_dim)

            # Load state dict
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def get_embeddings(
        self,
        mel_spectrograms: np.ndarray,
    ) -> np.ndarray:
        """
        Extract embeddings from mel spectrograms.

        Args:
            mel_spectrograms: Mel spectrogram array of shape (batch, time, freq)
                            or (time, freq) for single sample

        Returns:
            Embeddings array of shape (batch, embedding_dim) or (embedding_dim,)

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input shape is invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to tensor
        if not isinstance(mel_spectrograms, torch.Tensor):
            mel_spectrograms = torch.from_numpy(mel_spectrograms).float()

        # Add batch dimension if needed
        if mel_spectrograms.dim() == 2:
            mel_spectrograms = mel_spectrograms.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # Validate shape
        if mel_spectrograms.dim() != 3:
            raise ValueError(
                f"Expected 2D or 3D input, got shape {mel_spectrograms.shape}"
            )

        # Move to device
        mel_spectrograms = mel_spectrograms.to(self.device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model(mel_spectrograms)

        # Convert to numpy
        embeddings = embeddings.cpu().numpy()

        # Remove batch dimension for single sample
        if single_sample:
            embeddings = embeddings.squeeze(0)

        return embeddings

    def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding dimension

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.model.embedding_dim

    def get_metadata(self) -> dict:
        """
        Get model metadata.

        Returns:
            Dictionary containing model metadata

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.metadata.copy()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded


def load_base_model(
    model_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> BaseModelLoader:
    """
    Convenience function to load the base model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load on

    Returns:
        Loaded BaseModelLoader instance
    """
    loader = BaseModelLoader(model_path=model_path, device=device)
    loader.load()
    return loader
