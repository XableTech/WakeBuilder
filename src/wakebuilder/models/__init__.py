"""
Models module for WakeBuilder.

This module contains the base speech embedding model and wake word classifier.
"""

from .base_model import BaseModelLoader, SpeechEmbeddingModel, load_base_model

__all__ = [
    "BaseModelLoader",
    "SpeechEmbeddingModel",
    "load_base_model",
]
