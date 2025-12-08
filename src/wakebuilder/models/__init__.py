"""
Models module for WakeBuilder.

This module contains wake word detection models and training utilities
using Audio Spectrogram Transformer (AST) for transfer learning.

Architecture:
- Base: MIT/ast-finetuned-speech-commands-v2 (frozen)
- Classifier: Trainable feedforward network on AST embeddings
"""

from .classifier import (
    AST_MODEL_CHECKPOINT,
    ASTFeatureExtractorWrapper,
    ASTWakeWordModel,
    SelfAttentionPooling,
    WakeWordClassifier,
    count_parameters,
    get_model_info,
    load_ast_model,
    save_wake_word_model,
)
from .trainer import (
    ASTDataset,
    ASTTrainer,
    FocalLoss,
    ThresholdMetrics,
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    calibrate_threshold,
    print_threshold_report,
)

__all__ = [
    # AST Model
    "AST_MODEL_CHECKPOINT",
    "ASTWakeWordModel",
    "WakeWordClassifier",
    "SelfAttentionPooling",
    "ASTFeatureExtractorWrapper",
    "load_ast_model",
    "save_wake_word_model",
    "count_parameters",
    "get_model_info",
    # Training
    "ASTTrainer",
    "Trainer",  # Alias for backward compatibility
    "TrainingConfig",
    "TrainingMetrics",
    "ASTDataset",
    "FocalLoss",
    "ThresholdMetrics",
    "calibrate_threshold",
    "print_threshold_report",
]
