"""
Models module for WakeBuilder.

This module contains wake word detection models and training utilities.

Recommended models:
- TCResNet: Best for production (0.6ms latency, 250KB)
- BCResNet: Best for accuracy (6ms latency, 468KB)
"""

from .classifier import (
    BCResNet,
    TCResNet,
    count_parameters,
    create_model,
    get_model_info,
)
from .trainer import (
    ThresholdMetrics,
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    WakeWordDataset,
    calibrate_threshold,
    print_threshold_report,
)

__all__ = [
    # Models
    "BCResNet",
    "TCResNet",
    "create_model",
    "count_parameters",
    "get_model_info",
    # Training
    "Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "WakeWordDataset",
    "ThresholdMetrics",
    "calibrate_threshold",
    "print_threshold_report",
]
