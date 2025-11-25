"""
API route modules for WakeBuilder.

This package contains the FastAPI routers for:
- training: Training job management endpoints
- models: Model management endpoints
- testing: Real-time and file-based testing endpoints
"""

from . import models, testing, training

__all__ = ["training", "models", "testing"]
