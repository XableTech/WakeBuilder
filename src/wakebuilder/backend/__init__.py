"""
FastAPI backend for WakeBuilder web interface.

This module provides REST API endpoints and WebSocket handlers for
training orchestration and real-time testing.
"""

from .jobs import JobInfo, JobManager, JobStatus, get_job_manager
from .main import app, run_server

__all__ = [
    "app",
    "run_server",
    "JobInfo",
    "JobManager",
    "JobStatus",
    "get_job_manager",
]
