"""
FastAPI application entry point for WakeBuilder.

This module sets up the FastAPI application with:
- CORS middleware for frontend communication
- API routers for training, models, and testing
- OpenAPI documentation with Swagger UI and ReDoc
- Health check and system info endpoints
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Config, ensure_directories
from .routes import models, testing, training
from .schemas import APIInfo, HealthResponse, SystemInfo

# Frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend"

# API metadata for documentation
API_TITLE = "WakeBuilder API"
API_DESCRIPTION = """
## Wake Word Training Platform API

WakeBuilder enables you to create custom wake word detection models locally
without requiring cloud services or machine learning expertise.

### Features

- **Training**: Train custom wake word models from voice recordings
- **Model Management**: List, download, and delete trained models
- **Real-time Testing**: Test models with live audio via WebSocket

### Training Workflow

1. **Start Training**: Submit wake word text and audio recordings
2. **Monitor Progress**: Poll status endpoint for real-time updates
3. **Download Model**: Get the trained model files when complete

### Model Architectures

- **TC-ResNet**: Fast inference (~0.6ms), smaller size (~250KB)
- **BC-ResNet**: Higher accuracy, larger size (~468KB)

### Documentation

- **Swagger UI**: Interactive API documentation at `/docs`
- **ReDoc**: Alternative documentation at `/redoc`
- **OpenAPI Schema**: JSON schema at `/openapi.json`
"""

API_TAGS_METADATA = [
    {
        "name": "health",
        "description": "Health check and system information endpoints",
    },
    {
        "name": "training",
        "description": "Wake word model training endpoints",
    },
    {
        "name": "models",
        "description": "Model management endpoints (list, download, delete)",
    },
    {
        "name": "testing",
        "description": "Model testing endpoints (file-based and real-time WebSocket)",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print(f"Starting WakeBuilder API v{__version__}")
    ensure_directories()
    print(f"  Models directory: {Config.MODELS_DIR}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    yield

    # Shutdown
    print("Shutting down WakeBuilder API")


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=__version__,
    openapi_tags=API_TAGS_METADATA,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    contact={
        "name": "WakeBuilder",
        "url": "https://github.com/wakebuilder/wakebuilder",
    },
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router, prefix="/api/train", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(testing.router, prefix="/api/test", tags=["testing"])


# ============================================================================
# Root and Health Endpoints
# ============================================================================


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the frontend application."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    # Fallback to API docs if frontend not built
    return RedirectResponse(url="/docs")  # type: ignore[return-value]


# Mount static files for frontend assets (CSS, JS, images)
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
    app.mount(
        "/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health Check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the API service.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(),
    )


@app.get(
    "/api/info",
    response_model=APIInfo,
    tags=["health"],
    summary="API Information",
    description="Get detailed information about the API and system.",
)
async def api_info() -> APIInfo:
    """
    Get API and system information.

    Returns version information, system capabilities, and documentation URLs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return APIInfo(
        name=API_TITLE,
        version=__version__,
        description="Wake word training platform API",
        docs_url="/docs",
        system=SystemInfo(
            version=__version__,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            device=device,
        ),
    )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):  # type: ignore[no-untyped-def]
    """Handle ValueError exceptions."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": str(exc),
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError):  # type: ignore[no-untyped-def]
    """Handle FileNotFoundError exceptions."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": str(exc),
        },
    )


# ============================================================================
# Application Runner
# ============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    uvicorn.run(
        "wakebuilder.backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(reload=True)
