"""
Cache management endpoints for WakeBuilder API.

This module provides endpoints for:
- Checking cache status (negative chunks, spectrograms)
- Building cache with progress tracking via SSE
- Clearing cache
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...audio.real_data_loader import RealNegativeDataLoader
from ...config import Config

router = APIRouter()


def get_cache_status() -> dict:
    """Get comprehensive cache status."""
    loader = RealNegativeDataLoader()
    
    # Get source file counts
    file_counts = loader.get_file_count()
    
    # Get audio chunk cache info
    chunk_cache = loader.get_cache_info()
    
    # Get spectrogram cache info
    spec_cache = loader.get_spectrogram_cache_info()
    
    # Check if negative source files exist
    negative_dir = Path(Config.DATA_DIR) / "negative"
    has_source_files = negative_dir.exists() and file_counts["total"] > 0
    
    return {
        "source_files": {
            "available": has_source_files,
            "total": file_counts["total"],
            "by_type": {
                "wav": file_counts["wav"],
                "mp3": file_counts["mp3"],
                "flac": file_counts["flac"],
                "ogg": file_counts["ogg"],
            },
        },
        "audio_cache": {
            "ready": chunk_cache["cached"] and chunk_cache["chunk_count"] > 0,
            "chunk_count": chunk_cache.get("chunk_count", 0),
            "created_at": chunk_cache.get("created_at"),
        },
        "spectrogram_cache": {
            "ready": spec_cache["cached"] and spec_cache.get("count", 0) > 0,
            "count": spec_cache.get("count", 0),
            "created_at": spec_cache.get("created_at"),
        },
        "training_ready": (
            spec_cache["cached"] and spec_cache.get("count", 0) > 0
        ) or (
            chunk_cache["cached"] and chunk_cache["chunk_count"] > 0
        ),
    }


@router.get(
    "/status",
    summary="Get Cache Status",
    description="Get comprehensive status of all cache types.",
)
async def get_status():
    """Get status of all caches."""
    return get_cache_status()


@router.get(
    "/negative/info",
    summary="Get Negative Data Info",
    description="Get information about negative audio source files.",
)
async def get_negative_info():
    """Get info about negative source files."""
    loader = RealNegativeDataLoader()
    file_counts = loader.get_file_count()
    
    return {
        "available": loader.available,
        "total_files": file_counts["total"],
        "file_counts": file_counts,
        "directory": str(Path(Config.DATA_DIR) / "negative"),
    }


@router.post(
    "/build/audio",
    summary="Build Audio Cache",
    description="Build audio chunk cache from negative samples. Returns SSE stream with progress.",
)
async def build_audio_cache(max_workers: int = 4):
    """
    Build audio chunk cache with progress updates via SSE.
    """
    loader = RealNegativeDataLoader()
    
    if not loader.available:
        raise HTTPException(
            status_code=404,
            detail="No negative audio files found in data/negative/ directory"
        )
    
    async def generate_progress():
        """Generate SSE events for build progress."""
        file_counts = loader.get_file_count()
        total_files = file_counts["total"]
        
        yield f"data: {json.dumps({'type': 'start', 'total_files': total_files})}\n\n"
        
        # Track progress
        progress = {"processed": 0, "chunks": 0}
        
        def progress_callback(processed: int, total: int):
            progress["processed"] = processed
        
        # Run build in thread to not block
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                loader.build_cache,
                max_workers=max_workers,
                progress_callback=progress_callback,
            )
            
            # Send progress updates while building
            last_processed = 0
            while not future.done():
                await asyncio.sleep(0.5)
                if progress["processed"] != last_processed:
                    last_processed = progress["processed"]
                    percent = int((last_processed / total_files) * 100) if total_files > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'processed': last_processed, 'total': total_files, 'percent': percent})}\n\n"
            
            # Get result
            try:
                chunk_count = future.result()
                yield f"data: {json.dumps({'type': 'complete', 'chunk_count': chunk_count})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/build/spectrograms",
    summary="Build Spectrogram Cache",
    description="Build spectrogram cache from audio chunks. Returns SSE stream with progress.",
)
async def build_spectrogram_cache():
    """
    Build spectrogram cache with progress updates via SSE.
    
    Requires audio cache to be built first.
    """
    loader = RealNegativeDataLoader()
    
    # Check if audio cache exists
    chunk_cache = loader.get_cache_info()
    if not chunk_cache["cached"] or chunk_cache["chunk_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Audio cache must be built first. Call /api/cache/build/audio first."
        )
    
    async def generate_progress():
        """Generate SSE events for build progress."""
        chunk_count = chunk_cache["chunk_count"]
        
        yield f"data: {json.dumps({'type': 'start', 'total_chunks': chunk_count})}\n\n"
        
        # Import preprocessor
        from ...audio import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        
        # Progress tracking
        spec_progress = {"processed": 0, "total": chunk_count}
        
        def spec_progress_callback(processed: int, total: int):
            spec_progress["processed"] = processed
            spec_progress["total"] = total
        
        # Run build in thread
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                loader.build_spectrogram_cache,
                preprocessor=preprocessor,
                progress_callback=spec_progress_callback,
            )
            
            last_processed = 0
            while not future.done():
                await asyncio.sleep(0.5)
                if spec_progress["processed"] != last_processed:
                    last_processed = spec_progress["processed"]
                    total = spec_progress["total"]
                    percent = int((last_processed / total) * 100) if total > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'processed': last_processed, 'total': total, 'percent': percent})}\n\n"
            
            # Get result
            try:
                spec_count = future.result()
                yield f"data: {json.dumps({'type': 'complete', 'spectrogram_count': spec_count})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/build/all",
    summary="Build All Caches",
    description="Build both audio and spectrogram caches. Returns SSE stream with progress.",
)
async def build_all_caches(max_workers: int = 4):
    """
    Build all caches (audio chunks + spectrograms) with progress updates.
    """
    loader = RealNegativeDataLoader()
    
    if not loader.available:
        raise HTTPException(
            status_code=404,
            detail="No negative audio files found in data/negative/ directory"
        )
    
    async def generate_progress():
        """Generate SSE events for full build progress."""
        file_counts = loader.get_file_count()
        total_files = file_counts["total"]
        
        yield f"data: {json.dumps({'type': 'start', 'phase': 'audio', 'total_files': total_files})}\n\n"
        
        # Phase 1: Build audio cache
        progress = {"processed": 0}
        
        def progress_callback(processed: int, total: int):
            progress["processed"] = processed
        
        import concurrent.futures
        
        # Build audio cache
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                loader.build_cache,
                max_workers=max_workers,
                progress_callback=progress_callback,
            )
            
            last_processed = 0
            while not future.done():
                await asyncio.sleep(0.5)
                if progress["processed"] != last_processed:
                    last_processed = progress["processed"]
                    percent = int((last_processed / total_files) * 50) if total_files > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'phase': 'audio', 'processed': last_processed, 'total': total_files, 'percent': percent})}\n\n"
            
            try:
                chunk_count = future.result()
                yield f"data: {json.dumps({'type': 'phase_complete', 'phase': 'audio', 'chunk_count': chunk_count})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'phase': 'audio', 'message': str(e)})}\n\n"
                return
        
        # Phase 2: Build spectrogram cache
        yield f"data: {json.dumps({'type': 'start', 'phase': 'spectrograms', 'total_chunks': chunk_count})}\n\n"
        
        from ...audio import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        
        # Progress tracking for spectrograms
        spec_progress = {"processed": 0, "total": chunk_count}
        
        def spec_progress_callback(processed: int, total: int):
            spec_progress["processed"] = processed
            spec_progress["total"] = total
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                loader.build_spectrogram_cache,
                preprocessor=preprocessor,
                progress_callback=spec_progress_callback,
            )
            
            last_processed = 0
            while not future.done():
                await asyncio.sleep(0.5)
                if spec_progress["processed"] != last_processed:
                    last_processed = spec_progress["processed"]
                    total = spec_progress["total"]
                    # Calculate percent: 50-100% range for spectrogram phase
                    percent = 50 + int((last_processed / total) * 50) if total > 0 else 50
                    yield f"data: {json.dumps({'type': 'progress', 'phase': 'spectrograms', 'processed': last_processed, 'total': total, 'percent': percent})}\n\n"
            
            try:
                spec_count = future.result()
                yield f"data: {json.dumps({'type': 'complete', 'chunk_count': chunk_count, 'spectrogram_count': spec_count})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'phase': 'spectrograms', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/audio",
    summary="Clear Audio Cache",
    description="Delete all cached audio chunks.",
)
async def clear_audio_cache():
    """Clear the audio chunk cache."""
    loader = RealNegativeDataLoader()
    loader.clear_cache()
    return {"message": "Audio cache cleared", "success": True}


@router.delete(
    "/spectrograms",
    summary="Clear Spectrogram Cache",
    description="Delete all cached spectrograms.",
)
async def clear_spectrogram_cache():
    """Clear the spectrogram cache."""
    loader = RealNegativeDataLoader()
    loader.clear_spectrogram_cache()
    return {"message": "Spectrogram cache cleared", "success": True}


@router.delete(
    "/all",
    summary="Clear All Caches",
    description="Delete all cached data (audio chunks and spectrograms).",
)
async def clear_all_caches():
    """Clear all caches."""
    loader = RealNegativeDataLoader()
    loader.clear_cache()
    loader.clear_spectrogram_cache()
    return {"message": "All caches cleared", "success": True}
