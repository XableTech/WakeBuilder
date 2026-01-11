"""
Cache management endpoints for WakeBuilder API.

This module provides endpoints for:
- Checking cache status (negative chunks, spectrograms)
- Building cache with progress tracking via SSE
- Clearing cache
"""

import asyncio
import json
from pathlib import Path

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
        "training_ready": (spec_cache["cached"] and spec_cache.get("count", 0) > 0)
        or (chunk_cache["cached"] and chunk_cache["chunk_count"] > 0),
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
            detail="No negative audio files found in data/negative/ directory",
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
                    percent = (
                        int((last_processed / total_files) * 100)
                        if total_files > 0
                        else 0
                    )
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
            detail="Audio cache must be built first. Call /api/cache/build/audio first.",
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
            detail="No negative audio files found in data/negative/ directory",
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
                    percent = (
                        int((last_processed / total_files) * 50)
                        if total_files > 0
                        else 0
                    )
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
                    percent = (
                        50 + int((last_processed / total) * 50) if total > 0 else 50
                    )
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


# Dataset download URL (UNAC - Universal Negative Audio Corpus from Kaggle)
NEGATIVE_DATA_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/rajichisami/universal-negative-audio-corpus-unac"


@router.get(
    "/negative-data/status",
    summary="Get Negative Data Status",
    description="Check if negative audio data is available for training.",
)
async def get_negative_data_status():
    """Get status of negative data availability."""
    loader = RealNegativeDataLoader()
    file_counts = loader.get_file_count()
    negative_dir = Path(Config.DATA_DIR) / "negative"

    return {
        "available": loader.available and file_counts["total"] > 0,
        "file_count": file_counts["total"],
        "directory": str(negative_dir),
        "download_url": NEGATIVE_DATA_DOWNLOAD_URL,
        "required_minimum": 100,  # Minimum files needed for reasonable training
    }


@router.post(
    "/negative-data/download",
    summary="Download Negative Data",
    description="Download the UNAC dataset directly via HTTP. Returns SSE stream with progress.",
)
async def download_negative_data():
    """
    Download negative data from the UNAC dataset.

    Downloads directly via HTTP and extracts audio files to data/negative/.
    Returns SSE stream with progress updates including percentage.
    """
    import tempfile
    import zipfile
    import requests

    negative_dir = Path(Config.DATA_DIR) / "negative"

    async def generate_progress():
        """Generate SSE events for download progress."""
        try:
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting download...', 'percent': 0})}\n\n"

            # Create temp directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "unac.zip"

                try:
                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Connecting to server...', 'percent': 2})}\n\n"

                    # Direct download without authentication
                    response = requests.get(
                        NEGATIVE_DATA_DOWNLOAD_URL, stream=True, timeout=600
                    )
                    response.raise_for_status()

                    # Get total size for progress calculation
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    last_percent = 0

                    total_mb = total_size // (1024 * 1024) if total_size > 0 else 0
                    yield f"data: {json.dumps({'type': 'progress', 'message': f'Downloading... (0 MB / {total_mb} MB)', 'percent': 5})}\n\n"

                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(
                            chunk_size=1024 * 1024
                        ):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                if total_size > 0:
                                    # Download phase: 5% to 50%
                                    percent = 5 + int((downloaded / total_size) * 45)
                                    downloaded_mb = downloaded // (1024 * 1024)

                                    # Send update on every 1% change
                                    if percent >= last_percent + 1:
                                        last_percent = percent
                                        yield f"data: {json.dumps({'type': 'progress', 'message': f'Downloading... ({downloaded_mb} MB / {total_mb} MB)', 'percent': percent})}\n\n"
                                        # Flush the SSE event
                                        await asyncio.sleep(0)

                    # Close the response to release the connection
                    response.close()

                    # Small delay to ensure file is released on Windows
                    await asyncio.sleep(0.5)

                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Download complete. Extracting files...', 'percent': 52})}\n\n"

                except requests.exceptions.Timeout:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Download timed out after 10 minutes. Please try again.'})}\n\n"
                    return
                except requests.exceptions.RequestException as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Download failed: {str(e)}'})}\n\n"
                    return

                # Create negative directory if it doesn't exist
                negative_dir.mkdir(parents=True, exist_ok=True)

                # Extract zip file
                try:
                    extracted_count = 0
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        members = zf.namelist()
                        audio_files = [
                            m
                            for m in members
                            if m.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
                        ]
                        total_audio = len(audio_files)

                        if total_audio == 0:
                            yield f"data: {json.dumps({'type': 'error', 'message': 'No audio files found in the downloaded archive'})}\n\n"
                            return

                        yield f"data: {json.dumps({'type': 'progress', 'message': f'Extracting {total_audio} audio files...', 'percent': 55})}\n\n"

                        last_extract_percent = 55
                        for i, member in enumerate(audio_files):
                            # Extract to negative folder with flat structure
                            filename = Path(member).name
                            if filename:  # Skip directories
                                with zf.open(member) as source:
                                    target = negative_dir / filename
                                    with open(target, "wb") as f:
                                        f.write(source.read())
                                extracted_count += 1

                                # Extract phase: 55% to 95%
                                if total_audio > 0:
                                    percent = 55 + int((i / total_audio) * 40)
                                    # Update every 5% to avoid flooding
                                    if percent >= last_extract_percent + 5:
                                        last_extract_percent = percent
                                        yield f"data: {json.dumps({'type': 'progress', 'message': f'Extracting... ({extracted_count}/{total_audio} files)', 'percent': percent})}\n\n"
                                        await asyncio.sleep(0)

                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Cleaning up temporary files...', 'percent': 98})}\n\n"

                except zipfile.BadZipFile:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Downloaded file is not a valid zip archive'})}\n\n"
                    return
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Extraction failed: {str(e)}'})}\n\n"
                    return

            # Get final count
            loader = RealNegativeDataLoader()
            file_counts = loader.get_file_count()

            yield f"data: {json.dumps({'type': 'complete', 'message': f'Successfully downloaded {extracted_count} audio files!', 'file_count': file_counts['total'], 'percent': 100})}\n\n"

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
