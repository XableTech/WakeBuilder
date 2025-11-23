# WakeBuilder - Todo Plan

## Project Structure

```
WakeBuilder/
├── src/
│   ├── WakeBuilder/
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py          # Base embedding model loader
│   │   │   ├── classifier.py          # Wake word classifier architecture
│   │   │   └── trainer.py             # Training loop logic
│   │   ├── audio/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py       # Audio to spectrogram conversion
│   │   │   └── augmentation.py        # Data augmentation logic
│   │   ├── tts/
│   │   │   ├── __init__.py
│   │   │   └── generator.py           # TTS wrapper for Piper
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI app entry point
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── training.py        # Training endpoints
│   │   │   │   ├── models.py          # Model management endpoints
│   │   │   │   └── testing.py         # WebSocket testing endpoint
│   │   │   └── jobs.py                # Background job management
│   │   └── config.py                  # Configuration settings
│   └── tests/
│       ├── __init__.py
│       ├── test_preprocessing.py
│       ├── test_augmentation.py
│       ├── test_training.py
│       └── test_api.py
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── app.js
│   │   ├── trainer.js
│   │   └── tester.js
│   └── assets/
├── models/
│   ├── base/                          # Base embedding model
│   └── defaults/                      # Pre-trained default wake words
├── data/
│   ├── voices/                        # Piper TTS voice models
│   └── noise/                         # Background noise samples
├── scripts/
│   ├── download_base_model.py         # Download base embedding model
│   ├── download_voices.py             # Download TTS voices
│   ├── prepare_noise.py               # Prepare noise samples
│   └── train_defaults.py              # Train default wake words
├── project_spec/
│   ├── WakeBuilder_todo.md            # Project TODO list
│   ├── WakeBuilder_project_description.md    # Project Description
│   └── WakeBuilder_project_plan.md    # Project Plan
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Phase 0: Project Initialization

### Tasks

**0.1 Initialize Project**
- Command: `uv init .` done
- Create the directory structure as shown above
- Files: All directories and `__init__.py` files

**0.2 Install Core Dependencies** (Would be done manually)
- Command: `uv pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/cu128`
- Command: `uv add fastapi uvicorn websockets numpy librosa soundfile onnx onnxruntime`
- Files: `pyproject.toml` updated automatically by uv after each command

**0.3 Install Development Dependencies**
- Command: `uv add --dev pytest pytest-asyncio httpx black ruff mypy`
- Files: `pyproject.toml` updated automatically by uv after each command

**0.4 Setup Configuration**
- Create basic configuration file with paths and constants
- Files: `src/WakeBuilder/config.py`

**0.5 Create .gitignore**
- Add Python cache, virtual env, models, data directories
- Files: `.gitignore`

---

## Phase 1: Foundation - Base Model & Audio Pipeline

### Tasks

**1.1 Research and Select Base Model**
- Research options: YAMNet, Google Speech Commands, VGGish
- Document decision and model source URL
- Files: `README.md` (add notes), `scripts/download_base_model.py`

**1.2 Download Base Model Script**
- Create script to download and convert base model to ONNX
- Command: `uv run python scripts/download_base_model.py`
- Files: `scripts/download_base_model.py`, `models/base/model.onnx`

**1.3 Implement Base Model Loader**
- Load ONNX model using onnxruntime
- Expose function to get embeddings from spectrograms
- Files: `src/WakeBuilder/models/base_model.py`

**1.4 Implement Audio Preprocessing**
- Convert raw audio to mel spectrograms
- Match parameters expected by base model
- Files: `src/WakeBuilder/audio/preprocessing.py`

**1.5 Test Foundation**
- Test base model loads correctly
- Test preprocessing produces valid spectrograms
- Test end-to-end: audio → spectrogram → embedding
- Command: `uv run pytest tests/test_preprocessing.py -v`
- Files: `tests/test_preprocessing.py`, `tests/fixtures/sample_audio.wav`

---

## Phase 2: Training Pipeline

### Tasks

**2.1 Setup Piper TTS**
- Command: `uv add piper-tts`
- Create script to download diverse voice models
- Command: `uv run python scripts/download_voices.py`
- Files: `scripts/download_voices.py`, `data/voices/*`

**2.2 Implement TTS Generator**
- Wrapper around Piper to generate speech
- Support multiple voices with speed/pitch variations
- Files: `src/WakeBuilder/tts/generator.py`

**2.3 Download Noise Samples**
- Collect or generate background noise samples
- Command: `uv run python scripts/prepare_noise.py`
- Files: `scripts/prepare_noise.py`, `data/noise/*`

**2.4 Implement Data Augmentation**
- Generate synthetic variations using TTS
- Add background noise at various SNR levels
- Apply volume/speed/pitch variations
- Files: `src/WakeBuilder/audio/augmentation.py`

**2.5 Implement Negative Example Generator**
- Generate phonetically similar words
- Generate random speech
- Include silence and pure noise
- Files: `src/WakeBuilder/audio/augmentation.py` (add function)

**2.6 Test Augmentation Pipeline**
- Verify TTS generation works
- Verify noise mixing works
- Verify output diversity
- Command: `uv run pytest tests/test_augmentation.py -v`
- Files: `tests/test_augmentation.py`

**2.7 Design Classifier Architecture**
- Create simple feedforward network (2-3 layers)
- Input: embeddings, Output: binary classification
- Files: `src/WakeBuilder/models/classifier.py`

**2.8 Implement Training Loop**
- Process augmented data through base model
- Train classifier using PyTorch
- Implement validation split and early stopping
- Files: `src/WakeBuilder/models/trainer.py`

**2.9 Implement Threshold Calibration**
- Compute FAR/FRR at different thresholds
- Save recommended threshold with model
- Files: `src/WakeBuilder/models/trainer.py` (add function)

**2.10 Export to ONNX**
- Convert trained PyTorch classifier to ONNX
- Save with metadata JSON file
- Files: `src/WakeBuilder/models/trainer.py` (add function)

**2.11 Test Training Pipeline**
- Test full training with sample wake word
- Verify model export works
- Verify metadata is correct
- Command: `uv run pytest tests/test_training.py -v`
- Files: `tests/test_training.py`

---

## Phase 3: FastAPI Backend

### Tasks

**3.1 Setup FastAPI Application**
- Create basic FastAPI app with CORS
- Add health check endpoint
- Files: `src/WakeBuilder/api/main.py`

**3.2 Implement Background Job System**
- Simple threading-based job queue
- Track job status in memory
- Files: `src/WakeBuilder/api/jobs.py`

**3.3 Implement Training Endpoints**
- POST /api/train/start - initiate training
- GET /api/train/status/{job_id} - check progress
- GET /api/train/download/{job_id} - download model
- Files: `src/WakeBuilder/api/routes/training.py`

**3.4 Implement Model Management Endpoints**
- GET /api/models/list - list all models
- DELETE /api/models/{model_id} - delete custom model
- GET /api/models/{model_id}/metadata - get model info
- Files: `src/WakeBuilder/api/routes/models.py`

**3.5 Implement WebSocket Testing Endpoint**
- WebSocket /api/test/realtime - real-time detection
- Accept audio chunks, return detection events
- Files: `src/WakeBuilder/api/routes/testing.py`

**3.6 Implement File Management**
- Handle temporary recording storage
- Handle model file storage
- Implement cleanup logic
- Files: `src/WakeBuilder/api/routes/training.py` (add helpers)

**3.7 Test API Endpoints**
- Test training workflow
- Test model management
- Test WebSocket connection
- Command: `uv run pytest tests/test_api.py -v`
- Files: `tests/test_api.py`

**3.8 Manual API Testing**
- Command: `uv run uvicorn WakeBuilder.api.main:app --reload`
- Test in browser or with curl
- Verify all endpoints respond correctly

---

## Phase 4: Web User Interface

### Tasks

**4.1 Setup Static File Serving**
- Configure FastAPI to serve frontend
- Files: `src/WakeBuilder/api/main.py` (add static mount)

**4.2 Create Base HTML Template**
- Create index.html with navigation structure
- Include CSS and JS files
- Files: `frontend/index.html`

**4.3 Create CSS Styles**
- Clean, minimal design
- Responsive layout
- Files: `frontend/css/styles.css`

**4.4 Implement Home Page**
- Display model library
- Show default and custom models
- Add "Create New" button
- Files: `frontend/index.html`, `frontend/js/app.js`

**4.5 Implement Training Wizard - Step 1**
- Wake word text input with validation
- Files: `frontend/index.html`, `frontend/js/trainer.js`

**4.6 Implement Training Wizard - Step 2**
- Microphone recording interface
- Waveform visualization
- Record/playback functionality
- Files: `frontend/index.html`, `frontend/js/trainer.js`

**4.7 Implement Training Wizard - Step 3**
- Progress indicator
- Status updates via polling
- Files: `frontend/index.html`, `frontend/js/trainer.js`

**4.8 Implement Training Wizard - Step 4**
- Automatic redirect to testing page
- Files: `frontend/index.html`, `frontend/js/trainer.js`

**4.9 Implement Testing Page**
- Real-time detection visualization
- Sensitivity slider
- Detection log
- WebSocket connection
- Files: `frontend/index.html`, `frontend/js/tester.js`

**4.10 Manual UI Testing**
- Command: `uv run uvicorn WakeBuilder.api.main:app --reload`
- Open http://localhost:8000
- Test complete training workflow
- Test model testing functionality

---

## Phase 5: Docker & Default Models

### Tasks

**5.1 Create Dockerfile**
- Python slim base image
- Install system dependencies
- Copy application files
- Set up entry point
- Files: `Dockerfile`

**5.2 Create docker-compose.yml**
- Define service configuration
- Set up volume mounts
- Configure port mapping
- Files: `docker-compose.yml`

**5.3 Train Default Wake Words**
- Create script to train default models
- Train: "Computer", "Assistant", "System", "Listen", "Voice"
- Train: "Hey There", "Wake Up", "Hi Computer", "Hi Assistant"
- Command: `uv run python scripts/train_defaults.py`
- Files: `scripts/train_defaults.py`, `models/defaults/*.onnx`

**5.4 Test Docker Build**
- Command: `docker build -t WakeBuilder .`
- Verify build completes without errors

**5.5 Test Docker Run**
- Command: `docker-compose up`
- Verify application starts
- Test access at http://localhost:8000
- Test model persistence with volume mount

**5.6 Optimize Docker Image**
- Multi-stage build if needed
- Minimize image size
- Files: `Dockerfile` (refine)

---

## Phase 6: Testing, Optimization & Documentation

### Tasks

**6.1 Comprehensive Testing**
- Test multiple wake words (short, long, unusual)
- Test different accents if possible
- Document accuracy metrics
- Command: `uv run pytest tests/ -v --cov=WakeBuilder`

**6.2 Performance Profiling**
- Profile training pipeline
- Identify bottlenecks
- Command: `uv run python -m cProfile -o profile.stats scripts/profile_training.py`
- Files: `scripts/profile_training.py`

**6.3 Optimize Critical Paths**
- Optimize spectrogram computation
- Optimize ONNX inference
- Optimize data augmentation
- Files: Refine relevant modules

**6.4 Cross-Platform Testing**
- Test on Linux
- Test on macOS
- Test on Windows with Docker Desktop

**6.5 Write README**
- Project description
- Quick start guide
- Docker usage instructions
- System requirements
- Files: `README.md`

**6.6 Write User Guide**
- Step-by-step training tutorial
- Screenshots of web interface
- Troubleshooting section
- Files: `docs/USER_GUIDE.md`

**6.7 Write API Documentation**
- Document all endpoints
- Include request/response examples
- Files: `docs/API.md`

**6.8 Write Technical Documentation**
- System architecture
- How training works
- Model format specification
- Files: `docs/TECHNICAL.md`

**6.9 Create Contributing Guide**
- Development setup
- Testing procedures
- Pull request process
- Files: `CONTRIBUTING.md`

**6.10 Final QA Pass**
- Complete workflow test
- Verify all documentation is accurate
- Check for any remaining issues

---

## Phase 7: Release Preparation

### Tasks

**7.1 Code Quality Check**
- Command: `uv run black src/ tests/`
- Command: `uv run flake8 src/ tests/`
- Command: `uv run mypy src/`
- Fix any issues found

**7.2 Create GitHub Repository**
- Initialize git repo
- Create .gitignore
- Make initial commit
- Files: `.git/`, `.gitignore`

**7.3 Tag Initial Release**
- Command: `git tag -a v0.1.0 -m "Initial release"`
- Command: `git push origin v0.1.0`

**7.4 Build Docker Image for Release**
- Command: `docker build -t WakeBuilder:0.1.0 .`
- Command: `docker tag WakeBuilder:0.1.0 WakeBuilder:latest`

**7.5 Write Release Notes**
- Document features
- Document known limitations
- Files: `CHANGELOG.md`

**7.6 Create Demo Video/Screenshots**
- Record demo of training workflow
- Capture screenshots for documentation
- Files: `docs/images/`

**7.7 Publish Documentation**
- Set up GitHub Pages or similar
- Deploy documentation site

**7.8 Announce Release**
- Prepare announcement post
- Share on relevant communities

---

## Notes

### Testing Commands Summary
- Run all tests: `uv run pytest tests/ -v`
- Run with coverage: `uv run pytest tests/ -v --cov=WakeBuilder`
- Run specific test: `uv run pytest tests/test_preprocessing.py -v`

### Development Commands Summary
- Start dev server: `uv run uvicorn WakeBuilder.api.main:app --reload`
- Format code: `uv run black src/ tests/`
- Lint code: `uv run flake8 src/ tests/`
- Type check: `uv run mypy src/`

### Docker Commands Summary
- Build image: `docker build -t WakeBuilder .`
- Run with compose: `docker-compose up`
- Run detached: `docker-compose up -d`
- Stop: `docker-compose down`
- View logs: `docker-compose logs -f`

### Priority Order
1. Phase 0 (setup)
2. Phase 1 (foundation - critical)
3. Phase 2 (training - core functionality)
4. Phase 3 (backend - integration)
5. Phase 4 (frontend - user interface)
6. Phase 5 (docker - deployment)
7. Phase 6 (polish - quality)
8. Phase 7 (release - distribution)