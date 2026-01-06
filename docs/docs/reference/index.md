# Reference

Technical reference documentation for WakeBuilder.

---

## In This Section

<div class="grid cards" markdown>

- :material-folder-open:{ .lg .middle } **Project Structure**

    ---

    Complete directory and file structure reference.

    [:octicons-arrow-right-24: Structure](project-structure.md)

- :material-script-text:{ .lg .middle } **Scripts Reference**

    ---

    Available utility scripts and their usage.

    [:octicons-arrow-right-24: Scripts](scripts.md)

- :material-api:{ .lg .middle } **API Endpoints**

    ---

    REST API and WebSocket endpoint documentation.

    [:octicons-arrow-right-24: API](api-endpoints.md)

- :material-file-document:{ .lg .middle } **Data Formats**

    ---

    File formats, schemas, and data structures.

    [:octicons-arrow-right-24: Data Formats](data-formats.md)

</div>

---

## Quick Reference

### Key Paths

| Path | Purpose |
|------|---------|
| `src/wakebuilder/` | Main source code |
| `frontend/` | Web interface |
| `models/` | Trained models |
| `data/` | Training data |
| `tts_voices/` | TTS voice models |
| `scripts/` | Utility scripts |
| `tests/` | Test suite |

### Key Files

| File | Purpose |
|------|---------|
| `src/wakebuilder/config.py` | Configuration settings |
| `src/wakebuilder/backend/main.py` | API entry point |
| `src/wakebuilder/models/trainer.py` | Training pipeline |
| `src/wakebuilder/models/classifier.py` | Model architecture |
| `run.py` | Setup and run script |
| `pyproject.toml` | Project metadata |

### API Base URL

```
http://localhost:8000
```

### Documentation URLs

| URL | Description |
|-----|-------------|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc |
| `/openapi.json` | OpenAPI schema |
| `/health` | Health check |
| `/api/info` | System info |
