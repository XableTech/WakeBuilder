# Makefile for WakeBuilder development tasks
# Note: On Windows, use 'make' via WSL or Git Bash, or run commands directly

.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-up docker-down

# Default target
help:
	@echo "WakeBuilder Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            Run the development server"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run linting checks"
	@echo "  make format         Format code with black and ruff"
	@echo "  make type-check     Run type checking with mypy"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start Docker containers"
	@echo "  make docker-down    Stop Docker containers"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove generated files"

# Installation
install:
	uv sync

install-dev:
	uv sync --group dev

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/wakebuilder --cov-report=html --cov-report=term

test-unit:
	uv run pytest -m unit

test-integration:
	uv run pytest -m integration

# Code quality
lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix

format:
	uv run black src/ tests/
	uv run ruff check src/ tests/ --fix

type-check:
	uv run mypy src/

# Combined quality check
check: lint type-check test

# Development server
run:
	uv run uvicorn src.wakebuilder.backend.main:app --reload --host 0.0.0.0 --port 8000

# Docker commands
docker-build:
	docker build -t wakebuilder:latest .

docker-up:
	docker-compose up

docker-up-detached:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

clean-models:
	rm -f models/custom/*.onnx
	rm -f models/custom/*.json

clean-temp:
	rm -rf data/temp/*
	touch data/temp/.gitkeep

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Version bump (requires bump2version)
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major
