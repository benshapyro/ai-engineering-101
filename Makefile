# Makefile for Prompt Engineering 101 Curriculum
# Provides convenient shortcuts for common development tasks

.PHONY: help setup install test lint format clean run-example docker-build docker-test

# Default target: show help
help:
	@echo "Prompt Engineering 101 - Makefile Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Complete setup (venv + dependencies + .env)"
	@echo "  make install      - Install Python dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only (fast)"
	@echo "  make test-module M=04  - Run tests for specific module"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run linting checks"
	@echo "  make format       - Format code with black"
	@echo "  make type-check   - Run type checking with mypy"
	@echo ""
	@echo "Development:"
	@echo "  make run-example E=01-fundamentals/examples/basic_prompting.py  - Run example"
	@echo "  make jupyter      - Start Jupyter notebook server"
	@echo "  make clean        - Clean cache and build files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-test  - Run tests in Docker"
	@echo "  make docker-run   - Run interactive Docker shell"
	@echo "  make docker-clean - Clean Docker images and containers"
	@echo ""

# Setup commands
setup: venv install config
	@echo "✅ Setup complete!"
	@echo "Activate virtual environment: source venv/bin/activate"

venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created"

install:
	@echo "Installing dependencies..."
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:
	@echo "Installing development dependencies..."
	. venv/bin/activate && pip install -r requirements-dev.txt
	@echo "✅ Development dependencies installed"

config:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env and add your API keys"; \
	else \
		echo "✅ .env already exists"; \
	fi

# Testing commands
test:
	@echo "Running all tests..."
	. venv/bin/activate && pytest -v

test-unit:
	@echo "Running unit tests (fast)..."
	. venv/bin/activate && pytest -v -m unit

test-module:
	@echo "Running tests for module $(M)..."
	. venv/bin/activate && pytest -v -m module$(M)

test-cov:
	@echo "Running tests with coverage..."
	. venv/bin/activate && pytest -v --cov=shared --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

test-watch:
	@echo "Running tests in watch mode..."
	. venv/bin/activate && pytest-watch -v

# Code quality commands
lint:
	@echo "Running linting checks..."
	. venv/bin/activate && flake8 shared/ tests/
	. venv/bin/activate && pylint shared/ tests/
	@echo "✅ Linting complete"

format:
	@echo "Formatting code with black..."
	. venv/bin/activate && black shared/ tests/ --line-length 100
	@echo "✅ Code formatted"

format-check:
	@echo "Checking code formatting..."
	. venv/bin/activate && black shared/ tests/ --check --line-length 100

type-check:
	@echo "Running type checks..."
	. venv/bin/activate && mypy shared/ tests/ --ignore-missing-imports
	@echo "✅ Type checking complete"

# Development commands
run-example:
	@if [ -z "$(E)" ]; then \
		echo "Usage: make run-example E=path/to/example.py"; \
		echo "Example: make run-example E=01-fundamentals/examples/basic_prompting.py --all"; \
	else \
		echo "Running example: $(E)"; \
		. venv/bin/activate && python $(E); \
	fi

jupyter:
	@echo "Starting Jupyter notebook server..."
	@echo "Access at: http://localhost:8888"
	. venv/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888

clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleaned"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf venv
	@echo "✅ Full clean complete"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t prompting-101:latest .
	@echo "✅ Docker image built"

docker-test:
	@echo "Running tests in Docker..."
	docker compose up test

docker-run:
	@echo "Running interactive Docker shell..."
	docker compose run dev

docker-jupyter:
	@echo "Starting Jupyter in Docker..."
	@echo "Access at: http://localhost:8888"
	docker compose up jupyter

docker-clean:
	@echo "Cleaning Docker resources..."
	docker compose down -v
	docker rmi prompting-101:latest 2>/dev/null || true
	@echo "✅ Docker cleaned"

# CI/CD simulation
ci: install-dev lint type-check test
	@echo "✅ CI checks passed"

# Quick start for new users
quickstart:
	@echo "🚀 Quick Start for Prompt Engineering 101"
	@echo "========================================="
	@echo ""
	@$(MAKE) setup
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Edit .env and add your OPENAI_API_KEY"
	@echo "3. Run first example: make run-example E=01-fundamentals/examples/basic_prompting.py --all"
	@echo "4. Run tests: make test-unit"
	@echo ""
	@echo "For more commands: make help"
