# Multi-stage build for Prompt Engineering 101 curriculum
# This Dockerfile creates a production-ready container for running
# examples, exercises, and tests.

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY shared/ ./shared/
COPY 01-fundamentals/ ./01-fundamentals/
COPY 02-zero-shot-prompting/ ./02-zero-shot-prompting/
COPY 03-few-shot-learning/ ./03-few-shot-learning/
COPY 04-chain-of-thought/ ./04-chain-of-thought/
COPY 05-prompt-chaining/ ./05-prompt-chaining/
COPY 06-role-based-prompting/ ./06-role-based-prompting/
COPY 07-context-management/ ./07-context-management/
COPY 08-structured-outputs/ ./08-structured-outputs/
COPY 09-function-calling/ ./09-function-calling/
COPY 10-rag-basics/ ./10-rag-basics/
COPY 11-advanced-rag/ ./11-advanced-rag/
COPY 12-prompt-optimization/ ./12-prompt-optimization/
COPY 13-agent-design/ ./13-agent-design/
COPY 14-production-patterns/ ./14-production-patterns/
COPY tests/ ./tests/
COPY pytest.ini .
COPY .env.example .env.example

# Create runs directory for logging
RUN mkdir -p runs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 student && \
    chown -R student:student /app
USER student

# Default command: run tests
CMD ["pytest", "-v"]

# Alternative commands:
# Run specific module tests: docker run <image> pytest -m module01
# Run Python example: docker run <image> python 01-fundamentals/examples/basic_prompting.py --all
# Interactive shell: docker run -it <image> bash
