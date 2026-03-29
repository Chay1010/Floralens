# ──────────────────────────────────────────────
# FloraLens — Production Docker Image
# ──────────────────────────────────────────────
# Multi-stage build: builder for pip install, slim runtime image.
#
# Build:  docker build -t floralens:latest .
# Run:    docker run -p 8000:8000 floralens:latest
# Test:   curl http://localhost:8000/health
# ──────────────────────────────────────────────

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim AS runtime

# Security: run as non-root user
RUN groupadd -r floralens && useradd -r -g floralens -d /app floralens

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY training/ ./training/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/data /app/experiments /app/models/checkpoints && \
    chown -R floralens:floralens /app

USER floralens

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
