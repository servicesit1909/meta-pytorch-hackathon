# ============================================================================
# Stage 1: Builder -- install dependencies with uv for maximum speed
# ============================================================================
FROM python:3.11-slim AS builder

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Redirect all caches to /tmp for ephemeral builds
ENV UV_CACHE_DIR=/tmp/uv-cache \
    PIP_CACHE_DIR=/tmp/pip-cache \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy dependency specification first (Docker layer caching -- DRY)
COPY requirements.txt ./

# Install runtime dependencies from requirements.txt (single source of truth)
RUN uv pip install --system --no-cache --compile-bytecode -r requirements.txt

# ============================================================================
# Stage 2: Runtime -- minimal production image
# ============================================================================
FROM python:3.11-slim AS runtime

# HF Spaces metadata labels
LABEL org.opencontainers.image.title="OpSentrix SRE Harness" \
      org.opencontainers.image.description="Enterprise OpenEnv environment for SRE agent training" \
      org.opencontainers.image.version="1.1.0"

# Security: run as non-root user
RUN groupadd --gid 1000 opsentrix \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash opsentrix

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# -- Copy application source -----------------------------------------------
# inference.py MUST be at /app/ root (submission requirement)
COPY --chown=opsentrix:opsentrix inference.py ./inference.py
COPY --chown=opsentrix:opsentrix demo.py ./demo.py
COPY --chown=opsentrix:opsentrix models.py ./models.py
COPY --chown=opsentrix:opsentrix __init__.py ./__init__.py
COPY --chown=opsentrix:opsentrix openenv.yaml ./openenv.yaml
COPY --chown=opsentrix:opsentrix requirements.txt ./requirements.txt
COPY --chown=opsentrix:opsentrix server/ ./server/

# Ensure server has __init__.py
RUN touch /app/server/__init__.py 2>/dev/null || true

# Create outputs directory
RUN mkdir -p /app/outputs/logs /app/outputs/evals \
    && chown -R opsentrix:opsentrix /app/outputs

# Switch to non-root user
USER opsentrix

# Environment configuration
# PYTHONPATH set to /app so imports resolve correctly inside container
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPSENTRIX_HOST=0.0.0.0 \
    OPSENTRIX_PORT=7860 \
    OPSENTRIX_WORKERS=1 \
    LOG_LEVEL=INFO

# Expose the default port
EXPOSE 7860

# Health check (Kubernetes-compatible)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" || exit 1

# Resource constraint reminder:
# Run with: docker run --memory=8g --cpus=2 -p 7860:7860 -e HF_TOKEN=... opsentrix-sre

# Start the server -- inference.py is at /app/ root for direct execution
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]