FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cpu \
    PORT=8000

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies
COPY requirements-render.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-render.txt

# Application code, configs, and models
COPY ew_core/ ew_core/
COPY experiments/ experiments/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .
RUN pip install -e .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request, os; p = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{p}/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
