FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cpu

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libhdf5-dev \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU PyTorch)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# App code
COPY configs/ configs/
COPY src/ src/
COPY pyproject.toml .
COPY scripts/ scripts/

# Data and checkpoints are mounted at runtime, not baked in
# Expose API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
