# syntax=docker/dockerfile:1.7
#
# LedgerLens ingestion worker: consumes the Redis (arq) queue that POST /documents fills,
# and runs load -> OCR -> extract -> chunk -> embed -> store for each file. Build from the
# repo root:
#
#     docker build -f infra/docker/worker.Dockerfile -t ledgerlens-worker .
#
# PyTorch is the CPU wheel (the default one bundles CUDA, ~2 GB). Tesseract is installed for
# scanned pages. No model weights are baked in: the embedding model downloads on first use
# into HF_HOME, a volume, and the anomaly model is read from /app/models (a volume).
# Uploaded files are read from /app/data/uploads, the volume the API writes them to.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[storage,llm,ingestion,anomaly]"

FROM python:${PYTHON_VERSION}-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    KERAS_BACKEND=torch
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /cache/huggingface && chown app:app /cache/huggingface
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/src ./src
COPY configs ./configs
RUN mkdir -p data/uploads data/interim models && chown -R app:app data models
USER app
RUN python -c "import cv2, pytesseract, sentence_transformers, ledgerlens.workers.ingest_worker" \
    && tesseract --version | head -1
CMD ["python", "-m", "ledgerlens.workers.ingest_worker", "run"]
