# syntax=docker/dockerfile:1.7
#
# LedgerLens MCP server: the agents' tools (search, records, reconciliation, anomaly
# scores, reminders) over streamable HTTP on :8765/mcp. Build from the repo root:
#
#     docker build -f infra/docker/mcp.Dockerfile -t ledgerlens-mcp .
#
# Search embeds queries and reranks hits, so PyTorch is here too (the CPU wheel). The
# models download on first use into HF_HOME, a volume; nothing is baked in. It listens on
# 0.0.0.0 inside its container; only the API (MCP_URL=http://mcp:8765/mcp) reaches it.

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
    pip install -e ".[storage,llm,agents,ingestion,anomaly]"

FROM python:${PYTHON_VERSION}-slim AS runtime
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
RUN mkdir -p data/interim models && chown -R app:app data models
USER app
RUN python -c "import sentence_transformers, ledgerlens.mcp_server.server"
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=10 \
    CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', 8765), timeout=4).close()"]
CMD ["python", "-m", "ledgerlens.mcp_server.server", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
