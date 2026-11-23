# syntax=docker/dockerfile:1.7
#
# LedgerLens API: FastAPI + uvicorn, running the agent graph (its tools come from the MCP
# server). Build from the repo root:
#
#     docker build -f infra/docker/api.Dockerfile -t ledgerlens-api .
#
# Multi-stage: dependencies are installed into a venv in `build`; `runtime` gets the venv
# and the source only, runs as an unprivileged user, and holds no model weights or data
# (uploads go to a volume at /app/data/uploads). The repo layout is kept under /app, so
# configs/ and data/ resolve exactly as in a checkout. The same image runs migrations:
#
#     docker compose run --rm api alembic upgrade head

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[storage,llm,agents]"

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/src ./src
COPY configs ./configs
COPY migrations ./migrations
COPY alembic.ini ./
COPY scripts ./scripts
RUN mkdir -p data/uploads data/interim && chown -R app:app data
USER app
# fails the build if a dependency is missing, instead of at the first request
RUN python -c "import ledgerlens.api.main"
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "ledgerlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
