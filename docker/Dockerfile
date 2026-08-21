# syntax=docker/dockerfile:1
# Multi-stage build for the FinVet app tier. Two targets, `api` and `ui`,
# share a base layer whose venv is built reproducibly from uv.lock.
#
#   docker build -f docker/Dockerfile --target api -t finvet-api .
#   docker build -f docker/Dockerfile --target ui  -t finvet-ui  .

FROM python:3.13-slim AS base

# Pinned uv (the version that resolved uv.lock) for reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (cached until pyproject/uv.lock change), without
# the project itself so source edits don't bust the dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the finvet package itself. README is referenced by pyproject metadata.
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# --- API image ---------------------------------------------------------------
FROM base AS api
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"
CMD ["uvicorn", "finvet.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI image ----------------------------------------------------------------
# ui/app.py imports the installed finvet package (for __version__); the shared
# base provides it. WORKDIR /app so `streamlit run ui/app.py` puts ui/ on sys.path.
FROM base AS ui
COPY --chown=appuser:appuser ui ./ui
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)"
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
