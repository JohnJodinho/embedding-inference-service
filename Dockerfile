# ---------------------------------------------------------------------------
# Stage 1: build + test
#   - installs pinned dependencies
#   - downloads both pinned model revisions into the HF cache
#   - runs the full pytest suite (fails the build if any test fails)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep model revisions in the Python registry as the single source of truth.
COPY app/__init__.py app/config.py ./app/

# Download both pinned models during the Hugging Face build (not locally).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    from app.config import MODEL_REGISTRY; \
    [SentenceTransformer(cfg['model_id'], revision=cfg['revision'], device='cpu') \
     for cfg in MODEL_REGISTRY.values()]"

# Keep test tooling out of the runtime environment.
COPY requirements-test.txt .
RUN pip install --no-cache-dir --target=/opt/test-deps -r requirements-test.txt

COPY app ./app
COPY tests ./tests

# Run the complete test suite; a failure fails the Docker build.
RUN PYTHONPATH=/opt/test-deps python -m pytest

# ---------------------------------------------------------------------------
# Stage 2: runtime
#   - copies tested app, installed dependencies and model cache
#   - runs as non-root UID 1000
#   - one Uvicorn worker on port 7860
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN useradd -m -u 1000 appuser

COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=1000:1000 /root/.cache/huggingface /home/appuser/.cache/huggingface
COPY --from=build --chown=1000:1000 /build/app ./app

USER appuser
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
