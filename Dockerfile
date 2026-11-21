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
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Download both pinned models during the Hugging Face build (not locally).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', \
                        revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', \
                        device='cpu'); \
    SentenceTransformer('BAAI/bge-base-en-v1.5', \
                        revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', \
                        device='cpu')"

# Run the complete test suite; a failure fails the Docker build.
RUN pytest

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
COPY --from=build /root/.cache/huggingface /home/appuser/.cache/huggingface
COPY --chown=appuser:appuser app ./app

RUN chown -R appuser:appuser /app /home/appuser/.cache

USER appuser
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
