"""FastAPI application: embedding inference service.

One process, one Uvicorn worker. Both models are loaded once in lifespan and a
single CPU inference worker serves two bounded queues.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import MODEL_REGISTRY, get_model_config, settings
from .logging import log
from .metrics import metrics
from .models import ModelNotReadyError, get_model, load_model
from .scheduler import QueueFullError, scheduler
from .security import (
    authenticate,
    rate_limiter,
    validate_request_size,
    validate_texts,
)

app: Final[FastAPI] = FastAPI(title="Embedding Inference Service", version="1.0.0")


def error_response(status_code: int, code: str, retry_after: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(status_code=status_code, content={"error": code}, headers=headers)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load both models once and start the inference worker."""
    for alias in ("minilm-384-v1", "bge-base-768-v1"):
        load_model(alias)
    scheduler.start()
    log.info("service.started")
    yield
    await scheduler.stop()
    log.info("service.stopped")


app.router.lifespan_context = lifespan


@app.get("/health/live")
async def live() -> PlainTextResponse:
    """Process liveness. No authentication, no sensitive info."""
    return PlainTextResponse("ok", status_code=200)


@app.get("/health/ready")
async def ready() -> JSONResponse:
    """Model and worker readiness. Both models must be loaded."""
    from .models import _MODEL_CACHE

    models_ready = all(alias in _MODEL_CACHE for alias in ("minilm-384-v1", "bge-base-768-v1"))
    worker_ready = scheduler.ready()
    if models_ready and worker_ready:
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Return supported model aliases and dimensions. Authenticated."""
    client_label = authenticate(request, request.query_params.get("client_id"))
    log.info("models.list", client_label=client_label)
    return JSONResponse(
        status_code=200,
        content={
            "object": "list",
            "data": [
                {
                    "alias": alias,
                    "model_id": cfg["model_id"],
                    "revision": cfg["revision"],
                    "dimensions": cfg["dimensions"],
                    "license": cfg["license"],
                }
                for alias, cfg in MODEL_REGISTRY.items()
            ],
        },
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    """Embed one or many texts. Authenticated."""
    start = time.monotonic()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    # Body size
    body = await request.body()
    validate_request_size(len(body))

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.invalid_json",
            request_id=request_id,
            error_code="invalid_json",
        )
        return error_response(400, "invalid_json")

    model_alias = payload.get("model")
    input_value = payload.get("input")
    input_type = payload.get("input_type")
    normalize_embeddings = bool(payload.get("normalize_embeddings", False))

    # Authenticate early to enforce auth on this endpoint.
    client_label = authenticate(request, None)

    # Rate limit per client.
    if not rate_limiter.check(client_label):
        metrics.inc_rate_limited()
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.rate_limited",
            request_id=request_id,
            client_label=client_label,
            error_code="rate_limited",
        )
        return error_response(429, "rate_limited", retry_after=60)

    # Validate model alias.
    try:
        cfg = get_model_config(model_alias)
    except (KeyError, TypeError):
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.unknown_model",
            request_id=request_id,
            client_label=client_label,
            model=model_alias,
            error_code="unknown_model",
        )
        return error_response(400, "unknown_model")

    # Validate input shape.
    if isinstance(input_value, str):
        texts = [input_value]
    elif isinstance(input_value, list) and all(isinstance(t, str) for t in input_value):
        texts = input_value
    else:
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.invalid_input",
            request_id=request_id,
            client_label=client_label,
            model=model_alias,
            error_code="invalid_input",
        )
        return error_response(400, "invalid_input")

    try:
        validate_texts(texts)
    except Exception as exc:  # noqa: BLE001
        metrics.inc_request(ok=False)
        status_code = getattr(exc, "status_code", 400) or 400
        log.warning(
            "embeddings.validation_failed",
            request_id=request_id,
            client_label=client_label,
            model=model_alias,
            error_code=getattr(exc, "detail", "validation_failed"),
        )
        return error_response(status_code, getattr(exc, "detail", "validation_failed"))

    # Character/token-limit proxy (do not silently truncate beyond the limit).
    for text in texts:
        if len(text) > settings.max_tokens_per_text:
            metrics.inc_request(ok=False)
            log.warning(
                "embeddings.text_too_long",
                request_id=request_id,
                client_label=client_label,
                model=model_alias,
                error_code="text_too_long",
            )
            return error_response(400, "text_too_long")

    if input_type not in cfg["supports_input_type"]:
        error_code = (
            "input_type_required"
            if cfg["requires_input_type"] and input_type is None
            else "invalid_input_type"
        )
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.invalid_input_type",
            request_id=request_id,
            client_label=client_label,
            model=model_alias,
            error_code=error_code,
        )
        return error_response(400, error_code)

    # Submit to the bounded scheduler.
    try:
        vectors = await scheduler.submit(
            alias=model_alias,
            texts=texts,
            normalize_embeddings=normalize_embeddings,
            input_type=input_type,
        )
    except QueueFullError:
        metrics.inc_request(ok=False)
        log.warning(
            "embeddings.queue_full",
            request_id=request_id,
            client_label=client_label,
            model=model_alias,
            error_code="queue_full",
        )
        return error_response(503, "queue_full", retry_after=5)

    total_ms = (time.monotonic() - start) * 1000
    metrics.inc_request(ok=True)
    log.info(
        "embeddings.ok",
        request_id=request_id,
        client_label=client_label,
        model_alias=model_alias,
        batch_size=len(texts),
        total_ms=round(total_ms, 2),
        status="ok",
    )

    return JSONResponse(
        status_code=200,
        content={
            "object": "list",
            "model": model_alias,
            "dimensions": cfg["dimensions"],
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i, vector in enumerate(vectors)
            ],
            "usage": {"input_texts": len(texts)},
        },
    )


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """Lightweight process-local metrics. Authenticated."""
    authenticate(request, None)
    snap = metrics.snapshot()
    lines = [f"{k} {v}" for k, v in snap.items()]
    return PlainTextResponse("\n".join(lines) + "\n", status_code=200)

