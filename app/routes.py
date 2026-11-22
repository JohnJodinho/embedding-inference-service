"""Health, model-registry, and metrics routes."""

from __future__ import annotations

import time
from typing import Final

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import MODEL_REGISTRY, settings
from .logging import log
from .metrics import metrics
from .schemas import DetailErrorResponse, HealthResponse, ModelsResponse
from .scheduler import scheduler
from .security import authenticate, bearer_auth
from . import models

router: Final[APIRouter] = APIRouter()


@router.get("/health/live", response_class=PlainTextResponse)
async def live() -> PlainTextResponse:
    """Process liveness without sensitive information."""
    return PlainTextResponse("ok", status_code=200)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={
        503: {
            "model": HealthResponse,
            "description": "Models or inference worker are not ready",
        }
    },
)
async def ready() -> JSONResponse | dict[str, str]:
    """Report readiness only after both models and the worker are available."""
    if models.models_ready() and scheduler.ready():
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    dependencies=[Depends(bearer_auth)],
    responses={
        401: {
            "model": DetailErrorResponse,
            "description": "Missing or invalid Bearer API key",
        }
    },
)
async def list_models(request: Request) -> dict[str, object]:
    """Return supported model aliases and dimensions. Authenticated."""
    started_at = time.monotonic()
    request_id = request.state.request_id
    client_label = authenticate(request, None)
    log.info(
        "models.list",
        request_id=request_id,
        client_label=client_label,
        model_alias=None,
        batch_size=0,
        queue_wait_ms=0,
        inference_ms=0,
        total_ms=round((time.monotonic() - started_at) * 1000, 2),
        status="ok",
        error_code=None,
    )
    return {
        "object": "list",
        "data": [
            {
                "alias": alias,
                "model_id": config["model_id"],
                "revision": config["revision"],
                "dimensions": config["dimensions"],
                "license": config["license"],
                "max_input_texts": (
                    settings.max_bge_texts_per_request
                    if alias == "bge-base-768-v1"
                    else settings.max_texts_per_request
                ),
                "max_tokens_per_text": settings.max_tokens_per_text,
                "max_chars_per_text": settings.max_chars_per_text,
            }
            for alias, config in MODEL_REGISTRY.items()
        ],
    }


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(bearer_auth)],
    responses={
        401: {
            "model": DetailErrorResponse,
            "description": "Missing or invalid Bearer API key",
        }
    },
)
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """Return lightweight process-local metrics. Authenticated."""
    started_at = time.monotonic()
    request_id = request.state.request_id
    client_label = authenticate(request, None)
    log.info(
        "metrics.read",
        request_id=request_id,
        client_label=client_label,
        model_alias=None,
        batch_size=0,
        queue_wait_ms=0,
        inference_ms=0,
        total_ms=round((time.monotonic() - started_at) * 1000, 2),
        status="ok",
        error_code=None,
    )
    return PlainTextResponse(
        metrics.prometheus_text(),
        status_code=200,
        media_type="text/plain; version=0.0.4",
    )
