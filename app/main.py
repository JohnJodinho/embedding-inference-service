"""FastAPI application setup and process lifecycle."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI, Request, Response

from .embeddings_api import router as embeddings_router
from .logging import log
from .models import load_model
from .routes import router as service_router
from .scheduler import scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Load both models once and start the inference worker."""
    for alias in ("minilm-384-v1", "bge-base-768-v1"):
        load_model(alias)
    scheduler.start()
    log.info("service.started")
    try:
        yield
    finally:
        await scheduler.stop()
        log.info("service.stopped")


app: Final[FastAPI] = FastAPI(
    title="Embedding Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(service_router)
app.include_router(embeddings_router)
