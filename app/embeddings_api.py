"""Embedding request parsing, validation, and inference route."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .api_errors import APIError, reject_request
from .config import get_model_config, settings
from .logging import log
from .metrics import metrics
from .models import ModelNotReadyError, count_tokens, prepare_input
from .schemas import (
    EMBEDDING_REQUEST_SCHEMA,
    DetailErrorResponse,
    EmbeddingRequest,
    EmbeddingsResponse,
    ErrorResponse,
)
from .scheduler import (
    InferenceError,
    InferenceTimeoutError,
    InferenceResult,
    QueueFullError,
    RequestTimeoutError,
    SchedulerNotReadyError,
    scheduler,
)
from .security import authenticate, bearer_auth, rate_limiter, validate_texts

router: Final[APIRouter] = APIRouter()


async def _read_payload(request: Request) -> dict[str, object]:
    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > settings.max_request_bytes:
            raise APIError(413, "request_too_large", "embeddings.request_too_large")
        chunks.append(chunk)
    body = b"".join(chunks)
    if not body.strip():
        raise APIError(400, "empty_body", "embeddings.empty_body")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise APIError(415, "unsupported_media_type", "embeddings.unsupported_media_type")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise APIError(400, "invalid_json", "embeddings.invalid_json") from exc
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_body", "embeddings.invalid_body")
    return payload


def _validate_payload(
    payload: dict[str, object],
) -> tuple[EmbeddingRequest, dict[str, object]]:
    try:
        embedding_request = EmbeddingRequest.model_validate(payload)
    except ValidationError as exc:
        raise APIError(400, "invalid_input", "embeddings.invalid_input") from exc
    try:
        model_config = get_model_config(embedding_request.model)
    except KeyError as exc:
        raise APIError(400, "unknown_model", "embeddings.unknown_model") from exc
    if embedding_request.input_type not in model_config["supports_input_type"]:
        error_code = (
            "input_type_required"
            if model_config["requires_input_type"] and embedding_request.input_type is None
            else "invalid_input_type"
        )
        raise APIError(
            400,
            error_code,
            "embeddings.invalid_input_type",
            embedding_request.model,
            len(embedding_request.texts),
        )
    request_text_limit = (
        settings.max_bge_texts_per_request
        if embedding_request.model == "bge-base-768-v1"
        else settings.max_texts_per_request
    )
    try:
        validate_texts(embedding_request.texts, max_texts=request_text_limit)
    except HTTPException as exc:
        code = exc.detail if isinstance(exc.detail, str) else "invalid_input"
        raise APIError(
            exc.status_code,
            code,
            "embeddings.validation_failed",
            embedding_request.model,
            len(embedding_request.texts),
        ) from exc
    return embedding_request, model_config


def _validate_token_limits_sync(request: EmbeddingRequest) -> None:
    for text in request.texts:
        try:
            prepared_text = prepare_input(request.model, request.input_type, text)
            token_count = count_tokens(request.model, prepared_text)
        except ModelNotReadyError as exc:
            raise APIError(
                503,
                "model_not_ready",
                "embeddings.model_not_ready",
                request.model,
                len(request.texts),
            ) from exc
        except (UnicodeError, ValueError) as exc:
            raise APIError(
                400,
                "invalid_input",
                "embeddings.tokenization_failed",
                request.model,
                len(request.texts),
            ) from exc
        if token_count > settings.max_tokens_per_text:
            raise APIError(
                400,
                "too_many_tokens",
                "embeddings.too_many_tokens",
                request.model,
                len(request.texts),
            )


async def _validate_token_limits(request: EmbeddingRequest) -> None:
    await asyncio.to_thread(_validate_token_limits_sync, request)


async def _submit_for_inference(
    request: EmbeddingRequest,
    request_id: str,
    client_label: str,
) -> InferenceResult:
    try:
        return await scheduler.submit(
            alias=request.model,
            texts=request.texts,
            normalize_embeddings=request.normalize_embeddings,
            input_type=request.input_type,
            request_id=request_id,
            client_label=client_label,
        )
    except QueueFullError as exc:
        raise APIError(
            503,
            "queue_full",
            "embeddings.queue_full",
            request.model,
            len(request.texts),
            retry_after=5,
        ) from exc
    except SchedulerNotReadyError as exc:
        raise APIError(
            503,
            "not_ready",
            "embeddings.not_ready",
            request.model,
            len(request.texts),
            retry_after=5,
        ) from exc
    except RequestTimeoutError as exc:
        raise APIError(
            504,
            "request_timeout",
            "embeddings.timeout",
            request.model,
            len(request.texts),
            queue_wait_ms=exc.queue_wait_ms,
            inference_ms=exc.inference_ms,
        ) from exc
    except InferenceTimeoutError as exc:
        raise APIError(
            504,
            "inference_timeout",
            "embeddings.timeout",
            request.model,
            len(request.texts),
            queue_wait_ms=exc.queue_wait_ms,
            inference_ms=exc.inference_ms,
        ) from exc
    except (InferenceError, ModelNotReadyError) as exc:
        raise APIError(
            503,
            "inference_unavailable",
            "embeddings.inference_failed",
            request.model,
            len(request.texts),
        ) from exc


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingsResponse,
    dependencies=[Depends(bearer_auth)],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid JSON or embedding request"},
        401: {
            "model": DetailErrorResponse,
            "description": "Missing or invalid Bearer API key",
        },
        413: {
            "model": ErrorResponse,
            "description": "Request body exceeds the configured size limit",
        },
        415: {"model": ErrorResponse, "description": "Expected application/json"},
        429: {"model": ErrorResponse, "description": "Client rate limit exceeded"},
        503: {
            "model": ErrorResponse,
            "description": "Service is not ready or inference queue is full",
        },
        504: {"model": ErrorResponse, "description": "Request or inference timeout"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": EMBEDDING_REQUEST_SCHEMA,
                    "examples": {
                        "minilm": {
                            "summary": "MiniLM embedding",
                            "value": {
                                "model": "minilm-384-v1",
                                "input": "hello world",
                                "normalize_embeddings": True,
                            },
                        },
                        "bge_query": {
                            "summary": "BGE retrieval query",
                            "value": {
                                "model": "bge-base-768-v1",
                                "input": "What courses does ZIT offer?",
                                "input_type": "query",
                                "normalize_embeddings": True,
                            },
                        },
                    },
                }
            },
        }
    },
)
async def embeddings(request: Request) -> dict[str, object] | JSONResponse:
    """Embed one or many texts. Authenticated."""
    started_at = time.monotonic()
    request_id = request.state.request_id
    client_label = authenticate(request, None)
    if not rate_limiter.check(client_label):
        return reject_request(
            APIError(
                429,
                "rate_limited",
                "embeddings.rate_limited",
                retry_after=60,
            ),
            request_id,
            client_label,
            started_at,
        )
    try:
        payload = await _read_payload(request)
        embedding_request, model_config = _validate_payload(payload)
        await _validate_token_limits(embedding_request)
        result = await _submit_for_inference(
            embedding_request,
            request_id,
            client_label,
        )
    except APIError as exc:
        return reject_request(exc, request_id, client_label, started_at)

    total_ms = (time.monotonic() - started_at) * 1000
    metrics.inc_request(ok=True)
    metrics.observe_request_duration(embedding_request.model, total_ms)
    log.info(
        "embeddings.ok",
        request_id=request_id,
        client_label=client_label,
        model_alias=embedding_request.model,
        batch_size=len(embedding_request.texts),
        queue_wait_ms=result.queue_wait_ms,
        inference_ms=result.inference_ms,
        total_ms=round(total_ms, 2),
        status="ok",
        error_code=None,
    )
    return {
        "object": "list",
        "model": embedding_request.model,
        "dimensions": model_config["dimensions"],
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(result.vectors)
        ],
        "usage": {"input_texts": len(embedding_request.texts)},
    }
