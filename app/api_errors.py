"""Safe, structured HTTP errors for embedding requests."""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi.responses import JSONResponse

from .logging import log
from .metrics import metrics


@dataclass
class APIError(Exception):
    status_code: int
    code: str
    event: str
    model_alias: str | None = None
    batch_size: int = 0
    retry_after: int | None = None
    queue_wait_ms: float = 0
    inference_ms: float = 0

    def __post_init__(self) -> None:
        Exception.__init__(self, self.code)


def reject_request(
    error: APIError,
    request_id: str,
    client_label: str,
    started_at: float,
) -> JSONResponse:
    metrics.inc_request(ok=False)
    if error.status_code == 429:
        metrics.inc_rate_limited()
    log.warning(
        error.event,
        request_id=request_id,
        client_label=client_label,
        model_alias=error.model_alias,
        batch_size=error.batch_size,
        queue_wait_ms=round(error.queue_wait_ms, 2),
        inference_ms=round(error.inference_ms, 2),
        total_ms=round((time.monotonic() - started_at) * 1000, 2),
        status="error",
        error_code=error.code,
    )
    headers = (
        {"Retry-After": str(error.retry_after)}
        if error.retry_after is not None
        else None
    )
    return JSONResponse(
        status_code=error.status_code,
        content={"error": error.code},
        headers=headers,
    )
