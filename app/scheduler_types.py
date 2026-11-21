"""Scheduler result, request, and failure types."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


class QueueFullError(RuntimeError):
    """Raised when a bounded model queue cannot accept more work."""


class SchedulerNotReadyError(RuntimeError):
    """Raised when the scheduler is not accepting requests."""


class RequestTimeoutError(TimeoutError):
    """Raised when a request exceeds its total queue-and-inference deadline."""

    def __init__(self, queue_wait_ms: float, inference_ms: float) -> None:
        super().__init__("Request timed out")
        self.queue_wait_ms = queue_wait_ms
        self.inference_ms = inference_ms


class InferenceTimeoutError(TimeoutError):
    """Raised when model inference exceeds its configured deadline."""

    def __init__(self, queue_wait_ms: float, inference_ms: float) -> None:
        super().__init__("Model inference timed out")
        self.queue_wait_ms = queue_wait_ms
        self.inference_ms = inference_ms


class InferenceError(RuntimeError):
    """Raised when inference fails or returns invalid vectors."""


@dataclass(frozen=True)
class InferenceResult:
    vectors: list[list[float]]
    queue_wait_ms: float
    inference_ms: float


@dataclass
class _RequestItem:
    alias: str
    texts: list[str]
    normalize_embeddings: bool
    input_type: str | None
    request_id: str
    client_label: str
    enqueued_at: float
    future: asyncio.Future[InferenceResult]
    queue_wait_ms: float | None = None
    inference_started_at: float | None = None


def fail_future(entries: list[_RequestItem], error: BaseException) -> None:
    for item in entries:
        if not item.future.done():
            item.future.set_exception(error)


def create_request_item(
    alias: str,
    texts: list[str],
    normalize_embeddings: bool,
    input_type: str | None,
    request_id: str,
    client_label: str,
    future: asyncio.Future[InferenceResult],
) -> _RequestItem:
    return _RequestItem(
        alias=alias,
        texts=texts,
        normalize_embeddings=normalize_embeddings,
        input_type=input_type,
        request_id=request_id,
        client_label=client_label,
        enqueued_at=time.monotonic(),
        future=future,
    )
