"""Authentication, rate limiting and input validation.

Two distinct client secrets (RAG, ZIT) are compared in constant time. Keys come
from Hugging Face Space Secrets at runtime and are never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from dataclasses import dataclass, field
from typing import Final

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer

from .config import settings
from .logging import log
from .metrics import metrics

bearer_auth: Final[HTTPBearer] = HTTPBearer(
    scheme_name="BearerAuth",
    description="Use a configured RAG_API_KEY or ZIT_API_KEY.",
    auto_error=False,
)


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two secrets in constant time."""
    return hmac.compare_digest(hashlib.sha256(a.encode()).hexdigest(), hashlib.sha256(b.encode()).hexdigest())


def resolve_api_key(request: Request) -> str:
    """Extract the bearer API key from the Authorization header."""
    authorization = request.headers.get("authorization", "")
    scheme, separator, api_key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_auth",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key


def authenticate(request: Request, client_id: str | None) -> str:
    """Validate the API key and return the client label ('rag' or 'zit').

    Raises HTTPException(401) on missing/unknown keys.
    """
    try:
        api_key = resolve_api_key(request)
    except HTTPException:
        metrics.inc_auth_failure()
        metrics.inc_request(ok=False)
        log.warning(
            "auth.failed",
            request_id=getattr(getattr(request, "state", None), "request_id", "unknown"),
            client_label="unknown",
            model_alias=None,
            batch_size=0,
            queue_wait_ms=0,
            inference_ms=0,
            total_ms=0,
            status="error",
            error_code="invalid_auth",
        )
        raise

    if settings.rag_api_key and constant_time_compare(api_key, settings.rag_api_key):
        client_label = "rag"
    elif settings.zit_api_key and constant_time_compare(api_key, settings.zit_api_key):
        client_label = "zit"
    else:
        metrics.inc_auth_failure()
        metrics.inc_request(ok=False)
        log.warning(
            "auth.failed",
            request_id=getattr(getattr(request, "state", None), "request_id", "unknown"),
            client_label="unknown",
            model_alias=None,
            batch_size=0,
            queue_wait_ms=0,
            inference_ms=0,
            total_ms=0,
            status="error",
            error_code="invalid_key",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_auth",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if client_id and client_id != client_label:
        metrics.inc_auth_failure()
        metrics.inc_request(ok=False)
        # Client label hint must match the derived consumer.
        log.warning(
            "auth.client_mismatch",
            request_id=getattr(getattr(request, "state", None), "request_id", "unknown"),
            client_label=client_label,
            model_alias=None,
            batch_size=0,
            queue_wait_ms=0,
            inference_ms=0,
            total_ms=0,
            status="error",
            error_code="client_mismatch",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="client_mismatch",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return client_label


@dataclass
class _TokenBucket:
    capacity: int
    fill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.fill_rate)
        self.last = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Per-client token-bucket rate limiter."""

    def __init__(self) -> None:
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, client_label: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(client_label)
            if bucket is None:
                rpm = (
                    settings.zit_rate_limit_per_min
                    if client_label == "zit"
                    else settings.rag_rate_limit_per_min
                )
                capacity = max(1, rpm)
                bucket = _TokenBucket(capacity=capacity, fill_rate=rpm / 60.0)
                self._buckets[client_label] = bucket
            return bucket.consume(1)


rate_limiter: Final[RateLimiter] = RateLimiter()


def validate_request_size(body_bytes: int) -> None:
    if body_bytes > settings.max_request_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="request_too_large",
        )


def validate_texts(texts: list[str], max_texts: int | None = None) -> None:
    if not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty_input")
    text_limit = max_texts if max_texts is not None else settings.max_texts_per_request
    if len(texts) > text_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="too_many_texts",
        )
    for text in texts:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="empty_input",
            )
        if len(text) > settings.max_chars_per_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text_too_long",
            )
