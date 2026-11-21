"""Authentication, rate limiting and input validation.

Two distinct client secrets (RAG, ZIT) are compared in constant time. Keys come
from Hugging Face Space Secrets at runtime and are never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Final

from fastapi import HTTPException, Request, status

from .config import Settings, settings
from .logging import log

_CLIENT_HEADER: Final[str] = "X-Client-Id"


def constant_time_compare(a: str, b: str) -> bool:
    """Compare two secrets in constant time."""
    return hmac.compare_digest(hashlib.sha256(a.encode()).hexdigest(), hashlib.sha256(b.encode()).hexdigest())


def resolve_api_key(request: Request) -> str:
    """Extract the bearer API key from the Authorization header."""
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_auth",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return authorization[len("Bearer "):]


def authenticate(request: Request, client_id: str | None) -> str:
    """Validate the API key and return the client label ('rag' or 'zit').

    Raises HTTPException(401) on missing/unknown keys.
    """
    api_key = resolve_api_key(request)

    if constant_time_compare(api_key, settings.rag_api_key):
        client_label = "rag"
    elif constant_time_compare(api_key, settings.zit_api_key):
        client_label = "zit"
    else:
        log.warning("auth.failed", client_label=client_id or "unknown", error_code="invalid_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_auth",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if client_id and client_id != client_label:
        # Client label hint must match the derived consumer.
        log.warning("auth.client_mismatch", client_label=client_label, provided=client_id, error_code="client_mismatch")
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
    last: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
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
        self._lock = __import__("threading").Lock()

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


def validate_texts(texts: list[str]) -> None:
    if not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty_input")
    if len(texts) > settings.max_texts_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="too_many_texts",
        )
    for text in texts:
        if len(text) > settings.max_chars_per_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text_too_long",
            )
