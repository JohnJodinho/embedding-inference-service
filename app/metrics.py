"""Process-local service metrics.

Single Uvicorn worker, so plain module-level counters are sufficient and safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final


@dataclass
class Metrics:
    requests_total: int = 0
    requests_ok: int = 0
    requests_error: int = 0
    auth_failures: int = 0
    rate_limited: int = 0
    queue_full: int = 0
    model_load_failures: int = 0
    inference_errors: int = 0
    embeddings_total: int = 0

    _lock: Final[threading.Lock] = field(default_factory=threading.Lock, repr=False)

    def inc_request(self, ok: bool) -> None:
        with self._lock:
            self.requests_total += 1
            if ok:
                self.requests_ok += 1
            else:
                self.requests_error += 1

    def inc_auth_failure(self) -> None:
        with self._lock:
            self.auth_failures += 1

    def inc_rate_limited(self) -> None:
        with self._lock:
            self.rate_limited += 1

    def inc_queue_full(self) -> None:
        with self._lock:
            self.queue_full += 1

    def inc_model_load_failure(self) -> None:
        with self._lock:
            self.model_load_failures += 1

    def inc_inference_error(self) -> None:
        with self._lock:
            self.inference_errors += 1

    def inc_embeddings(self, count: int) -> None:
        with self._lock:
            self.embeddings_total += count

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_error": self.requests_error,
                "auth_failures": self.auth_failures,
                "rate_limited": self.rate_limited,
                "queue_full": self.queue_full,
                "model_load_failures": self.model_load_failures,
                "inference_errors": self.inference_errors,
                "embeddings_total": self.embeddings_total,
            }


metrics: Final[Metrics] = Metrics()
