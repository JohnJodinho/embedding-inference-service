"""Thread-safe process-local counters, gauges, and latency histograms."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Final

MODEL_ALIASES: Final[tuple[str, ...]] = (
    "minilm-384-v1",
    "bge-base-768-v1",
)
HISTOGRAM_BUCKETS_SECONDS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
    60,
    90,
    120,
)


@dataclass
class _Histogram:
    bucket_counts: list[int] = field(
        default_factory=lambda: [0] * len(HISTOGRAM_BUCKETS_SECONDS)
    )
    count: int = 0
    total_seconds: float = 0

    def observe(self, seconds: float) -> None:
        value = max(0.0, seconds)
        self.count += 1
        self.total_seconds += value
        for index, boundary in enumerate(HISTOGRAM_BUCKETS_SECONDS):
            if value <= boundary:
                self.bucket_counts[index] += 1


def _new_model_histograms() -> dict[str, _Histogram]:
    return {alias: _Histogram() for alias in MODEL_ALIASES}


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
    queue_depths: dict[str, int] = field(
        default_factory=lambda: {alias: 0 for alias in MODEL_ALIASES}
    )
    request_duration: dict[str, _Histogram] = field(
        default_factory=_new_model_histograms
    )
    queue_wait_duration: dict[str, _Histogram] = field(
        default_factory=_new_model_histograms
    )
    inference_duration: dict[str, _Histogram] = field(
        default_factory=_new_model_histograms
    )
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

    def set_queue_depth(self, alias: str, depth: int) -> None:
        if alias not in self.queue_depths or depth < 0:
            raise ValueError("Invalid model alias or queue depth")
        with self._lock:
            self.queue_depths[alias] = depth

    def observe_request_duration(self, alias: str, duration_ms: float) -> None:
        self._observe(self.request_duration, alias, duration_ms)

    def observe_scheduler_durations(
        self,
        alias: str,
        queue_wait_ms: float,
        inference_ms: float,
    ) -> None:
        with self._lock:
            self._observe_locked(self.queue_wait_duration, alias, queue_wait_ms)
            self._observe_locked(self.inference_duration, alias, inference_ms)

    def _observe(
        self,
        histograms: dict[str, _Histogram],
        alias: str,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._observe_locked(histograms, alias, duration_ms)

    @staticmethod
    def _observe_locked(
        histograms: dict[str, _Histogram],
        alias: str,
        duration_ms: float,
    ) -> None:
        if alias not in histograms:
            raise ValueError(f"Unknown model alias: {alias}")
        histograms[alias].observe(duration_ms / 1000)

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

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# TYPE requests_total counter",
                f"requests_total {self.requests_total}",
                "# TYPE requests_ok counter",
                f"requests_ok {self.requests_ok}",
                "# TYPE requests_error counter",
                f"requests_error {self.requests_error}",
                "# TYPE auth_failures counter",
                f"auth_failures {self.auth_failures}",
                "# TYPE rate_limited counter",
                f"rate_limited {self.rate_limited}",
                "# TYPE queue_full counter",
                f"queue_full {self.queue_full}",
                "# TYPE model_load_failures counter",
                f"model_load_failures {self.model_load_failures}",
                "# TYPE inference_errors counter",
                f"inference_errors {self.inference_errors}",
                "# TYPE embeddings_total counter",
                f"embeddings_total {self.embeddings_total}",
                "# TYPE embedding_queue_depth gauge",
            ]
            for alias, depth in self.queue_depths.items():
                lines.append(f'embedding_queue_depth{{model="{alias}"}} {depth}')
            self._append_histogram(
                lines,
                "embedding_request_duration_seconds",
                self.request_duration,
            )
            self._append_histogram(
                lines,
                "embedding_queue_wait_seconds",
                self.queue_wait_duration,
            )
            self._append_histogram(
                lines,
                "embedding_inference_duration_seconds",
                self.inference_duration,
            )
            return "\n".join(lines) + "\n"

    @staticmethod
    def _append_histogram(
        lines: list[str],
        name: str,
        histograms: dict[str, _Histogram],
    ) -> None:
        lines.append(f"# TYPE {name} histogram")
        for alias, histogram in histograms.items():
            for boundary, count in zip(
                HISTOGRAM_BUCKETS_SECONDS,
                histogram.bucket_counts,
            ):
                lines.append(
                    f'{name}_bucket{{model="{alias}",le="{boundary:g}"}} {count}'
                )
            lines.append(f'{name}_bucket{{model="{alias}",le="+Inf"}} {histogram.count}')
            lines.append(f'{name}_sum{{model="{alias}"}} {histogram.total_seconds:g}')
            lines.append(f'{name}_count{{model="{alias}"}} {histogram.count}')


metrics: Final[Metrics] = Metrics()
