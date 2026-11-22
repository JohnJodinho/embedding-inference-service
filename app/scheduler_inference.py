"""Prepare, execute, and validate one scheduler inference batch."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from .config import Settings, get_model_config
from .logging import log
from .metrics import metrics
from .scheduler_types import (
    InferenceError,
    InferenceResult,
    InferenceTimeoutError,
    SchedulerNotReadyError,
    _RequestItem,
    fail_future,
)


class InferenceBatchProcessor:
    def __init__(self, config: Settings) -> None:
        self._settings = config

    async def run(
        self,
        alias: str,
        entries: list[_RequestItem],
        normalize_embeddings: bool,
        executor: ThreadPoolExecutor | None,
    ) -> None:
        texts, ranges = self._prepare_inputs(alias, entries)
        if not texts:
            return
        queue_wait = self._mark_inference_started(ranges)
        try:
            vectors, inference_ms = await self._execute_batch(
                alias,
                texts,
                normalize_embeddings,
                queue_wait,
                executor,
            )
        except Exception as exc:  # noqa: BLE001 - resolve every waiting request
            self._handle_inference_failure(alias, ranges, queue_wait, exc)
            return
        try:
            self._resolve_vectors(alias, texts, ranges, queue_wait, vectors, inference_ms)
        except (TypeError, ValueError):
            fail_future(
                [item for item, _start, _end in ranges],
                InferenceError("Encoder returned invalid vectors"),
            )
            metrics.inc_inference_error()
            self._log_batch_failure(
                alias,
                ranges,
                queue_wait,
                inference_ms,
                "invalid_vectors",
            )

    def _prepare_inputs(
        self,
        alias: str,
        entries: list[_RequestItem],
    ) -> tuple[list[str], list[tuple[_RequestItem, int, int]]]:
        from .models import prepare_input

        texts: list[str] = []
        ranges: list[tuple[_RequestItem, int, int]] = []
        for item in entries:
            if item.future.cancelled():
                continue
            start = len(texts)
            try:
                prepared = [
                    prepare_input(alias, item.input_type, text)
                    for text in item.texts
                ]
            except ValueError as exc:
                fail_future([item], exc)
                continue
            texts.extend(prepared)
            ranges.append((item, start, len(texts)))
        return texts, ranges

    def _mark_inference_started(
        self,
        ranges: list[tuple[_RequestItem, int, int]],
    ) -> dict[int, float]:
        queue_wait = {
            id(item): max(0.0, (time.monotonic() - item.enqueued_at) * 1000)
            for item, _start, _end in ranges
        }
        started_at = time.monotonic()
        for item, _start, _end in ranges:
            item.queue_wait_ms = queue_wait[id(item)]
            item.inference_started_at = started_at
        return queue_wait

    async def _execute_batch(
        self,
        alias: str,
        texts: list[str],
        normalize_embeddings: bool,
        queue_wait: dict[int, float],
        executor: ThreadPoolExecutor | None,
    ) -> tuple[list[list[float]], float]:
        from .models import encode_texts, get_model

        get_model(alias)
        if executor is None:
            raise SchedulerNotReadyError("Inference worker is unavailable")
        started_at = time.monotonic()
        inference_future = asyncio.get_running_loop().run_in_executor(
            executor,
            encode_texts,
            alias,
            texts,
            normalize_embeddings,
        )
        try:
            vectors = await asyncio.wait_for(
                asyncio.shield(inference_future),
                timeout=self._settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self._wait_for_inference_end(inference_future, alias)
            raise InferenceTimeoutError(
                max(queue_wait.values(), default=0.0),
                self._settings.inference_timeout_seconds * 1000,
            ) from exc
        return vectors, (time.monotonic() - started_at) * 1000

    async def _wait_for_inference_end(
        self,
        inference_future: asyncio.Future,
        alias: str,
    ) -> None:
        try:
            await asyncio.shield(inference_future)
        except Exception:
            log.error(
                "scheduler.inference_finished_after_timeout_with_error",
                model_alias=alias,
                error_code="inference_error",
            )

    def _resolve_vectors(
        self,
        alias: str,
        texts: list[str],
        ranges: list[tuple[_RequestItem, int, int]],
        queue_wait: dict[int, float],
        vectors: list[list[float]],
        inference_ms: float,
    ) -> None:
        from .models import validate_vector

        expected_dimensions = get_model_config(alias)["dimensions"]
        if len(vectors) != len(texts):
            raise ValueError("Encoder returned an unexpected number of vectors")
        for vector in vectors:
            validate_vector(vector, expected_dimensions)
        for item, start, end in ranges:
            metrics.observe_scheduler_durations(
                alias,
                queue_wait[id(item)],
                inference_ms,
            )
            if not item.future.done():
                item.future.set_result(
                    InferenceResult(
                        vectors=vectors[start:end],
                        queue_wait_ms=round(queue_wait[id(item)], 2),
                        inference_ms=round(inference_ms, 2),
                    )
                )
        metrics.inc_embeddings(len(vectors))
        log.info(
            "scheduler.batch_done",
            model_alias=alias,
            batch_size=len(texts),
            queue_wait_ms=round(max(queue_wait.values(), default=0), 2),
            inference_ms=round(inference_ms, 2),
            status="ok",
        )

    def _handle_inference_failure(
        self,
        alias: str,
        ranges: list[tuple[_RequestItem, int, int]],
        queue_wait: dict[int, float],
        error: Exception,
    ) -> None:
        from .models import ModelNotReadyError

        items = [item for item, _start, _end in ranges]
        if isinstance(error, ModelNotReadyError):
            fail_future(items, error)
            metrics.inc_model_load_failure()
            error_code = "model_not_ready"
            inference_ms = 0
        elif isinstance(error, SchedulerNotReadyError):
            fail_future(items, error)
            error_code = "worker_unavailable"
            inference_ms = 0
        elif isinstance(error, InferenceTimeoutError):
            fail_future(items, error)
            metrics.inc_inference_error()
            error_code = "inference_timeout"
            inference_ms = error.inference_ms
        else:
            fail_future(items, InferenceError("Model inference failed"))
            metrics.inc_inference_error()
            error_code = "inference_error"
            inference_ms = 0
            log.error(
                "scheduler.inference_error",
                model_alias=alias,
                batch_size=sum(len(item.texts) for item in items),
                error_code=error_code,
            )
        self._log_batch_failure(alias, ranges, queue_wait, inference_ms, error_code)

    def _log_batch_failure(
        self,
        alias: str,
        ranges: list[tuple[_RequestItem, int, int]],
        queue_wait: dict[int, float],
        inference_ms: float,
        error_code: str,
    ) -> None:
        for item, _start, _end in ranges:
            metrics.observe_scheduler_durations(
                alias,
                queue_wait.get(id(item), 0),
                inference_ms,
            )
            log.error(
                "scheduler.request_failed",
                request_id=item.request_id,
                client_label=item.client_label,
                model_alias=alias,
                batch_size=len(item.texts),
                queue_wait_ms=round(queue_wait.get(id(item), 0), 2),
                inference_ms=round(inference_ms, 2),
                total_ms=round((time.monotonic() - item.enqueued_at) * 1000, 2),
                status="error",
                error_code=error_code,
            )
