"""Bounded per-model queues and a single CPU inference worker.

The worker runs inside the FastAPI event loop (one process, one worker). It
micro-batches texts per model and applies a fairness rule so BGE receives
preferential scheduling while MiniLM can never starve.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Final

from .config import Settings, settings
from .logging import log
from .metrics import metrics

MAX_CONSECUTIVE_BGE: Final[int] = 2


class QueueFullError(RuntimeError):
    """Raised when a bounded queue cannot accept more work."""


@dataclass
class _RequestItem:
    texts: list[str]
    normalize_embeddings: bool
    input_type: str | None
    future: asyncio.Future


class InferenceScheduler:
    def __init__(self) -> None:
        self._minilm_queue: Final[asyncio.Queue[_RequestItem]] = asyncio.Queue(
            maxsize=settings.minilm_queue_capacity
        )
        self._bge_queue: Final[asyncio.Queue[_RequestItem]] = asyncio.Queue(
            maxsize=settings.bge_queue_capacity
        )
        self._queues: Final[dict[str, asyncio.Queue[_RequestItem]]] = {
            "minilm-384-v1": self._minilm_queue,
            "bge-base-768-v1": self._bge_queue,
        }
        self._worker_task: asyncio.Task | None = None
        self._consecutive_bge: int = 0
        self._started: bool = False

    @property
    def queues(self) -> dict[str, asyncio.Queue[_RequestItem]]:
        return self._queues

    def start(self) -> None:
        if self._started:
            return
        loop = asyncio.get_event_loop()
        self._worker_task = loop.create_task(self._run())
        self._started = True
        log.info("scheduler.started")

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                pass
        self._started = False
        log.info("scheduler.stopped")

    def ready(self) -> bool:
        return (
            self._started
            and self._worker_task is not None
            and not self._worker_task.done()
        )

    async def submit(
        self,
        alias: str,
        texts: list[str],
        normalize_embeddings: bool,
        input_type: str | None,
    ) -> list[list[float]]:
        """Submit texts for inference and await the result.

        Raises QueueFullError if the bounded queue for *alias* is full.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        item = _RequestItem(
            texts=texts,
            normalize_embeddings=normalize_embeddings,
            input_type=input_type,
            future=future,
        )
        queue = self._queues[alias]
        if queue.full():
            metrics.inc_queue_full()
            log.warning(
                "scheduler.queue_full",
                alias=alias,
                queue_size=queue.qsize(),
                error_code="queue_full",
            )
            raise QueueFullError(f"Queue full for model {alias}")
        await queue.put(item)
        return await future

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                raise
            if not batch:
                continue
            by_model: dict[tuple[str, bool], list[tuple[_RequestItem, int]]] = {}
            for idx, (alias, item) in enumerate(batch):
                by_model.setdefault((alias, item.normalize_embeddings), []).append((item, idx))
            for (alias, normalize_embeddings), entries in by_model.items():
                self._consecutive_bge = (
                    self._consecutive_bge + 1 if alias == "bge-base-768-v1" else 0
                )
                await self._encode_and_resolve(alias, entries, normalize_embeddings)

    async def _collect_batch(self) -> list[tuple[str, _RequestItem]]:
        """Wait for the first item(s), then opportunistically fill a micro-batch."""
        tasks = {
            asyncio.create_task(self._minilm_queue.get()): "minilm-384-v1",
            asyncio.create_task(self._bge_queue.get()): "bge-base-768-v1",
        }
        done, pending = await asyncio.wait(
            tasks.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        batch: list[tuple[str, _RequestItem]] = []
        for task in done:
            alias = tasks[task]
            try:
                item = task.result()
            except (asyncio.CancelledError, Exception):
                continue
            batch.append((alias, item))

        window_s = settings.micro_batch_window_ms / 1000.0
        deadline = time.monotonic() + window_s
        while len(batch) < settings.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            alias = self._pick_nonblocking()
            if alias is None:
                await asyncio.sleep(min(0.05, remaining))
                continue
            try:
                item = self._queues[alias].get_nowait()
            except asyncio.QueueEmpty:
                continue
            batch.append((alias, item))
        return batch

    def _pick_nonblocking(self) -> str | None:
        """Pick a model that has a queued item, applying BGE priority + fairness.

        Only queues with immediately-available items are considered so this never
        blocks.
        """
        non_empty = [a for a, q in self._queues.items() if q.qsize() > 0]
        if not non_empty:
            return None
        if len(non_empty) == 1:
            return non_empty[0]
        if (
            "bge-base-768-v1" in non_empty
            and self._consecutive_bge < MAX_CONSECUTIVE_BGE
        ):
            return "bge-base-768-v1"
        return "minilm-384-v1"

    async def _encode_and_resolve(
        self,
        alias: str,
        entries: list[tuple[_RequestItem, int]],
        normalize_embeddings: bool,
    ) -> None:
        from .config import get_model_config
        from .models import (
            ModelNotReadyError,
            encode_texts,
            get_model,
            prepare_input,
            validate_vector,
        )

        texts: list[str] = []
        ranges: list[tuple[_RequestItem, int, int]] = []
        for item, _idx in entries:
            start = len(texts)
            try:
                prepared = [
                    prepare_input(alias, item.input_type, text)
                    for text in item.texts
                ]
            except Exception as exc:  # noqa: BLE001
                if not item.future.done():
                    item.future.set_exception(exc)
                return
            texts.extend(prepared)
            ranges.append((item, start, len(texts)))

        t0 = time.monotonic()
        try:
            get_model(alias)
            vectors = encode_texts(alias, texts, normalize_embeddings)
        except ModelNotReadyError as exc:
            self._fail_future(entries, exc)
            metrics.inc_model_load_failure()
            return
        except Exception as exc:  # noqa: BLE001
            log.error(
                "scheduler.inference_error",
                alias=alias,
                batch_size=len(texts),
                error_code="inference_error",
            )
            metrics.inc_inference_error()
            self._fail_future(entries, exc)
            return
        inference_ms = (time.monotonic() - t0) * 1000

        expected_dims = get_model_config(alias)["dimensions"]
        try:
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Encoder returned {len(vectors)} vectors for {len(texts)} texts"
                )
            for item, start, end in ranges:
                item_vectors = vectors[start:end]
                for vector in item_vectors:
                    validate_vector(vector, expected_dims)
                if not item.future.done():
                    item.future.set_result(item_vectors)
            metrics.inc_embeddings(len(vectors))
            log.info(
                "scheduler.batch_done",
                alias=alias,
                batch_size=len(texts),
                inference_ms=round(inference_ms, 2),
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "scheduler.validation_error",
                alias=alias,
                error_code="validation_error",
            )
            metrics.inc_inference_error()
            self._fail_future(entries, exc)

    def _fail_future(
        self,
        entries: list[tuple[_RequestItem, int]],
        exc: BaseException,
    ) -> None:
        for item, _idx in entries:
            if not item.future.done():
                item.future.set_exception(exc)


scheduler: Final[InferenceScheduler] = InferenceScheduler()

