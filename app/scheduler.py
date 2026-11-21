"""Bounded per-model queues and one off-event-loop CPU inference worker."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from .config import Settings, settings
from .logging import log
from .metrics import metrics
from .scheduler_inference import InferenceBatchProcessor
from .scheduler_types import (
    InferenceError,
    InferenceResult,
    InferenceTimeoutError,
    QueueFullError,
    RequestTimeoutError,
    SchedulerNotReadyError,
    _RequestItem,
    create_request_item,
    fail_future,
)

MAX_CONSECUTIVE_BGE: Final[int] = 2

__all__ = [
    "InferenceError",
    "InferenceResult",
    "InferenceTimeoutError",
    "InferenceScheduler",
    "MAX_CONSECUTIVE_BGE",
    "QueueFullError",
    "RequestTimeoutError",
    "SchedulerNotReadyError",
    "scheduler",
]


class InferenceScheduler:
    def __init__(self, config: Settings = settings) -> None:
        self._settings = config
        self._processor = InferenceBatchProcessor(config)
        self._minilm_queue: asyncio.Queue[_RequestItem] = asyncio.Queue(
            maxsize=config.minilm_queue_capacity
        )
        self._bge_queue: asyncio.Queue[_RequestItem] = asyncio.Queue(
            maxsize=config.bge_queue_capacity
        )
        self._queues: Final[dict[str, asyncio.Queue[_RequestItem]]] = {
            "minilm-384-v1": self._minilm_queue,
            "bge-base-768-v1": self._bge_queue,
        }
        self._worker_task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._wakeup = asyncio.Event()
        self._consecutive_bge = 0
        self._started = False
        self._stopping = False

    @property
    def queues(self) -> dict[str, asyncio.Queue[_RequestItem]]:
        return self._queues

    def start(self) -> None:
        if self._started:
            return
        self._wakeup = asyncio.Event()
        self._stopping = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="embedding-inference",
        )
        self._worker_task = asyncio.get_running_loop().create_task(self._run())
        self._started = True
        log.info("scheduler.started")

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        self._wakeup.set()
        if self._worker_task is not None:
            await self._worker_task
        self._reject_queued(SchedulerNotReadyError("Inference scheduler stopped"))
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._started = False
        log.info("scheduler.stopped")

    def ready(self) -> bool:
        return (
            self._started
            and not self._stopping
            and self._worker_task is not None
            and not self._worker_task.done()
        )

    async def submit(
        self,
        alias: str,
        texts: list[str],
        normalize_embeddings: bool,
        input_type: str | None,
        request_id: str = "unknown",
        client_label: str = "unknown",
    ) -> InferenceResult:
        """Enqueue a request and bound its total wait time."""
        if not self.ready():
            raise SchedulerNotReadyError("Inference scheduler is not ready")
        future = asyncio.get_running_loop().create_future()
        item = create_request_item(
            alias,
            texts,
            normalize_embeddings,
            input_type,
            request_id,
            client_label,
            future,
        )
        queue = self._queues[alias]
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            metrics.inc_queue_full()
            log.warning(
                "scheduler.queue_full",
                request_id=request_id,
                client_label=client_label,
                model_alias=alias,
                batch_size=len(texts),
                queue_wait_ms=0,
                inference_ms=0,
                total_ms=0,
                status="error",
                error_code="queue_full",
                queue_size=queue.qsize(),
            )
            raise QueueFullError(f"Queue full for model {alias}") from exc
        self._wakeup.set()
        return await self._wait_for_result(item)

    async def _wait_for_result(self, item: _RequestItem) -> InferenceResult:
        try:
            return await asyncio.wait_for(
                item.future,
                timeout=self._settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if item.future.done() and not item.future.cancelled():
                raise
            if not item.future.done():
                item.future.cancel()
            now = time.monotonic()
            queue_wait_ms = (
                item.queue_wait_ms
                if item.queue_wait_ms is not None
                else (now - item.enqueued_at) * 1000
            )
            inference_ms = (
                (now - item.inference_started_at) * 1000
                if item.inference_started_at is not None
                else 0
            )
            log.warning(
                "scheduler.request_timeout",
                request_id=item.request_id,
                client_label=item.client_label,
                model_alias=item.alias,
                batch_size=len(item.texts),
                queue_wait_ms=round(queue_wait_ms, 2),
                inference_ms=round(inference_ms, 2),
                total_ms=round((now - item.enqueued_at) * 1000, 2),
                status="error",
                error_code="request_timeout",
            )
            raise RequestTimeoutError(queue_wait_ms, inference_ms) from exc

    async def _run(self) -> None:
        while not self._stopping:
            batch = await self._collect_batch()
            if self._stopping:
                break
            grouped: dict[tuple[str, bool], list[_RequestItem]] = {}
            for alias, item in batch:
                grouped.setdefault((alias, item.normalize_embeddings), []).append(item)
            for (alias, normalize_embeddings), entries in grouped.items():
                await self._encode_and_resolve(alias, entries, normalize_embeddings)
                if self._stopping:
                    break
        self._reject_queued(SchedulerNotReadyError("Inference scheduler stopped"))

    async def _collect_batch(self) -> list[tuple[str, _RequestItem]]:
        if not await self._wait_for_work():
            return []
        collected: list[tuple[str, _RequestItem]] = []
        text_count = 0
        deadline = time.monotonic() + self._settings.micro_batch_window_ms / 1000
        while not self._stopping:
            if collected and text_count >= self._settings.max_batch_size:
                break
            alias = self._pick_nonblocking()
            if alias is None:
                if await self._wait_for_microbatch(deadline):
                    continue
                break
            item = self._queues[alias].get_nowait()
            if item.future.cancelled():
                continue
            self._consecutive_bge = (
                self._consecutive_bge + 1
                if alias == "bge-base-768-v1"
                else 0
            )
            collected.append((alias, item))
            text_count += len(item.texts)
        return collected

    async def _wait_for_work(self) -> bool:
        while not self._stopping:
            if any(queue.qsize() for queue in self._queues.values()):
                return True
            self._wakeup.clear()
            if any(queue.qsize() for queue in self._queues.values()):
                return True
            await self._wakeup.wait()
        return False

    async def _wait_for_microbatch(self, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self._wakeup.clear()
        if any(queue.qsize() for queue in self._queues.values()):
            return True
        try:
            await asyncio.wait_for(self._wakeup.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            return False
        return not self._stopping

    def _pick_nonblocking(self) -> str | None:
        """Prefer BGE but serve MiniLM after a bounded run of BGE requests."""
        minilm_ready = not self._minilm_queue.empty()
        bge_ready = not self._bge_queue.empty()
        if bge_ready and (not minilm_ready or self._consecutive_bge < MAX_CONSECUTIVE_BGE):
            return "bge-base-768-v1"
        if minilm_ready:
            return "minilm-384-v1"
        return None

    async def _encode_and_resolve(
        self,
        alias: str,
        entries: list[_RequestItem],
        normalize_embeddings: bool,
    ) -> None:
        await self._processor.run(
            alias,
            entries,
            normalize_embeddings,
            self._executor,
        )

    def _reject_queued(self, error: BaseException) -> None:
        queued: list[_RequestItem] = []
        for queue in self._queues.values():
            while not queue.empty():
                queued.append(queue.get_nowait())
        fail_future(queued, error)


scheduler: Final[InferenceScheduler] = InferenceScheduler()
