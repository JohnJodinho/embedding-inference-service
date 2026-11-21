"""Scheduler tests: bounded queues, fairness, timeouts and off-loop inference."""

import asyncio
import threading
import time
from dataclasses import replace

import pytest

import app.models as models_module
from app.config import settings
from app.scheduler import (
    MAX_CONSECUTIVE_BGE,
    InferenceScheduler,
    InferenceTimeoutError,
    QueueFullError,
    RequestTimeoutError,
    _RequestItem,
)


@pytest.fixture
def empty_scheduler():
    return InferenceScheduler()


def _item(loop, texts=None, alias="minilm-384-v1"):
    return _RequestItem(
        alias=alias,
        texts=texts or ["x"],
        normalize_embeddings=False,
        input_type=None,
        request_id="test-request",
        client_label="rag",
        enqueued_at=time.monotonic(),
        future=loop.create_future(),
    )


def test_queues_are_bounded(empty_scheduler):
    assert empty_scheduler.queues["minilm-384-v1"].maxsize > 0
    assert empty_scheduler.queues["bge-base-768-v1"].maxsize > 0


def test_submit_raises_when_queue_full(empty_scheduler):
    loop = asyncio.get_event_loop()

    async def run():
        empty_scheduler.start()
        queue = empty_scheduler.queues["minilm-384-v1"]
        for _ in range(queue.maxsize):
            queue.put_nowait(_item(loop))
        with pytest.raises(QueueFullError):
            await empty_scheduler.submit(
                alias="minilm-384-v1",
                texts=["x"],
                normalize_embeddings=False,
                input_type=None,
            )
        while not queue.empty():
            queue.get_nowait().future.cancel()
        await empty_scheduler.stop()

    loop.run_until_complete(run())


def test_pick_nonblocking_prefers_bge_but_serves_minilm(empty_scheduler):
    minilm_queue = empty_scheduler.queues["minilm-384-v1"]
    bge_queue = empty_scheduler.queues["bge-base-768-v1"]
    loop = asyncio.get_event_loop()

    minilm_queue.put_nowait(_item(loop))
    bge_queue.put_nowait(_item(loop))
    assert empty_scheduler._pick_nonblocking() == "bge-base-768-v1"
    empty_scheduler._consecutive_bge = MAX_CONSECUTIVE_BGE
    assert empty_scheduler._pick_nonblocking() == "minilm-384-v1"


def test_encode_resolves_all_texts_and_preserves_vector_order(
    empty_scheduler,
    monkeypatch,
):
    loop = asyncio.get_event_loop()
    item = _item(loop, ["first", "second"])
    monkeypatch.setattr(models_module, "get_model", lambda alias: object())
    monkeypatch.setattr(
        models_module,
        "encode_texts",
        lambda alias, texts, normalize_embeddings: [
            [float(index)] * 384 for index, _text in enumerate(texts)
        ],
    )

    async def run():
        empty_scheduler.start()
        await empty_scheduler._encode_and_resolve(
            "minilm-384-v1",
            [item],
            False,
        )
        result = item.future.result()
        await empty_scheduler.stop()
        return result

    result = loop.run_until_complete(run())
    assert len(result.vectors) == 2
    assert result.vectors[0] == [0.0] * 384
    assert result.vectors[1] == [1.0] * 384


def test_inference_runs_off_the_event_loop(empty_scheduler, monkeypatch):
    loop = asyncio.get_event_loop()
    caller_thread = threading.get_ident()
    inference_threads = []
    monkeypatch.setattr(models_module, "get_model", lambda alias: object())

    def encode(alias, texts, normalize_embeddings):
        inference_threads.append(threading.get_ident())
        return [[0.0] * 384 for _ in texts]

    monkeypatch.setattr(models_module, "encode_texts", encode)

    async def run():
        empty_scheduler.start()
        result = await empty_scheduler.submit(
            "minilm-384-v1",
            ["text"],
            False,
            None,
            request_id="async-test",
            client_label="rag",
        )
        await empty_scheduler.stop()
        return result

    result = loop.run_until_complete(run())
    assert result.vectors == [[0.0] * 384]
    assert inference_threads and inference_threads[0] != caller_thread


def test_queued_requests_obey_total_timeout(monkeypatch):
    config = replace(
        settings,
        request_timeout_seconds=0.01,
        micro_batch_window_ms=0,
    )
    scheduler = InferenceScheduler(config)
    monkeypatch.setattr(models_module, "get_model", lambda alias: object())

    def slow_encode(alias, texts, normalize_embeddings):
        time.sleep(0.05)
        return [[0.0] * 384 for _ in texts]

    monkeypatch.setattr(models_module, "encode_texts", slow_encode)

    async def run():
        scheduler.start()
        first = asyncio.create_task(
            scheduler.submit("minilm-384-v1", ["first"], False, None)
        )
        await asyncio.sleep(0.005)
        with pytest.raises(RequestTimeoutError):
            await scheduler.submit("minilm-384-v1", ["second"], False, None)
        with pytest.raises(RequestTimeoutError):
            await first
        await scheduler.stop()

    asyncio.get_event_loop().run_until_complete(run())


def test_inference_timeout_reaches_caller(monkeypatch):
    config = replace(
        settings,
        request_timeout_seconds=1,
        inference_timeout_seconds=0.01,
        micro_batch_window_ms=0,
    )
    scheduler = InferenceScheduler(config)
    monkeypatch.setattr(models_module, "get_model", lambda alias: object())

    def slow_encode(alias, texts, normalize_embeddings):
        time.sleep(0.03)
        return [[0.0] * 384 for _ in texts]

    monkeypatch.setattr(models_module, "encode_texts", slow_encode)

    async def run():
        scheduler.start()
        with pytest.raises(InferenceTimeoutError):
            await scheduler.submit("minilm-384-v1", ["text"], False, None)
        await scheduler.stop()

    asyncio.get_event_loop().run_until_complete(run())
