"""Scheduler tests: bounded queues, fairness, overload rejection."""

import asyncio

import pytest

from app.scheduler import (
    MAX_CONSECUTIVE_BGE,
    QueueFullError,
    InferenceScheduler,
    scheduler,
)


@pytest.fixture
def empty_scheduler():
    return InferenceScheduler()


def test_queues_are_bounded(empty_scheduler):
    assert empty_scheduler.queues["minilm-384-v1"].maxsize > 0
    assert empty_scheduler.queues["bge-base-768-v1"].maxsize > 0


def test_submit_raises_when_queue_full(empty_scheduler):
    alias = "minilm-384-v1"
    queue = empty_scheduler.queues[alias]
    # Fill the queue to capacity.
    for _ in range(queue.maxsize):
        loop = asyncio.get_event_loop()
        queue.put_nowait(
            type("_", (), {"texts": ["x"], "normalize_embeddings": False, "input_type": None, "future": loop.create_future()})()
        )
    assert queue.full()

    async def _submit():
        return await empty_scheduler.submit(
            alias=alias,
            texts=["x"],
            normalize_embeddings=False,
            input_type=None,
        )

    with pytest.raises(QueueFullError):
        asyncio.get_event_loop().run_until_complete(_submit())


def test_pick_nonblocking_prefers_bge_but_serves_minilm(empty_scheduler):
    sched = empty_scheduler
    minilm_q = sched.queues["minilm-384-v1"]
    bge_q = sched.queues["bge-base-768-v1"]
    loop = asyncio.get_event_loop()

    def make_item():
        return type(
            "_",
            (),
            {
                "texts": ["x"],
                "normalize_embeddings": False,
                "input_type": None,
                "future": loop.create_future(),
            },
        )()

    # Both queues have items -> BGE preferred while consecutive count < limit.
    minilm_q.put_nowait(make_item())
    bge_q.put_nowait(make_item())
    assert sched._pick_nonblocking() == "bge-base-768-v1"
    sched._consecutive_bge = MAX_CONSECUTIVE_BGE
    # Force fairness: after consecutive BGE cap, MiniLM is picked.
    bge_q.put_nowait(make_item())
    minilm_q.put_nowait(make_item())
    assert sched._pick_nonblocking() == "minilm-384-v1"


def test_pick_nonblocking_only_one_non_empty(empty_scheduler):
    sched = empty_scheduler
    loop = asyncio.get_event_loop()
    minilm_q = sched.queues["minilm-384-v1"]
    minilm_q.put_nowait(
        type(
            "_",
            (),
            {"texts": ["x"], "normalize_embeddings": False, "input_type": None, "future": loop.create_future()},
        )()
    )
    assert sched._pick_nonblocking() == "minilm-384-v1"


def test_encode_resolves_all_texts_for_one_request(empty_scheduler, monkeypatch):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    item = type(
        "_",
        (),
        {
            "texts": ["first", "second"],
            "normalize_embeddings": False,
            "input_type": None,
            "future": future,
        },
    )()

    import app.models as models_module

    monkeypatch.setattr(models_module, "get_model", lambda alias: object())
    monkeypatch.setattr(
        models_module,
        "encode_texts",
        lambda alias, texts, normalize_embeddings: [
            [float(index)] * 384
            for index, _text in enumerate(texts)
        ],
    )

    async def run():
        await empty_scheduler._encode_and_resolve(
            "minilm-384-v1",
            [(item, 0)],
            False,
        )

    loop.run_until_complete(run())
    assert future.done()
    result = future.result()
    assert len(result) == 2
    assert len(result[0]) == 384
