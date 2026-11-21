"""Pytest fixtures for the embedding inference service.

Tests use fakes/mocks so no real model inference is required locally. Real
model smoke tests live in tests/test_smoke.py and only run when the pinned
models are actually available (i.e. during the Hugging Face Docker build).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.models as models_module
import app.security as security_module
from app.config import settings

_TEST_RAG_KEY: Final[str] = "rag-test-key"
_TEST_ZIT_KEY: Final[str] = "zit-test-key"


class _FakeModel:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def encode(self, texts, **kwargs):  # noqa: ARG002
        import numpy as np

        return np.zeros((len(texts), self.dimensions), dtype=np.float32)


class _FakeScheduler:
    def __init__(self) -> None:
        self.started = False
        self.queues: dict = {}

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        pass

    def ready(self) -> bool:
        return self.started

    async def submit(self, alias, texts, normalize_embeddings=False, input_type=None):
        dims = 384 if alias == "minilm-384-v1" else 768
        return [[0.1] * dims for _ in texts]


@pytest.fixture(autouse=True)
def _set_test_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        security_module,
        "settings",
        replace(
            settings,
            rag_api_key=_TEST_RAG_KEY,
            zit_api_key=_TEST_ZIT_KEY,
        ),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """A TestClient with faked models, scheduler and loaded cache."""
    monkeypatch.setattr(main_module, "load_model", lambda alias: None)
    monkeypatch.setitem(
        models_module._MODEL_CACHE,
        "minilm-384-v1",
        _FakeModel(384),
    )
    monkeypatch.setitem(
        models_module._MODEL_CACHE,
        "bge-base-768-v1",
        _FakeModel(768),
    )
    fake_scheduler = _FakeScheduler()
    fake_scheduler.start()
    monkeypatch.setattr(main_module, "scheduler", fake_scheduler)

    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def rag_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TEST_RAG_KEY}"}


@pytest.fixture
def zit_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TEST_ZIT_KEY}"}
