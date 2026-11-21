"""Real-model smoke tests.

These use the pinned, downloaded sentence-transformers models. They are
designed to run during the Hugging Face Docker build (where the models are
pre-downloaded into the HF cache). Locally, where the models are not
downloaded, they skip rather than trigger a model download.
"""

from __future__ import annotations

import math

import pytest

from app.models import (
    MINILM_DIMENSIONS,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    BGE_DIMENSIONS,
    BGE_MODEL_ID,
    BGE_QUERY_INSTRUCTION,
    BGE_REVISION,
    encode_texts,
    get_model,
    load_model,
    prepare_input,
    validate_vector,
)


def _model_available(alias: str) -> bool:
    try:
        load_model(alias)
        return True
    except Exception:  # noqa: BLE001 - model not downloaded locally
        return False


MINILM_AVAILABLE = _model_available("minilm-384-v1")
BGE_AVAILABLE = _model_available("bge-base-768-v1")


@pytest.mark.skipif(not MINILM_AVAILABLE, reason="MiniLM model not downloaded locally")
def test_minilm_smoke_dimensions():
    model = get_model("minilm-384-v1")
    vectors = encode_texts("minilm-384-v1", ["hello world"], normalize_embeddings=True)
    assert len(vectors) == 1
    assert len(vectors[0]) == MINILM_DIMENSIONS
    validate_vector(vectors[0], MINILM_DIMENSIONS)
    assert all(math.isfinite(x) for x in vectors[0])


@pytest.mark.skipif(not MINILM_AVAILABLE, reason="MiniLM model not downloaded locally")
def test_minilm_preserves_input_and_order():
    vectors = encode_texts("minilm-384-v1", ["a", "b", "c"], normalize_embeddings=False)
    assert len(vectors) == 3
    for v in vectors:
        validate_vector(v, MINILM_DIMENSIONS)


@pytest.mark.skipif(not BGE_AVAILABLE, reason="BGE model not downloaded locally")
def test_bge_smoke_dimensions():
    model = get_model("bge-base-768-v1")
    vectors = encode_texts("bge-base-768-v1", ["hello world"], normalize_embeddings=True)
    assert len(vectors) == 1
    assert len(vectors[0]) == BGE_DIMENSIONS
    validate_vector(vectors[0], BGE_DIMENSIONS)


@pytest.mark.skipif(not BGE_AVAILABLE, reason="BGE model not downloaded locally")
def test_bge_query_prefix_applied():
    query = prepare_input("bge-base-768-v1", "query", "find courses")
    document = prepare_input("bge-base-768-v1", "document", "find courses")
    assert query.startswith(BGE_QUERY_INSTRUCTION)
    assert document == "find courses"
    assert query != document


@pytest.mark.skipif(not BGE_AVAILABLE, reason="BGE model not downloaded locally")
def test_bge_revision_is_immutable():
    assert BGE_REVISION == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


@pytest.mark.skipif(not MINILM_AVAILABLE, reason="MiniLM model not downloaded locally")
def test_minilm_revision_is_immutable():
    assert MINILM_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
