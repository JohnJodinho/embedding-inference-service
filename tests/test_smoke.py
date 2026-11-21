"""Real-model smoke tests.

These use the pinned, downloaded sentence-transformers models. They are
required to run during the Hugging Face Docker build after the model cache is
populated. A missing model or failed inference fails the build.
"""

from __future__ import annotations

import math

from app.models import (
    MINILM_DIMENSIONS,
    MINILM_REVISION,
    BGE_DIMENSIONS,
    BGE_QUERY_INSTRUCTION,
    BGE_REVISION,
    encode_texts,
    count_tokens,
    get_model,
    load_model,
    prepare_input,
    validate_vector,
)


def test_minilm_smoke_dimensions():
    load_model("minilm-384-v1")
    assert get_model("minilm-384-v1").dimensions == MINILM_DIMENSIONS
    assert count_tokens("minilm-384-v1", "hello world") > 0
    vectors = encode_texts("minilm-384-v1", ["hello world"], normalize_embeddings=True)
    assert len(vectors) == 1
    assert len(vectors[0]) == MINILM_DIMENSIONS
    validate_vector(vectors[0], MINILM_DIMENSIONS)
    assert all(math.isfinite(x) for x in vectors[0])
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0, rel_tol=1e-4)


def test_minilm_preserves_input_and_order():
    load_model("minilm-384-v1")
    vectors = encode_texts("minilm-384-v1", ["a", "b", "c"], normalize_embeddings=False)
    assert len(vectors) == 3
    for v in vectors:
        validate_vector(v, MINILM_DIMENSIONS)


def test_bge_smoke_dimensions():
    load_model("bge-base-768-v1")
    assert get_model("bge-base-768-v1").dimensions == BGE_DIMENSIONS
    assert count_tokens("bge-base-768-v1", "hello world") > 0
    vectors = encode_texts("bge-base-768-v1", ["hello world"], normalize_embeddings=True)
    assert len(vectors) == 1
    assert len(vectors[0]) == BGE_DIMENSIONS
    validate_vector(vectors[0], BGE_DIMENSIONS)
    assert math.isclose(math.sqrt(sum(value * value for value in vectors[0])), 1.0, rel_tol=1e-4)


def test_bge_query_prefix_applied():
    query = prepare_input("bge-base-768-v1", "query", "find courses")
    document = prepare_input("bge-base-768-v1", "document", "find courses")
    assert query.startswith(BGE_QUERY_INSTRUCTION)
    assert document == "find courses"
    assert query != document


def test_bge_revision_is_immutable():
    assert BGE_REVISION == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def test_minilm_revision_is_immutable():
    assert MINILM_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
