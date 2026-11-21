"""Model registry and configuration tests."""

from app.config import (
    BGE_DIMENSIONS,
    BGE_MODEL_ID,
    BGE_QUERY_INSTRUCTION,
    BGE_REVISION,
    MINILM_DIMENSIONS,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    MODEL_REGISTRY,
    get_model_config,
)


def test_model_registry_contains_both_aliases():
    assert set(MODEL_REGISTRY) == {"minilm-384-v1", "bge-base-768-v1"}


def test_minilm_registry_values():
    assert MINILM_MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert MINILM_DIMENSIONS == 384
    assert len(MINILM_REVISION) == 40  # full SHA-1


def test_bge_registry_values():
    assert BGE_MODEL_ID == "BAAI/bge-base-en-v1.5"
    assert BGE_DIMENSIONS == 768
    assert len(BGE_REVISION) == 40
    assert BGE_QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "


def test_get_model_config_unknown_alias():
    try:
        get_model_config("does-not-exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown alias")


def test_get_model_config_returns_dimensions():
    assert get_model_config("minilm-384-v1")["dimensions"] == 384
    assert get_model_config("bge-base-768-v1")["dimensions"] == 768
