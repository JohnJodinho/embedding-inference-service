"""Model helper regression tests."""

from types import SimpleNamespace

import app.models as models_module
from app.models import _to_vector, count_tokens, encode_texts


def test_to_vector_flattens_single_row_batch():
    assert _to_vector([[0.1, 0.2, 0.3]]) == [0.1, 0.2, 0.3]


def test_to_vector_accepts_one_dimensional_values():
    assert _to_vector([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]


def test_count_tokens_uses_tokenizer_without_truncation(monkeypatch):
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens is True
            assert truncation is False
            return [101, 42, 43, 102]

    handle = SimpleNamespace(model=[SimpleNamespace(tokenizer=FakeTokenizer())])
    monkeypatch.setattr(models_module, "get_model", lambda alias: handle)
    assert count_tokens("minilm-384-v1", "this text is not truncated") == 4


def test_encode_texts_passes_normalization_through(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return [[0.6, 0.8] for _ in texts]

    handle = SimpleNamespace(model=FakeModel())
    monkeypatch.setattr(models_module, "get_model", lambda alias: handle)
    assert encode_texts("minilm-384-v1", ["text"], True) == [[0.6, 0.8]]
