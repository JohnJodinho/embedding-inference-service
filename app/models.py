"""Model loading, validation and encoding.

Each model is loaded exactly once (in FastAPI lifespan) and kept in memory for
the life of the process. Encoding runs on CPU only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import torch
from sentence_transformers import SentenceTransformer

from .config import (
    BGE_DIMENSIONS,
    BGE_MODEL_ID,
    BGE_QUERY_INSTRUCTION,
    BGE_REVISION,
    MAX_SEQ_LENGTH,
    MINILM_DIMENSIONS,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    MODEL_REGISTRY,
    get_model_config,
)
from .logging import log

# CPU-only inference. Limit PyTorch thread contention.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)

_MODEL_CACHE: Final[dict[str, SentenceTransformer]] = {}


@dataclass
class ModelHandle:
    alias: str
    model: SentenceTransformer
    dimensions: int


class ModelNotReadyError(RuntimeError):
    """Raised when a model has not finished loading."""


def load_model(alias: str) -> ModelHandle:
    """Load *alias* once, verify its output dimension, and cache the handle."""
    cfg = get_model_config(alias)
    model_id: str = cfg["model_id"]
    revision: str = cfg["revision"]
    expected_dims: int = cfg["dimensions"]

    if alias in _MODEL_CACHE:
        return _MODEL_CACHE[alias]

    log.info("model.loading", alias=alias, model_id=model_id, revision=revision)
    try:
        model = SentenceTransformer(model_id, revision=revision, device="cpu")
        model.max_seq_length = MAX_SEQ_LENGTH

        # Probe the model to confirm it produces the expected dimension.
        probe = model.encode(["dimension check"], normalize_embeddings=False)
        probe_vector = _to_vector(probe)
        actual_dims = len(probe_vector)
        if actual_dims != expected_dims:
            raise ValueError(
                f"Model {alias} produced dimension {actual_dims}; expected {expected_dims}"
            )

        handle = ModelHandle(alias=alias, model=model, dimensions=actual_dims)
        _MODEL_CACHE[alias] = handle
        log.info(
            "model.loaded",
            alias=alias,
            model_id=model_id,
            revision=revision,
            dimensions=actual_dims,
        )
        return handle
    except Exception as exc:  # noqa: BLE001 - startup must report any load failure
        log.error("model.load_failed", alias=alias, error_code="model_load_failed")
        raise ModelNotReadyError(f"Failed to load model {alias}") from exc


def get_model(alias: str) -> ModelHandle:
    """Return a loaded model handle, or raise if not yet loaded."""
    if alias not in _MODEL_CACHE:
        raise ModelNotReadyError(f"Model {alias} is not loaded")
    return _MODEL_CACHE[alias]


def _to_vector(value) -> list[float]:
    """Normalise a tensor/ndarray result into a list of floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        value = [value]
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [float(x) for x in value]


def prepare_input(alias: str, input_type: str | None, text: str) -> str:
    """Apply model-specific input formatting without altering MiniLM text."""
    cfg = get_model_config(alias)
    if cfg["adds_query_instruction"]:
        if input_type not in ("query", "document"):
            raise ValueError(f"Model {alias} requires input_type 'query' or 'document'")
        if input_type == "query":
            return BGE_QUERY_INSTRUCTION + text
        return text
    # MiniLM: input passed unchanged. input_type may be omitted or 'text'.
    if input_type not in (None, "text"):
        raise ValueError(f"Model {alias} does not support input_type '{input_type}'")
    return text


def encode_texts(
    alias: str,
    texts: list[str],
    normalize_embeddings: bool,
) -> list[list[float]]:
    """Encode *texts* for *alias*, preserving order.

    BGE query documents are prepended with the retrieval instruction inside
    prepare_input; callers should pass already-prepared texts.
    """
    handle = get_model(alias)
    vectors = handle.model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [list(map(float, v)) for v in vectors]


def validate_vector(vector: list[float], expected_dimensions: int) -> None:
    """Raise if *vector* is the wrong length or contains non-finite values."""
    import math

    if len(vector) != expected_dimensions:
        raise ValueError(
            f"Embedding dimension {len(vector)} != expected {expected_dimensions}"
        )
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains non-finite value")


def download_models() -> None:
    """Download both pinned models into the HF cache (used at build time)."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MINILM_MODEL_ID, revision=MINILM_REVISION, device="cpu")
    SentenceTransformer(BGE_MODEL_ID, revision=BGE_REVISION, device="cpu")
    log.info("models.downloaded", minilm=MINILM_MODEL_ID, bge=BGE_MODEL_ID)
