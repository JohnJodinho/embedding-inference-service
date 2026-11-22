"""Service configuration and immutable model registry.

All model revisions are pinned to immutable Hugging Face commit hashes resolved
from the official model repositories. These values must not drift to `main`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Immutable model registry
# ---------------------------------------------------------------------------

MINILM_MODEL_ID: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION: Final[str] = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_DIMENSIONS: Final[int] = 384

BGE_MODEL_ID: Final[str] = "BAAI/bge-base-en-v1.5"
BGE_REVISION: Final[str] = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
BGE_DIMENSIONS: Final[int] = 768

# Exact BGE retrieval query instruction (trailing space significant).
BGE_QUERY_INSTRUCTION: Final[str] = "Represent this sentence for searching relevant passages: "

MAX_SEQ_LENGTH: Final[int] = 512

MODEL_REGISTRY: Final[dict[str, dict]] = {
    "minilm-384-v1": {
        "alias": "minilm-384-v1",
        "model_id": MINILM_MODEL_ID,
        "revision": MINILM_REVISION,
        "dimensions": MINILM_DIMENSIONS,
        "license": "apache-2.0",
        "supports_input_type": ("text", None),
        "requires_input_type": False,
        "adds_query_instruction": False,
    },
    "bge-base-768-v1": {
        "alias": "bge-base-768-v1",
        "model_id": BGE_MODEL_ID,
        "revision": BGE_REVISION,
        "dimensions": BGE_DIMENSIONS,
        "license": "mit",
        "supports_input_type": ("query", "document"),
        "requires_input_type": True,
        "adds_query_instruction": True,
    },
}


def get_model_config(alias: str) -> dict:
    """Return model config for *alias* or raise KeyError if unknown."""
    if alias not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model alias: {alias}")
    return MODEL_REGISTRY[alias]


# ---------------------------------------------------------------------------
# Runtime settings (overridable via environment)
# ---------------------------------------------------------------------------

import os  # noqa: E402  (imported after constants for readability)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    rag_api_key: str = _env_str("RAG_API_KEY", "")
    zit_api_key: str = _env_str("ZIT_API_KEY", "")
    minilm_queue_capacity: int = _env_int("MINILM_QUEUE_CAPACITY", 32)
    bge_queue_capacity: int = _env_int("BGE_QUEUE_CAPACITY", 32)
    max_batch_size: int = _env_int("MAX_BATCH_SIZE", 8)
    micro_batch_window_ms: int = _env_int("MICRO_BATCH_WINDOW_MS", 200)
    rag_rate_limit_per_min: int = _env_int("RAG_RATE_LIMIT_PER_MIN", 60)
    zit_rate_limit_per_min: int = _env_int("ZIT_RATE_LIMIT_PER_MIN", 60)
    request_timeout_seconds: int = _env_int("REQUEST_TIMEOUT_SECONDS", 90)
    inference_timeout_seconds: int = _env_int("INFERENCE_TIMEOUT_SECONDS", 60)
    torch_num_threads: int = _env_int("TORCH_NUM_THREADS", 2)
    torch_num_interop_threads: int = _env_int("TORCH_NUM_INTEROP_THREADS", 1)
    max_request_bytes: int = _env_int("MAX_REQUEST_BYTES", 262_144)  # 256 KiB
    max_texts_per_request: int = _env_int("MAX_TEXTS_PER_REQUEST", 8)
    max_bge_texts_per_request: int = _env_int("MAX_BGE_TEXTS_PER_REQUEST", 4)
    max_chars_per_text: int = _env_int("MAX_CHARS_PER_TEXT", 2048)
    max_tokens_per_text: int = _env_int("MAX_TOKENS_PER_TEXT", 512)

    def __post_init__(self) -> None:
        positive_values = (
            "minilm_queue_capacity",
            "bge_queue_capacity",
            "max_batch_size",
            "request_timeout_seconds",
            "inference_timeout_seconds",
            "torch_num_threads",
            "torch_num_interop_threads",
            "max_request_bytes",
            "max_texts_per_request",
            "max_bge_texts_per_request",
            "max_chars_per_text",
            "max_tokens_per_text",
            "rag_rate_limit_per_min",
            "zit_rate_limit_per_min",
        )
        for field_name in positive_values:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.micro_batch_window_ms < 0:
            raise ValueError("micro_batch_window_ms cannot be negative")
        if self.max_tokens_per_text > MAX_SEQ_LENGTH:
            raise ValueError(
                f"max_tokens_per_text cannot exceed the model limit of {MAX_SEQ_LENGTH}"
            )
        if self.max_texts_per_request > self.max_batch_size:
            raise ValueError("max_texts_per_request cannot exceed max_batch_size")
        if self.max_bge_texts_per_request > self.max_texts_per_request:
            raise ValueError(
                "max_bge_texts_per_request cannot exceed max_texts_per_request"
            )


settings: Final[Settings] = Settings()
