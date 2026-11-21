"""Typed HTTP response contracts used by OpenAPI and runtime validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from .config import settings


class ErrorResponse(BaseModel):
    error: str


class DetailErrorResponse(BaseModel):
    detail: str


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr
    input: StrictStr | list[StrictStr]
    input_type: StrictStr | None = None
    normalize_embeddings: StrictBool = False

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class ModelInfo(BaseModel):
    alias: str
    model_id: str
    revision: str
    dimensions: int
    license: str


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: list[ModelInfo]


class EmbeddingItem(BaseModel):
    object: Literal["embedding"]
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    input_texts: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"]
    model: str
    dimensions: int
    data: list[EmbeddingItem]
    usage: EmbeddingUsage


EMBEDDING_REQUEST_SCHEMA: dict[str, object] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["model", "input"],
            "properties": {
                "model": {"type": "string", "enum": ["minilm-384-v1"]},
                "input": {
                    "anyOf": [
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": settings.max_chars_per_text,
                        },
                        {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": settings.max_texts_per_request,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": settings.max_chars_per_text,
                            },
                        },
                    ]
                },
                "input_type": {
                    "anyOf": [
                        {"type": "string", "enum": ["text"]},
                        {"type": "null"},
                    ],
                    "default": None,
                },
                "normalize_embeddings": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": ["model", "input", "input_type"],
            "properties": {
                "model": {"type": "string", "enum": ["bge-base-768-v1"]},
                "input": {
                    "anyOf": [
                        {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": settings.max_chars_per_text,
                        },
                        {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": settings.max_texts_per_request,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": settings.max_chars_per_text,
                            },
                        },
                    ]
                },
                "input_type": {"type": "string", "enum": ["query", "document"]},
                "normalize_embeddings": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    ]
}
