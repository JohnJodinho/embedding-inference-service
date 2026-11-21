"""Structured logging for the embedding inference service.

Logs are JSON lines. Secrets, raw input text and embeddings are never logged.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Final


class StructuredLogger:
    """A small structured logger emitting JSON lines to stdout."""

    def __init__(self, name: str = "embedding-inference") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(event, extra=fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(event, extra=fields)

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(event, extra=fields)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ):
                continue
            if key == "extra":
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


log: Final = StructuredLogger()  # noqa: F821
