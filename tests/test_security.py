"""Authentication, rate limiting and input validation tests."""

from dataclasses import replace

import app.security as security_module
from app.security import (
    authenticate,
    constant_time_compare,
    rate_limiter,
    validate_request_size,
    validate_texts,
)


def test_constant_time_compare_same_and_different():
    assert constant_time_compare("secret", "secret") is True
    assert constant_time_compare("secret", "different") is False


def test_authenticate_rag_key(rag_headers):
    class FakeRequest:
        headers = {"authorization": rag_headers["Authorization"]}

    assert authenticate(FakeRequest(), None) == "rag"


def test_authenticate_zit_key(zit_headers):
    class FakeRequest:
        headers = {"authorization": zit_headers["Authorization"]}

    assert authenticate(FakeRequest(), None) == "zit"


def test_bearer_scheme_is_case_insensitive(rag_headers):
    class FakeRequest:
        headers = {"authorization": rag_headers["Authorization"].replace("Bearer", "bearer")}

    assert authenticate(FakeRequest(), None) == "rag"


def test_authenticate_missing_key_raises():
    from fastapi import HTTPException
    from app.metrics import metrics

    class FakeRequest:
        headers = {}

    before = metrics.snapshot()["auth_failures"]
    try:
        authenticate(FakeRequest(), None)
    except HTTPException as exc:
        assert exc.status_code == 401
        assert metrics.snapshot()["auth_failures"] == before + 1
        return
    raise AssertionError("expected 401 for missing key")


def test_authenticate_unknown_key_raises():
    from fastapi import HTTPException

    class FakeRequest:
        headers = {"authorization": "Bearer unknown-key"}

    try:
        authenticate(FakeRequest(), None)
    except HTTPException as exc:
        assert exc.status_code == 401
        return
    raise AssertionError("expected 401 for unknown key")


def test_authenticate_rejects_empty_configured_keys(monkeypatch):
    monkeypatch.setattr(
        security_module,
        "settings",
        replace(
            security_module.settings,
            rag_api_key="",
            zit_api_key="",
        ),
    )

    class FakeRequest:
        headers = {"authorization": "Bearer "}

    from fastapi import HTTPException

    try:
        authenticate(FakeRequest(), None)
    except HTTPException as exc:
        assert exc.status_code == 401
        return
    raise AssertionError("expected 401 when API keys are not configured")


def test_rate_limiter_allows_and_blocks():
    """Token bucket allows up to capacity then blocks."""
    from app.security import _TokenBucket

    bucket = _TokenBucket(capacity=2, fill_rate=0.0)
    assert bucket.consume(1) is True
    assert bucket.consume(1) is True
    assert bucket.consume(1) is False


def test_rate_limiter_recharges():
    from app.security import _TokenBucket
    import time

    bucket = _TokenBucket(capacity=1, fill_rate=10.0)
    assert bucket.consume(1) is True
    assert bucket.consume(1) is False
    time.sleep(0.15)
    assert bucket.consume(1) is True


def test_validate_texts_empty():
    from fastapi import HTTPException

    try:
        validate_texts([])
    except HTTPException as exc:
        assert exc.status_code == 400
        return
    raise AssertionError("expected 400 for empty input")


def test_validate_texts_too_many(monkeypatch):
    monkeypatch.setattr(
        security_module,
        "settings",
        replace(security_module.settings, max_texts_per_request=2),
    )
    from fastapi import HTTPException

    try:
        validate_texts(["a", "b", "c"])
    except HTTPException as exc:
        assert exc.status_code == 400
        return
    raise AssertionError("expected 400 for too many texts")


def test_validate_texts_too_long(monkeypatch):
    monkeypatch.setattr(
        security_module,
        "settings",
        replace(security_module.settings, max_chars_per_text=5),
    )
    from fastapi import HTTPException

    try:
        validate_texts(["this is way too long"])
    except HTTPException as exc:
        assert exc.status_code == 400
        return
    raise AssertionError("expected 400 for text too long")


def test_validate_request_size():
    from fastapi import HTTPException

    try:
        validate_request_size(10)
    except HTTPException:
        raise AssertionError("small body should be fine")
    try:
        validate_request_size(10**9)
    except HTTPException as exc:
        assert exc.status_code == 413
        return
    raise AssertionError("expected 413 for oversized body")
