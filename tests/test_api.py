"""HTTP API contract tests using faked models/scheduler."""

from app.config import get_model_config


def test_openapi_documents_bearer_authentication():
    from app.main import app

    schema = app.openapi()
    scheme = schema["components"]["securitySchemes"]["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"

    protected_routes = (
        ("/v1/models", "get"),
        ("/v1/embeddings", "post"),
        ("/metrics", "get"),
    )
    for path, method in protected_routes:
        assert schema["paths"][path][method]["security"] == [{"BearerAuth": []}]
        assert "401" in schema["paths"][path][method]["responses"]

    for path in ("/health/live", "/health/ready"):
        assert "security" not in schema["paths"][path]["get"]

    request_body = schema["paths"]["/v1/embeddings"]["post"]["requestBody"]
    assert request_body["required"] is True
    body_schema = request_body["content"]["application/json"]["schema"]
    assert len(body_schema["oneOf"]) == 2
    assert "examples" in request_body["content"]["application/json"]
    assert "400" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "413" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "415" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "429" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "503" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "504" in schema["paths"]["/v1/embeddings"]["post"]["responses"]
    assert "embedding" in schema["components"]["schemas"]["EmbeddingItem"]["properties"]
    response_properties = schema["components"]["schemas"]["EmbeddingsResponse"]["properties"]
    assert {"model", "dimensions", "data", "usage"} <= response_properties.keys()
    assert schema["paths"]["/health/live"]["get"]["responses"]["200"]["content"] == {
        "text/plain": {"schema": {"type": "string"}}
    }
    assert "text/plain" in schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"]


def test_models_list(client, rag_headers):
    response = client.get("/v1/models", headers=rag_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    aliases = {item["alias"] for item in body["data"]}
    assert aliases == {"minilm-384-v1", "bge-base-768-v1"}
    for item in body["data"]:
        cfg = get_model_config(item["alias"])
        assert item["dimensions"] == cfg["dimensions"]
        assert item["model_id"] == cfg["model_id"]
        assert item["revision"] == cfg["revision"]


def test_models_list_requires_auth(client):
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_embeddings_single_text(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": "hello world", "normalize_embeddings": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "minilm-384-v1"
    assert body["dimensions"] == 384
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 384
    assert body["usage"]["input_texts"] == 1


def test_embeddings_preserves_order(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": ["first", "second", "third"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert len(body["data"]) == 3


def test_embeddings_bge(client, zit_headers):
    response = client.post(
        "/v1/embeddings",
        headers=zit_headers,
        json={
            "model": "bge-base-768-v1",
            "input": "What courses does ZIT offer?",
            "input_type": "query",
            "normalize_embeddings": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "bge-base-768-v1"
    assert body["dimensions"] == 768
    assert len(body["data"][0]["embedding"]) == 768


def test_embeddings_bge_document_no_prefix(client, zit_headers):
    response = client.post(
        "/v1/embeddings",
        headers=zit_headers,
        json={
            "model": "bge-base-768-v1",
            "input": ["doc one", "doc two"],
            "input_type": "document",
            "normalize_embeddings": True,
        },
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2


def test_embeddings_unknown_model(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "not-a-model", "input": "hi"},
    )
    assert response.status_code == 400


def test_embeddings_invalid_input(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": 123},
    )
    assert response.status_code == 400


def test_embeddings_empty_input(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": []},
    )
    assert response.status_code == 400


def test_embeddings_empty_body_returns_clear_error(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        content=b"",
    )
    assert response.status_code == 400
    assert response.json() == {"error": "empty_body"}


def test_embeddings_non_object_json_returns_bad_request(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json=["not", "an", "object"],
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_body"}


def test_embeddings_invalid_utf8_returns_bad_request(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers={
            "Authorization": rag_headers["Authorization"],
            "Content-Type": "application/json",
        },
        content=b"\xff",
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_json"}


def test_embeddings_normalize_embeddings_must_be_boolean(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={
            "model": "minilm-384-v1",
            "input": "hello",
            "normalize_embeddings": "false",
        },
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_input"}


def test_embeddings_rejects_unknown_fields(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": "hello", "unexpected": True},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_input"}


def test_embeddings_requires_json_content_type(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": rag_headers["Authorization"], "Content-Type": "text/plain"},
        content=b'{"model":"minilm-384-v1","input":"hello"}',
    )
    assert response.status_code == 415
    assert response.json() == {"error": "unsupported_media_type"}


def test_embeddings_streaming_body_limit(client, rag_headers, monkeypatch):
    from dataclasses import replace

    import app.embeddings_api as embeddings_api_module

    monkeypatch.setattr(
        embeddings_api_module,
        "settings",
        replace(embeddings_api_module.settings, max_request_bytes=10),
    )
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        content=b'{"model":"minilm"}',
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request_too_large"}


def test_embeddings_enforces_token_limit(client, rag_headers, monkeypatch):
    from dataclasses import replace

    import app.embeddings_api as embeddings_api_module

    monkeypatch.setattr(
        embeddings_api_module,
        "settings",
        replace(embeddings_api_module.settings, max_tokens_per_text=1),
    )
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": "one two"},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "too_many_tokens"}


def test_request_id_is_generated_and_returned(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32


def test_embeddings_requires_auth_before_body_validation(client):
    response = client.post(
        "/v1/embeddings",
        content=b"",
    )
    assert response.status_code == 401


def test_embeddings_bge_requires_input_type(client, zit_headers):
    response = client.post(
        "/v1/embeddings",
        headers=zit_headers,
        json={"model": "bge-base-768-v1", "input": "hi"},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "input_type_required"}


def test_embeddings_minilm_rejects_query_input_type(client, rag_headers):
    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": "hi", "input_type": "query"},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_input_type"}


def test_embeddings_rate_limit_429(client, rag_headers):
    from app.security import rate_limiter

    rate_limiter._buckets.clear()
    bucket = rate_limiter._buckets.get("rag")
    if bucket is None:
        from app.security import _TokenBucket

        bucket = _TokenBucket(capacity=0, fill_rate=0.0)
        rate_limiter._buckets["rag"] = bucket

    response = client.post(
        "/v1/embeddings",
        headers=rag_headers,
        json={"model": "minilm-384-v1", "input": "hi"},
    )
    assert response.status_code == 429


def test_metrics_requires_auth(client, rag_headers):
    response = client.get("/metrics")
    assert response.status_code == 401

    response = client.get("/metrics", headers=rag_headers)
    assert response.status_code == 200
