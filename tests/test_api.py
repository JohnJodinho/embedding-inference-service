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

    for path in ("/health/live", "/health/ready"):
        assert "security" not in schema["paths"][path]["get"]


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


def test_embeddings_rate_limit_429(client, rag_headers, monkeypatch):
    from app import security
    import app.main as main_module

    # Exhaust the rate limiter bucket for 'rag'
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
