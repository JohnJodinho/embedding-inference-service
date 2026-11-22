"""Health endpoint tests."""

import app.models as models_module


def test_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.text == "ok"
    assert response.headers["X-Request-ID"]


def test_ready_when_models_loaded(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_when_models_missing(monkeypatch, client):
    monkeypatch.setattr(
        models_module,
        "_MODEL_CACHE",
        {"minilm-384-v1": models_module._MODEL_CACHE["minilm-384-v1"]},
    )
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ready_when_scheduler_stopped(monkeypatch, client):
    import app.routes as routes_module

    monkeypatch.setattr(routes_module.scheduler, "ready", lambda: False)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
