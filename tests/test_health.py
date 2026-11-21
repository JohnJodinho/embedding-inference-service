"""Health endpoint tests."""

from app import main as main_module
import app.models as models_module


def test_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200


def test_ready_when_models_loaded(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_when_models_missing(monkeypatch, client):
    # Replace the model cache with an empty one so readiness fails.
    monkeypatch.setattr(models_module, "_MODEL_CACHE", {})
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
