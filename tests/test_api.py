from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_dashboard_returns_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "SpamGuard AI" in response.text


def test_health_endpoint_returns_status():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_prediction_rejects_short_message():
    response = client.post(
        "/api/predict",
        json={"text": "a"},
    )

    assert response.status_code == 422


def test_prediction_rejects_missing_text():
    response = client.post(
        "/api/predict",
        json={},
    )

    assert response.status_code == 422


def test_metrics_endpoint_has_expected_structure():
    response = client.get("/api/metrics")

    assert response.status_code == 200

    report = response.json()

    assert "metrics" in report
    assert "accuracy" in report["metrics"]
    assert "confusion_matrix" in report["metrics"]