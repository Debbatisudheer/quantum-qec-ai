from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_experiment_summary():
    response = client.get(
        "/experiments/summary"
    )

    assert response.status_code == 200

    data = response.json()

    assert "count" in data
    assert "best_logical_accuracy" in data
    assert "average_logical_accuracy" in data

    assert data["count"] >= 0

    if data["count"] > 0:
        assert (
            data["best_logical_accuracy"]
            is not None
        )

        assert (
            data["average_logical_accuracy"]
            is not None
        )


def test_experiment_analysis():
    response = client.get(
        "/experiments/analysis"
    )

    assert response.status_code == 200

    data = response.json()

    assert "analysis" in data

    assert isinstance(
        data["analysis"],
        dict,
    )