from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_experiment_visualization():
    response = client.get(
        "/experiments/visualization"
    )

    assert response.status_code == 200

    data = response.json()

    assert "charts" in data
    assert "performance" in data

    charts = data["charts"]

    assert "rounds" in charts
    assert "physical_noise" in charts
    assert "measurement_noise" in charts
    assert "decoders" in charts

    assert isinstance(
        charts["rounds"],
        list,
    )

    assert isinstance(
        charts["physical_noise"],
        list,
    )

    assert isinstance(
        charts["measurement_noise"],
        list,
    )

    assert isinstance(
        charts["decoders"],
        list,
    )

    assert isinstance(
        data["performance"],
        list,
    )