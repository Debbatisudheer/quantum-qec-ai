from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_get_experiments():
    response = client.get(
        "/experiments"
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(
        data,
        list
    )

    if data:
        experiment = data[0]

        assert "experiment_id" in experiment
        assert "qec_code" in experiment
        assert "num_qubits" in experiment
        assert "rounds" in experiment

        assert (
            "physical_noise_probability"
            in experiment
        )

        assert (
            "measurement_noise_probability"
            in experiment
        )

        assert "decoder_type" in experiment
        assert "logical_accuracy" in experiment
        assert "physical_accuracy" in experiment
        assert "bit_accuracy" in experiment
        assert "exact_accuracy" in experiment


def test_get_experiments_returns_valid_metrics():
    response = client.get(
        "/experiments"
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:
        assert 0.0 <= (
            experiment["logical_accuracy"]
        ) <= 1.0

        assert 0.0 <= (
            experiment["physical_accuracy"]
        ) <= 1.0

        assert 0.0 <= (
            experiment["bit_accuracy"]
        ) <= 1.0

        assert 0.0 <= (
            experiment["exact_accuracy"]
        ) <= 1.0