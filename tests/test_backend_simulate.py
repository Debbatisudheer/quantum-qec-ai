from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_simulate_endpoint():
    response = client.post(
        "/simulate",
        json={
            "qec_code": "bit_flip_3",
            "num_qubits": 3,
            "logical_state": 1,
            "rounds": 1,
            "physical_noise_probability": 0.10,
            "measurement_noise_probability": 0.10,
            "training_samples": 20,
            "test_samples": 10,
            "random_forest_estimators": 10,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["experiment_id"]
    assert data["qec_code"] == "bit_flip_3"
    assert data["num_qubits"] == 3
    assert data["rounds"] == 1
    assert data["logical_state"] == 1

    assert data["training_samples"] == 20
    assert data["test_samples"] == 10

    assert (
        data["decoder_type"]
        == "logical_target_random_forest"
    )

    assert "exact_accuracy" in data
    assert "physical_accuracy" in data
    assert "bit_accuracy" in data
    assert "logical_accuracy" in data

    assert "training_seconds" in data
    assert "inference_seconds" in data
    assert "samples_per_second" in data

    assert data["status"] == "completed"


def test_simulate_validation():
    response = client.post(
        "/simulate",
        json={
            "qec_code": "bit_flip_3",
            "num_qubits": 3,
            "logical_state": 2,
            "rounds": 5,
            "physical_noise_probability": 0.10,
            "measurement_noise_probability": 0.10,
        },
    )

    assert response.status_code == 422