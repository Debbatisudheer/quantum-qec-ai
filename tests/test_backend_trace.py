from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_simulate_trace():
    response = client.post(
        "/simulate/trace",
        json={
            "qec_code": "bit_flip_3",
            "num_qubits": 3,
            "logical_state": 0,
            "rounds": 5,
            "physical_noise_probability": 0.10,
            "measurement_noise_probability": 0.10,
            "training_samples": 100,
            "random_forest_estimators": 10,
            "seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["qec_code"] == "bit_flip_3"
    assert data["num_qubits"] == 3
    assert data["rounds"] == 5

    assert data["logical_state"] in (0, 1)

    assert len(data["encoded_state"]) == 3

    assert "noise" in data
    assert "quantum_state" in data
    assert "syndrome" in data
    assert "decoder" in data
    assert "correction" in data
    assert "recovery" in data

    assert len(
        data["syndrome"]["perfect_history"]
    ) == 5

    assert len(
        data["syndrome"]["observed_history"]
    ) == 5

    assert len(
        data["syndrome"]["detection_events"]
    ) == 5

    assert len(
        data["syndrome"]["rounds"]
    ) == 5

    assert len(
        data["noise"]["physical_error_history"]
    ) == 5

    assert len(
        data["decoder"]["predicted_correction"]
    ) == 3

    assert len(
        data["correction"]["actual_error"]
    ) == 3

    assert len(
        data["correction"]["predicted_correction"]
    ) == 3

    assert len(
        data["correction"]["corrected_state"]
    ) == 3

    assert data["recovery"][
        "recovered_logical_state"
    ] in (0, 1)

    assert isinstance(
        data["recovery"]["logical_success"],
        bool,
    )


def test_simulate_trace_rejects_invalid_rounds():
    response = client.post(
        "/simulate/trace",
        json={
            "rounds": 0,
            "training_samples": 10,
            "random_forest_estimators": 5,
        },
    )

    assert response.status_code == 422


def test_simulate_trace_rejects_invalid_noise():
    response = client.post(
        "/simulate/trace",
        json={
            "rounds": 5,
            "physical_noise_probability": 1.5,
            "training_samples": 10,
            "random_forest_estimators": 5,
        },
    )

    assert response.status_code == 422


def test_simulate_trace_rejects_invalid_training_samples():
    response = client.post(
        "/simulate/trace",
        json={
            "rounds": 5,
            "training_samples": 0,
            "random_forest_estimators": 5,
        },
    )

    assert response.status_code == 422