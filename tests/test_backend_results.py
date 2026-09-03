from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_get_result():
    experiments_response = client.get(
        "/experiments"
    )

    assert experiments_response.status_code == 200

    experiments = experiments_response.json()

    if not experiments:
        return

    experiment_id = experiments[0][
        "experiment_id"
    ]

    response = client.get(
        f"/results/{experiment_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["experiment_id"]
        == experiment_id
    )

    assert "config" in data

    assert "training_samples" in data
    assert "test_samples" in data

    assert "logical_targets_learned" in data
    assert "average_target_score" in data

    assert "exact_accuracy" in data
    assert "physical_accuracy" in data
    assert "bit_accuracy" in data
    assert "logical_accuracy" in data

    assert "training_seconds" in data
    assert "inference_seconds" in data
    assert "samples_per_second" in data

    assert "decoder_type" in data


def test_get_result_not_found():
    response = client.get(
        "/results/does-not-exist-123456"
    )

    assert response.status_code == 404

    data = response.json()

    assert (
        data["detail"]
        == "Experiment result not found: "
        "does-not-exist-123456"
    )