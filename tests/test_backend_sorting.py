from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_sort_experiments_by_logical_accuracy():
    response = client.get(
        "/experiments",
        params={
            "sort_by": "logical_accuracy",
        },
    )

    assert response.status_code == 200

    data = response.json()

    for index in range(
        len(data) - 1
    ):
        assert (
            data[index]["logical_accuracy"]
            >=
            data[index + 1]["logical_accuracy"]
        )


def test_sort_experiments_ascending():
    response = client.get(
        "/experiments",
        params={
            "sort_by": "logical_accuracy",
            "descending": False,
        },
    )

    assert response.status_code == 200

    data = response.json()

    for index in range(
        len(data) - 1
    ):
        assert (
            data[index]["logical_accuracy"]
            <=
            data[index + 1]["logical_accuracy"]
        )


def test_sort_by_training_seconds():
    response = client.get(
        "/experiments",
        params={
            "sort_by": "training_seconds",
        },
    )

    assert response.status_code == 200

    data = response.json()

    for index in range(
        len(data) - 1
    ):
        assert (
            data[index]["training_seconds"]
            >=
            data[index + 1]["training_seconds"]
        )


def test_invalid_sort_metric():
    response = client.get(
        "/experiments",
        params={
            "sort_by": "invalid_metric",
        },
    )

    assert response.status_code == 400


def test_best_experiment():
    response = client.get(
        "/experiments/best",
        params={
            "metric": "logical_accuracy",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["experiment_id"]
    assert "logical_accuracy" in data


def test_worst_experiment():
    response = client.get(
        "/experiments/worst",
        params={
            "metric": "logical_accuracy",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["experiment_id"]
    assert "logical_accuracy" in data


def test_best_invalid_metric():
    response = client.get(
        "/experiments/best",
        params={
            "metric": "invalid_metric",
        },
    )

    assert response.status_code == 400


def test_worst_invalid_metric():
    response = client.get(
        "/experiments/worst",
        params={
            "metric": "invalid_metric",
        },
    )

    assert response.status_code == 400


def test_compare_experiments():
    experiments_response = client.get(
        "/experiments"
    )

    assert experiments_response.status_code == 200

    experiments = (
        experiments_response.json()
    )

    if len(experiments) < 2:
        return

    experiment_ids = ",".join(
        [
            experiments[0]["experiment_id"],
            experiments[1]["experiment_id"],
        ]
    )

    response = client.get(
        "/experiments/compare",
        params={
            "experiment_ids": experiment_ids,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert (
        data[0]["logical_accuracy"]
        >=
        data[1]["logical_accuracy"]
    )


def test_compare_empty_ids():
    response = client.get(
        "/experiments/compare",
        params={
            "experiment_ids": "",
        },
    )

    assert response.status_code == 400