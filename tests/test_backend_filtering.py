from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_experiments_without_filter():
    response = client.get(
        "/experiments"
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(
        data,
        list
    )


def test_filter_by_rounds():
    response = client.get(
        "/experiments",
        params={
            "rounds": 5
        },
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:
        assert experiment["rounds"] == 5


def test_filter_by_physical_noise():
    response = client.get(
        "/experiments",
        params={
            "physical_noise_probability": 0.10
        },
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:
        assert (
            experiment[
                "physical_noise_probability"
            ]
            == 0.10
        )


def test_filter_by_measurement_noise():
    response = client.get(
        "/experiments",
        params={
            "measurement_noise_probability": 0.10
        },
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:
        assert (
            experiment[
                "measurement_noise_probability"
            ]
            == 0.10
        )


def test_filter_by_decoder():
    response = client.get(
        "/experiments",
        params={
            "decoder_type": (
                "logical_target_random_forest"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:
        assert (
            experiment["decoder_type"]
            == "logical_target_random_forest"
        )


def test_combined_filters():
    response = client.get(
        "/experiments",
        params={
            "rounds": 5,
            "physical_noise_probability": 0.10,
            "measurement_noise_probability": 0.10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    for experiment in data:

        assert experiment["rounds"] == 5

        assert (
            experiment[
                "physical_noise_probability"
            ]
            == 0.10
        )

        assert (
            experiment[
                "measurement_noise_probability"
            ]
            == 0.10
        )