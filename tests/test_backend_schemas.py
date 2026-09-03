import pytest
from pydantic import ValidationError

from backend.api.schemas import (
    HealthResponse,
    SimulationRequest,
    SimulationResponse,
    ExperimentSummary,
)


# ============================================================
# HEALTH RESPONSE
# ============================================================


def test_health_response():
    response = HealthResponse(
        status="ok",
        service="quantum-qec-ai",
    )

    assert response.status == "ok"
    assert response.service == "quantum-qec-ai"


# ============================================================
# SIMULATION REQUEST
# ============================================================


def test_simulation_request_defaults():
    request = SimulationRequest()

    assert request.qec_code == "bit_flip_3"
    assert request.num_qubits == 3
    assert request.logical_state == 0
    assert request.rounds == 5

    assert (
        request.physical_noise_probability
        == 0.01
    )

    assert (
        request.measurement_noise_probability
        == 0.10
    )

    assert request.training_samples == 5000
    assert request.test_samples == 1000
    assert request.random_forest_estimators == 100
    assert request.seed == 42


def test_simulation_request_custom_values():
    request = SimulationRequest(
        qec_code="bit_flip_3",
        num_qubits=3,
        logical_state=1,
        rounds=7,
        physical_noise_probability=0.20,
        measurement_noise_probability=0.05,
        training_samples=1000,
        test_samples=200,
        random_forest_estimators=50,
        seed=123,
    )

    assert request.qec_code == "bit_flip_3"
    assert request.num_qubits == 3
    assert request.logical_state == 1
    assert request.rounds == 7

    assert (
        request.physical_noise_probability
        == 0.20
    )

    assert (
        request.measurement_noise_probability
        == 0.05
    )

    assert request.training_samples == 1000
    assert request.test_samples == 200
    assert request.random_forest_estimators == 50
    assert request.seed == 123


# ============================================================
# INVALID SIMULATION REQUESTS
# ============================================================


def test_invalid_logical_state():
    with pytest.raises(ValidationError):
        SimulationRequest(
            logical_state=2
        )


def test_invalid_rounds():
    with pytest.raises(ValidationError):
        SimulationRequest(
            rounds=0
        )


def test_invalid_physical_noise():
    with pytest.raises(ValidationError):
        SimulationRequest(
            physical_noise_probability=1.5
        )


def test_invalid_measurement_noise():
    with pytest.raises(ValidationError):
        SimulationRequest(
            measurement_noise_probability=-0.1
        )


def test_invalid_training_samples():
    with pytest.raises(ValidationError):
        SimulationRequest(
            training_samples=0
        )


def test_invalid_test_samples():
    with pytest.raises(ValidationError):
        SimulationRequest(
            test_samples=0
        )


def test_invalid_random_forest_estimators():
    with pytest.raises(ValidationError):
        SimulationRequest(
            random_forest_estimators=0
        )


# ============================================================
# SIMULATION RESPONSE
# ============================================================


def test_simulation_response():
    response = SimulationResponse(
        experiment_id="test-123",

        qec_code="bit_flip_3",
        num_qubits=3,
        rounds=5,
        logical_state=0,

        physical_noise_probability=0.01,
        measurement_noise_probability=0.10,

        training_samples=5000,
        test_samples=1000,

        decoder_type=(
            "logical_target_random_forest"
        ),

        logical_targets_learned=100,
        average_target_score=0.85,

        exact_accuracy=0.33,
        physical_accuracy=0.33,
        bit_accuracy=0.69,
        logical_accuracy=0.78,

        training_seconds=2.5,
        inference_seconds=0.1,
        samples_per_second=10000.0,

        status="completed",
    )

    assert response.experiment_id == "test-123"

    assert response.qec_code == "bit_flip_3"
    assert response.num_qubits == 3
    assert response.rounds == 5
    assert response.logical_state == 0

    assert (
        response.physical_noise_probability
        == 0.01
    )

    assert (
        response.measurement_noise_probability
        == 0.10
    )

    assert response.training_samples == 5000
    assert response.test_samples == 1000

    assert (
        response.decoder_type
        == "logical_target_random_forest"
    )

    assert response.logical_targets_learned == 100
    assert response.average_target_score == 0.85

    assert response.exact_accuracy == 0.33
    assert response.physical_accuracy == 0.33
    assert response.bit_accuracy == 0.69
    assert response.logical_accuracy == 0.78

    assert response.training_seconds == 2.5
    assert response.inference_seconds == 0.1
    assert response.samples_per_second == 10000.0

    assert response.status == "completed"


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================


def test_experiment_summary():
    summary = ExperimentSummary(
        experiment_id="test-123",

        qec_code="bit_flip_3",
        num_qubits=3,
        rounds=5,

        physical_noise_probability=0.10,
        measurement_noise_probability=0.10,

        training_samples=5000,
        test_samples=1000,

        decoder_type=(
            "logical_target_random_forest"
        ),

        exact_accuracy=0.33,
        physical_accuracy=0.33,
        bit_accuracy=0.69,
        logical_accuracy=0.78,

        training_seconds=2.5,
        inference_seconds=0.1,
        samples_per_second=10000.0,
    )

    assert summary.experiment_id == "test-123"
    assert summary.qec_code == "bit_flip_3"
    assert summary.num_qubits == 3
    assert summary.rounds == 5

    assert (
        summary.physical_noise_probability
        == 0.10
    )

    assert (
        summary.measurement_noise_probability
        == 0.10
    )

    assert summary.training_samples == 5000
    assert summary.test_samples == 1000

    assert (
        summary.decoder_type
        == "logical_target_random_forest"
    )

    assert summary.exact_accuracy == 0.33
    assert summary.physical_accuracy == 0.33
    assert summary.bit_accuracy == 0.69
    assert summary.logical_accuracy == 0.78

    assert summary.training_seconds == 2.5
    assert summary.inference_seconds == 0.1
    assert summary.samples_per_second == 10000.0