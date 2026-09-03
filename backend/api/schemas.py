from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# HEALTH
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str


# ============================================================
# STANDARD SIMULATION
# ============================================================

class SimulationRequest(BaseModel):
    qec_code: str = Field(
        default="bit_flip_3"
    )

    num_qubits: int = Field(
        default=3,
        gt=0,
    )

    logical_state: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    rounds: int = Field(
        default=5,
        gt=0,
    )

    physical_noise_probability: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
    )

    measurement_noise_probability: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )

    training_samples: int = Field(
        default=5000,
        gt=0,
    )

    test_samples: int = Field(
        default=1000,
        gt=0,
    )

    random_forest_estimators: int = Field(
        default=100,
        gt=0,
    )

    seed: Optional[int] = Field(
        default=42
    )


class SimulationResponse(BaseModel):
    experiment_id: str

    qec_code: str

    num_qubits: int

    rounds: int

    logical_state: int

    physical_noise_probability: float

    measurement_noise_probability: float

    training_samples: int

    test_samples: int

    decoder_type: str

    logical_targets_learned: int

    average_target_score: float

    exact_accuracy: float

    physical_accuracy: float

    bit_accuracy: float

    logical_accuracy: float

    training_seconds: float

    inference_seconds: float

    samples_per_second: float

    status: str


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

class ExperimentSummary(BaseModel):
    experiment_id: str

    qec_code: str

    num_qubits: int

    rounds: int

    physical_noise_probability: float

    measurement_noise_probability: float

    training_samples: int

    test_samples: int

    decoder_type: str

    exact_accuracy: float

    physical_accuracy: float

    bit_accuracy: float

    logical_accuracy: float

    training_seconds: float

    inference_seconds: float

    samples_per_second: float


# ============================================================
# EXPERIMENT RESULT
# ============================================================

class ExperimentResultResponse(BaseModel):
    experiment_id: str

    config: dict

    training_samples: int

    test_samples: int

    logical_targets_learned: int

    average_target_score: float

    exact_accuracy: float

    physical_accuracy: float

    bit_accuracy: float

    logical_accuracy: float

    training_seconds: float

    inference_seconds: float

    samples_per_second: float

    decoder_type: str


# ============================================================
# EXPERIMENT STATISTICS
# ============================================================

class ExperimentSummaryStatistics(BaseModel):
    count: int

    best_logical_accuracy: float | None

    average_logical_accuracy: float | None


class ExperimentAnalysisResponse(BaseModel):
    analysis: dict[str, Any]


# ============================================================
# TRACE REQUEST
# ============================================================

class SimulationTraceRequest(BaseModel):
    """
    Request for generating one complete
    single-sample QEC trace.
    """

    qec_code: str = Field(
        default="bit_flip_3"
    )

    num_qubits: int = Field(
        default=3,
        gt=0,
    )

    logical_state: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    rounds: int = Field(
        default=5,
        gt=0,
    )

    physical_noise_probability: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )

    measurement_noise_probability: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )

    training_samples: int = Field(
        default=5000,
        gt=0,
    )

    random_forest_estimators: int = Field(
        default=100,
        gt=0,
    )

    seed: Optional[int] = Field(
        default=42
    )


# ============================================================
# TRACE RESPONSE COMPONENTS
# ============================================================

class TraceRound(BaseModel):
    round: int

    physical_error_state: str

    perfect_syndrome: str

    observed_syndrome: str

    detection_event: str


class TraceNoise(BaseModel):
    physical_error_probability: float

    measurement_noise_probability: float

    physical_error_history: list[str]

    final_error_state: str

    final_error_description: str


class TraceQuantumState(BaseModel):
    encoded: str

    corrupted: str


class TraceSyndrome(BaseModel):
    perfect_history: list[str]

    observed_history: list[str]

    detection_events: list[str]

    final_perfect: str

    final_observed: str

    rounds: list[TraceRound]


class TraceDecoder(BaseModel):
    type: str

    training_samples: int

    random_forest_estimators: int

    predicted_correction: str

    predicted_correction_bits: list[int]

    confidence: float | None


class TraceCorrection(BaseModel):
    actual_error: str

    predicted_correction: str

    corrected_state: str


class TraceRecovery(BaseModel):
    original_logical_state: int

    recovered_logical_state: int

    logical_success: bool

    logical_failure: bool

    physical_recovery: bool

    exact_error_match: bool


# ============================================================
# TRACE RESPONSE
# ============================================================

class SimulationTraceResponse(BaseModel):
    sample_id: int

    qec_code: str

    num_qubits: int

    rounds: int

    logical_state: int

    encoded_state: str

    noise: TraceNoise

    quantum_state: TraceQuantumState

    syndrome: TraceSyndrome

    decoder: TraceDecoder

    correction: TraceCorrection

    recovery: TraceRecovery


# ============================================================
# SCIENTIFIC EVALUATION
# ============================================================

class ScientificResultResponse(BaseModel):
    physical_noise: float
    measurement_noise: float

    baseline_logical_success: float
    ai_logical_success: float

    absolute_gain: float
    relative_gain: float

    baseline_logical_error: float
    ai_logical_error: float
    logical_error_reduction: float

    bootstrap_ci_low: float
    bootstrap_ci_high: float

    permutation_p_value: float

    seed_count: int
    test_samples_per_seed: int


class ScientificSummaryResponse(BaseModel):
    conditions: int
    positive_gain_conditions: int

    maximum_absolute_gain: dict[str, Any]
    maximum_error_reduction: dict[str, Any]


class ScientificEvaluationResponse(BaseModel):
    source: str
    metadata: dict[str, Any]
    results: list[ScientificResultResponse]
    summary: ScientificSummaryResponse

# ============================================================
# BACKWARD-COMPATIBLE TRACE RESPONSE
# ============================================================

class TraceResponse(SimulationTraceResponse):
    """
    Backward-compatible alias/subclass for the
    single-simulation trace response.
    """

    pass

