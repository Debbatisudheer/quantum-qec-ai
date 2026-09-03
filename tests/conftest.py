import pytest
from collections import Counter, defaultdict

from dataset.time_varying_generator import TimeVaryingQECDatasetGenerator


ROUNDS = 5
PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10
SAMPLES = 25000
SEED = 42

CORRECTIONS = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]


@pytest.fixture(scope="session")
def samples():
    """Shared dataset fixture for the diagnostic tests."""
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=PHYSICAL_ERROR_PROBABILITY,
        measurement_noise_probability=MEASUREMENT_NOISE_PROBABILITY,
        seed=SEED,
    )

    return [
        generator.generate_sample(sample_id)
        for sample_id in range(SAMPLES)
    ]


@pytest.fixture(scope="session")
def groups(samples):
    """Observation-history -> final-error distribution."""
    result = defaultdict(Counter)

    for sample in samples:
        observation = "|".join(
            sample["observed_syndrome_history"]
        )
        error_state = tuple(
            int(bit)
            for bit in sample["final_error_state"]
        )
        result[observation][error_state] += 1

    return result


@pytest.fixture(scope="session")
def correction_statistics(groups):
    """Logical/exact/bit success for every correction and observation."""
    result = {}

    for observation, error_counts in groups.items():
        group_total = sum(error_counts.values())
        result[observation] = {}

        for correction in CORRECTIONS:
            logical_success_count = 0
            exact_error_count = 0
            bit_correct_count = 0
            total_bits = 0

            for actual_error, count in error_counts.items():
                if correction == actual_error:
                    exact_error_count += count

                for predicted_bit, actual_bit in zip(
                    correction,
                    actual_error,
                ):
                    if predicted_bit == actual_bit:
                        bit_correct_count += count
                    total_bits += count

                residual = [
                    int(a) ^ int(c)
                    for a, c in zip(actual_error, correction)
                ]

                # 3-qubit repetition-code logical preservation:
                # majority residual bit = 0.
                if sum(residual) <= 1:
                    logical_success_count += count

            result[observation][correction] = {
                "logical_probability": (
                    logical_success_count / group_total
                ),
                "exact_probability": (
                    exact_error_count / group_total
                ),
                "bit_probability": (
                    bit_correct_count / total_bits
                ),
            }

    return result


@pytest.fixture(params=[0.00, 0.10])
def measurement_noise_probability(request):
    """Measurement-noise values used by the perfect-decoder test."""
    return request.param
