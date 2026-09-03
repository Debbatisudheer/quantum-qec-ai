from typing import Optional

from dataset.schema import QECSample
from syndrome.measurement_noise import (
    SyndromeMeasurementNoise
)


class NoisyQECDatasetGenerator:
    """
    Generate QEC samples containing both:

        perfect_syndrome
        observed_syndrome

    The observed syndrome contains measurement noise.

    Ground truth:
        error_qubit
        perfect_syndrome

    Observable input:
        observed_syndrome
    """

    def __init__(
        self,
        measurement_noise_probability=0.0,
        seed=None
    ):
        self.logical_states = [0, 1]
        self.error_qubits = [
            None,
            0,
            1,
            2
        ]

        self.syndrome_map = {
            None: "00",
            0: "10",
            1: "11",
            2: "01",
        }

        self.noise = SyndromeMeasurementNoise(
            probability=measurement_noise_probability,
            seed=seed
        )

    def generate_sample(
        self,
        sample_id
    ):
        """
        Generate one noisy QEC sample.
        """

        logical_state = self.noise.random.choice(
            self.logical_states
        )

        error_qubit = self.noise.random.choice(
            self.error_qubits
        )

        perfect_syndrome = (
            self.syndrome_map[error_qubit]
        )

        observed_syndrome = self.noise.apply(
            perfect_syndrome
        )

        if logical_state == 0:
            original_state = "000"
        else:
            original_state = "111"

        corrupted_state = original_state

        if error_qubit is not None:

            state_bits = list(
                original_state
            )

            state_bits[error_qubit] = (
                "1"
                if state_bits[error_qubit] == "0"
                else "0"
            )

            corrupted_state = "".join(
                state_bits
            )

        if error_qubit is None:

            error_type = "none"

            error_description = (
                "No error"
            )

        else:

            error_type = "bit_flip"

            error_description = (
                f"X on q{error_qubit}"
            )

        return {
            "sample_id": sample_id,
            "qec_code": "bit_flip_3",
            "num_qubits": 3,
            "logical_state": logical_state,
            "original_state": original_state,
            "corrupted_state": corrupted_state,
            "error_type": error_type,
            "error_qubit": error_qubit,
            "error_description": error_description,
            "perfect_syndrome": perfect_syndrome,
            "observed_syndrome": observed_syndrome,
            "target": error_qubit,
        }

    def generate_dataset(
        self,
        num_samples
    ):
        """
        Generate a noisy dataset.
        """

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0"
            )

        return [
            self.generate_sample(
                sample_id=i
            )
            for i in range(num_samples)
        ]