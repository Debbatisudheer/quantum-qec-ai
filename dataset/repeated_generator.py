from syndrome.measurement_noise import (
    SyndromeMeasurementNoise
)


class RepeatedSyndromeDatasetGenerator:
    """
    Generate QEC samples containing repeated
    noisy syndrome measurements.

    Each sample contains:

        perfect_syndrome
        syndrome_history
        error_qubit

    Example:

        Perfect syndrome:
            01

        Measurement rounds:
            Round 1 -> 01
            Round 2 -> 01
            Round 3 -> 11
            Round 4 -> 01
            Round 5 -> 01

        Syndrome history:
            ["01", "01", "11", "01", "01"]

    Ground truth:
        error_qubit
        perfect_syndrome

    Observable input:
        syndrome_history
    """

    def __init__(
        self,
        rounds=5,
        measurement_noise_probability=0.0,
        seed=None
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability "
                "must be between 0 and 1"
            )

        self.rounds = rounds

        self.logical_states = [
            0,
            1
        ]

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
            probability=(
                measurement_noise_probability
            ),
            seed=seed
        )

    def generate_syndrome_history(
        self,
        perfect_syndrome
    ):
        """
        Generate repeated noisy syndrome
        observations.

        Every round independently applies
        measurement noise to the perfect syndrome.
        """

        if perfect_syndrome not in (
            "00",
            "10",
            "11",
            "01"
        ):
            raise ValueError(
                f"Invalid syndrome: "
                f"{perfect_syndrome}"
            )

        history = []

        for _ in range(self.rounds):

            observed_syndrome = (
                self.noise.apply(
                    perfect_syndrome
                )
            )

            history.append(
                observed_syndrome
            )

        return history

    def generate_sample(
        self,
        sample_id
    ):
        """
        Generate one repeated-syndrome sample.
        """

        logical_state = (
            self.noise.random.choice(
                self.logical_states
            )
        )

        error_qubit = (
            self.noise.random.choice(
                self.error_qubits
            )
        )

        perfect_syndrome = (
            self.syndrome_map[
                error_qubit
            ]
        )

        syndrome_history = (
            self.generate_syndrome_history(
                perfect_syndrome
            )
        )

        if logical_state == 0:

            original_state = "000"

        else:

            original_state = "111"

        corrupted_state = (
            original_state
        )

        if error_qubit is not None:

            state_bits = list(
                original_state
            )

            state_bits[
                error_qubit
            ] = (
                "1"
                if state_bits[
                    error_qubit
                ] == "0"
                else "0"
            )

            corrupted_state = (
                "".join(
                    state_bits
                )
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
            "syndrome_history": syndrome_history,
            "target": error_qubit,
        }

    def generate_dataset(
        self,
        num_samples
    ):
        """
        Generate a dataset containing
        repeated noisy syndrome histories.
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