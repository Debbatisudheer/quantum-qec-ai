import random

from syndrome.measurement_noise import (
    SyndromeMeasurementNoise
)


class TimeVaryingQECDatasetGenerator:
    """
    Generate repeated-QEC samples with time-dependent
    physical bit-flip errors.

    Each sample now explicitly contains a logical state:

        logical_state = 0
            |0>L = |000>

        logical_state = 1
            |1>L = |111>

    The physical X-error process is modeled independently
    from the logical state.

    This is still a controlled synthetic model rather
    than a full round-by-round quantum-circuit QEC
    implementation.
    """

    def __init__(
        self,
        rounds=5,
        physical_error_probability=0.01,
        measurement_noise_probability=0.10,
        seed=None
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= physical_error_probability <= 1.0:
            raise ValueError(
                "physical_error_probability "
                "must be between 0 and 1"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability "
                "must be between 0 and 1"
            )

        self.rounds = rounds

        self.physical_error_probability = (
            physical_error_probability
        )

        self.measurement_noise_probability = (
            measurement_noise_probability
        )

        self.random = random.Random(seed)

        self.measurement_noise = (
            SyndromeMeasurementNoise(
                probability=(
                    measurement_noise_probability
                ),
                seed=seed
            )
        )

    def apply_physical_noise(
        self,
        error_state
    ):
        """
        Independently apply a possible X error
        to each physical qubit.

        error_state:

            [q0, q1, q2]

        0 = even X-error parity
        1 = odd X-error parity
        """

        if len(error_state) != 3:
            raise ValueError(
                "error_state must contain 3 qubits"
            )

        for qubit in range(3):

            if (
                self.random.random()
                < self.physical_error_probability
            ):
                error_state[qubit] ^= 1

        return error_state

    def calculate_syndrome(
        self,
        error_state
    ):
        """
        Calculate perfect syndrome.

        S1 = q0 XOR q1
        S2 = q1 XOR q2
        """

        if len(error_state) != 3:
            raise ValueError(
                "error_state must contain 3 qubits"
            )

        q0, q1, q2 = error_state

        s1 = q0 ^ q1
        s2 = q1 ^ q2

        return f"{s1}{s2}"

    def describe_error_state(
        self,
        error_state
    ):
        """
        Convert accumulated physical error state
        into a readable description.
        """

        active_qubits = [
            index
            for index, value
            in enumerate(error_state)
            if value == 1
        ]

        if len(active_qubits) == 0:
            return "No accumulated X error"

        return (
            "Accumulated X errors on "
            + ", ".join(
                f"q{qubit}"
                for qubit in active_qubits
            )
        )

    def calculate_detection_events(
        self,
        syndrome_history
    ):
        """
        Calculate syndrome-change events.
        """

        if len(syndrome_history) == 0:
            raise ValueError(
                "syndrome_history cannot be empty"
            )

        detection_events = []

        previous_syndrome = "00"

        for syndrome in syndrome_history:

            if len(syndrome) != 2:
                raise ValueError(
                    "Each syndrome must contain 2 bits"
                )

            previous_bits = [
                int(previous_syndrome[0]),
                int(previous_syndrome[1])
            ]

            current_bits = [
                int(syndrome[0]),
                int(syndrome[1])
            ]

            event_s1 = (
                previous_bits[0]
                ^ current_bits[0]
            )

            event_s2 = (
                previous_bits[1]
                ^ current_bits[1]
            )

            detection_events.append(
                f"{event_s1}{event_s2}"
            )

            previous_syndrome = syndrome

        return detection_events

    def generate_sample(
        self,
        sample_id
    ):
        """
        Generate one time-dependent QEC sample.
        """

        # --------------------------------
        # Logical state
        # --------------------------------

        logical_state = (
            self.random.choice([0, 1])
        )

        if logical_state == 0:
            encoded_state = "000"
        else:
            encoded_state = "111"

        # --------------------------------
        # Physical error state
        # --------------------------------

        error_state = [0, 0, 0]

        syndrome_history = []
        observed_syndrome_history = []
        physical_error_history = []

        for _ in range(self.rounds):

            self.apply_physical_noise(
                error_state
            )

            physical_error_history.append(
                error_state.copy()
            )

            perfect_syndrome = (
                self.calculate_syndrome(
                    error_state
                )
            )

            syndrome_history.append(
                perfect_syndrome
            )

            observed_syndrome = (
                self.measurement_noise.apply(
                    perfect_syndrome
                )
            )

            observed_syndrome_history.append(
                observed_syndrome
            )

        detection_events = (
            self.calculate_detection_events(
                observed_syndrome_history
            )
        )

        final_error_state = (
            error_state.copy()
        )

        final_syndrome = (
            syndrome_history[-1]
        )

        final_observed_syndrome = (
            observed_syndrome_history[-1]
        )

        return {
            "sample_id":
                sample_id,

            "qec_code":
                "bit_flip_3",

            "num_qubits":
                3,

            "rounds":
                self.rounds,

            "logical_state":
                logical_state,

            "encoded_state":
                encoded_state,

            "physical_error_probability":
                self.physical_error_probability,

            "measurement_noise_probability":
                self.measurement_noise_probability,

            "syndrome_history":
                syndrome_history,

            "observed_syndrome_history":
                observed_syndrome_history,

            "detection_events":
                detection_events,

            "physical_error_history":
                physical_error_history,

            "final_error_state":
                final_error_state,

            "final_syndrome":
                final_syndrome,

            "final_observed_syndrome":
                final_observed_syndrome,

            "final_error_description":
                self.describe_error_state(
                    final_error_state
                )
        }

    def generate_dataset(
        self,
        num_samples
    ):
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