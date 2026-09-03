import random

from dataset.schema import QECSample


class QECDatasetGenerator:
    """
    Dataset generator for the 3-qubit bit-flip
    quantum error-correcting code.
    """

    def __init__(self, seed=None):
        """
        Initialize the dataset generator.

        Args:
            seed: Optional random seed for
                  reproducible datasets.
        """

        self.random = random.Random(seed)

        self.logical_states = [0, 1]

        self.error_qubits = [
            None,
            0,
            1,
            2,
        ]

        self.syndrome_map = {
            None: "00",
            0: "10",
            1: "11",
            2: "01",
        }

    def generate_sample(self, sample_id):
        """
        Generate one QEC dataset sample.
        """

        # ---------------------------------
        # Logical state
        # ---------------------------------

        logical_state = self.random.choice(
            self.logical_states
        )

        # ---------------------------------
        # Physical error
        # ---------------------------------

        error_qubit = self.random.choice(
            self.error_qubits
        )

        # ---------------------------------
        # Syndrome
        # ---------------------------------

        syndrome = self.syndrome_map[
            error_qubit
        ]

        # ---------------------------------
        # Original logical state
        # ---------------------------------

        if logical_state == 0:
            original_state = "000"
        else:
            original_state = "111"

        # ---------------------------------
        # Corrupted physical state
        # ---------------------------------

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

        # ---------------------------------
        # Error description
        # ---------------------------------

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

        # ---------------------------------
        # Create standardized sample
        # ---------------------------------

        sample = QECSample(
            sample_id=sample_id,

            qec_code="bit_flip_3",

            num_qubits=3,

            logical_state=logical_state,

            original_state=original_state,

            corrupted_state=corrupted_state,

            error_type=error_type,

            error_qubit=error_qubit,

            error_description=error_description,

            syndrome=syndrome,

            target=error_qubit,
        )

        return sample

    def generate_dataset(self, num_samples):
        """
        Generate multiple QEC samples.

        Args:
            num_samples: Number of samples.

        Returns:
            List of QECSample objects.
        """

        if num_samples <= 0:

            raise ValueError(
                "num_samples must be greater than 0"
            )

        dataset = []

        for sample_id in range(num_samples):

            sample = self.generate_sample(
                sample_id=sample_id
            )

            dataset.append(sample)

        return dataset