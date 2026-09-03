import random


class BitFlipNoise:
    """
    Bit-flip noise model.

    Applies an X error to a selected qubit.
    """

    def __init__(self, probability=1.0, seed=None):
        """
        Args:
            probability: Probability of applying an X error.
            seed: Optional random seed for reproducibility.
        """

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        self.probability = probability

        if seed is not None:
            random.seed(seed)

    def should_error(self):
        """
        Decide whether an error occurs.
        """

        return random.random() < self.probability

    def apply(self, circuit, qubit):
        """
        Apply a bit-flip error to the selected qubit.

        Args:
            circuit: QuantumCircuit
            qubit: Physical qubit index

        Returns:
            QuantumCircuit
        """

        if self.should_error():
            circuit.x(qubit)

        return circuit