import random


class QuantumBitFlipNoise:
    """
    Stochastic bit-flip noise for quantum circuits.

    For each selected qubit:

        random() < probability
                    ↓
                  X gate

    The actual error pattern is returned separately
    as ground truth for evaluation.
    """

    def __init__(
        self,
        probability=0.01,
        seed=None
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        self.probability = probability

        self.random = random.Random(
            seed
        )

    def should_error(self):
        return (
            self.random.random()
            < self.probability
        )

    def apply(
        self,
        circuit,
        qubits
    ):
        if not qubits:
            raise ValueError(
                "qubits cannot be empty"
            )

        error_state = [
            0
            for _ in qubits
        ]

        for index, qubit in enumerate(
            qubits
        ):

            if self.should_error():

                circuit.x(qubit)

                error_state[index] = 1

        return circuit, error_state