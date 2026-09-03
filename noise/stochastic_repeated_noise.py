from noise.quantum_bit_flip import (
    QuantumBitFlipNoise
)


class StochasticRepeatedBitFlipNoise:
    """
    Generates stochastic bit-flip errors
    independently across repeated QEC rounds.

    The returned error history represents
    the accumulated physical X-error state
    after each round.

    Example:

        Round 1 -> [0, 0, 0]
        Round 2 -> [1, 0, 0]
        Round 3 -> [1, 0, 1]

    The error history is ground truth.
    It must NOT be given directly to the AI decoder.
    """

    def __init__(
        self,
        rounds=5,
        probability=0.01,
        seed=None
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        self.rounds = rounds

        self.noise = QuantumBitFlipNoise(
            probability=probability,
            seed=seed
        )

    def generate_error_history(self):
        """
        Generate stochastic physical X errors
        for every QEC round.

        Each round samples new X errors.

        The state is accumulated using XOR
        because:

            X * X = I
        """

        accumulated_state = [
            0,
            0,
            0
        ]

        error_history = []

        for _ in range(self.rounds):

            _, new_errors = self.noise.apply(
                self._dummy_circuit(),
                [0, 1, 2]
            )

            for qubit in range(3):

                accumulated_state[qubit] ^= (
                    new_errors[qubit]
                )

            error_history.append(
                accumulated_state.copy()
            )

        return error_history

    @staticmethod
    def _dummy_circuit():
        """
        The QuantumBitFlipNoise class requires
        a circuit so it can insert X gates.

        This method provides a minimal circuit
        for sampling the stochastic errors.

        The actual QEC circuit is built later
        using the generated error history.
        """

        from qiskit import QuantumCircuit

        return QuantumCircuit(3)