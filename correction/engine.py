class CorrectionEngine:
    """
    Correction engine for the 3-qubit bit-flip code.

    The decoder tells us which physical qubit
    contains the bit-flip error.

    We then apply X to that qubit.

    Example:

        Syndrome = 11
             ↓
        Decoder = q1
             ↓
        Apply X(q1)
    """

    def apply(self, circuit, error_qubit):
        """
        Apply the correction predicted by the decoder.

        Args:
            circuit: QuantumCircuit
            error_qubit: Physical qubit index or None.

        Returns:
            Corrected QuantumCircuit
        """

        # No error detected.
        if error_qubit is None:
            return circuit

        # Validate qubit.
        if error_qubit not in (0, 1, 2):
            raise ValueError(
                "error_qubit must be 0, 1, or 2"
            )

        # Apply X correction.
        circuit.x(error_qubit)

        return circuit