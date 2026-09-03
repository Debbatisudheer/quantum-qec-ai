class RepeatedQuantumMeasurementParser:
    """
    Parse results from a repeated quantum QEC circuit.

    The circuit uses:

        classical bits 0,1,2
            -> final physical state

        classical bits 3+
            -> syndrome history

    Qiskit returns classical bits in reverse
    display order, so the result string is first
    reversed.
    """

    def __init__(self, rounds):

        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        self.rounds = rounds

    def validate_bitstring(
        self,
        bitstring
    ):
        if not isinstance(
            bitstring,
            str
        ):
            raise ValueError(
                "bitstring must be a string"
            )

        expected_length = (
            3
            + (2 * self.rounds)
        )

        if len(bitstring) != expected_length:
            raise ValueError(
                "Unexpected bitstring length"
            )

        if any(
            bit not in "01"
            for bit in bitstring
        ):
            raise ValueError(
                "bitstring must contain only 0 and 1"
            )

    def parse(
        self,
        bitstring
    ):
        """
        Return:

            final_state
            syndrome_history
        """

        self.validate_bitstring(
            bitstring
        )

        # Convert Qiskit display ordering
        # into classical-bit ordering.
        bits = bitstring[::-1]

        # c0,c1,c2
        final_state = (
            bits[0:3]
        )

        syndrome_history = []

        for round_index in range(
            self.rounds
        ):

            first_index = (
                3
                + (round_index * 2)
            )

            second_index = (
                first_index + 1
            )

            syndrome = (
                bits[first_index]
                + bits[second_index]
            )

            syndrome_history.append(
                syndrome
            )

        return {
            "final_state":
                final_state,

            "syndrome_history":
                syndrome_history
        }

    def extract_most_likely_result(
        self,
        counts
    ):
        if not counts:
            raise ValueError(
                "counts cannot be empty"
            )

        return max(
            counts,
            key=counts.get
        )