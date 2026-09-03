class RepeatedLookupDecoder:
    """
    Traditional lookup decoder for repeated
    3-qubit bit-flip QEC.

    The decoder uses the final observed
    syndrome from the repeated syndrome history.

    Syndrome mapping:

        00 -> no correction
        10 -> q0
        11 -> q1
        01 -> q2
    """

    def __init__(self):
        self.lookup_table = {
            "00": [0, 0, 0],
            "10": [1, 0, 0],
            "11": [0, 1, 0],
            "01": [0, 0, 1],
        }

    def validate_syndrome(self, syndrome):
        if not isinstance(syndrome, str):
            raise ValueError(
                "syndrome must be a string"
            )

        if len(syndrome) != 2:
            raise ValueError(
                "syndrome must contain 2 bits"
            )

        if any(
            bit not in "01"
            for bit in syndrome
        ):
            raise ValueError(
                "syndrome must contain only 0 and 1"
            )

    def decode(self, syndrome):
        self.validate_syndrome(syndrome)

        if syndrome not in self.lookup_table:
            raise ValueError(
                f"Unknown syndrome: {syndrome}"
            )

        return self.lookup_table[
            syndrome
        ].copy()

    def decode_history(
        self,
        observed_syndrome_history
    ):
        if not observed_syndrome_history:
            raise ValueError(
                "observed_syndrome_history "
                "cannot be empty"
            )

        final_syndrome = (
            observed_syndrome_history[-1]
        )

        return self.decode(
            final_syndrome
        )