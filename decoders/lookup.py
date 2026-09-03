class LookupDecoder:
    """
    Traditional lookup-table decoder for the
    3-qubit bit-flip code.

    Syndrome mapping:

        00 -> No error
        10 -> X on q0
        11 -> X on q1
        01 -> X on q2
    """

    def __init__(self):
        self.lookup_table = {
            "00": None,
            "10": 0,
            "11": 1,
            "01": 2,
        }

    def decode(self, syndrome):
        """
        Decode a syndrome.

        Args:
            syndrome: Two-bit syndrome string.

        Returns:
            Physical qubit index containing the
            detected X error, or None.
        """

        if syndrome not in self.lookup_table:
            raise ValueError(
                f"Unknown syndrome: {syndrome}"
            )

        return self.lookup_table[syndrome]