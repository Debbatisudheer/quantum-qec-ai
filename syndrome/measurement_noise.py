import random


class SyndromeMeasurementNoise:
    """
    Measurement noise applied to syndrome bits.

    Each syndrome bit independently has a probability
    of being flipped.

    Example:

        Perfect syndrome:
            10

        Measurement noise:
            10% per bit

        Possible observed syndrome:
            00
            11
            10
            10
    """

    def __init__(
        self,
        probability=0.0,
        seed=None
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        self.probability = probability

        self.random = random.Random(seed)

    def flip_bit_if_needed(self, bit):
        """
        Flip one syndrome bit according to
        the configured measurement-noise probability.
        """

        if bit not in (0, 1):
            raise ValueError(
                "bit must be 0 or 1"
            )

        if self.random.random() < self.probability:
            return 1 - bit

        return bit

    def apply(self, syndrome):
        """
        Apply measurement noise to a syndrome.

        Example:

            "10" -> "00"

        or:

            "10" -> "11"

        depending on random noise.
        """

        if not isinstance(syndrome, str):
            raise ValueError(
                "syndrome must be a string"
            )

        if len(syndrome) == 0:
            raise ValueError(
                "syndrome cannot be empty"
            )

        if any(bit not in "01" for bit in syndrome):
            raise ValueError(
                "syndrome must contain only 0 and 1"
            )

        observed_bits = []

        for bit in syndrome:
            observed_bit = self.flip_bit_if_needed(
                int(bit)
            )

            observed_bits.append(
                str(observed_bit)
            )

        return "".join(observed_bits)

    def apply_to_bits(self, bits):
        """
        Apply measurement noise to a list of
        syndrome bits.

        Example:

            [1, 0] -> [0, 0]
        """

        if len(bits) == 0:
            raise ValueError(
                "bits cannot be empty"
            )

        observed_bits = []

        for bit in bits:
            observed_bit = self.flip_bit_if_needed(
                bit
            )

            observed_bits.append(
                observed_bit
            )

        return observed_bits