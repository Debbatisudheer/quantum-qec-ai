class LogicalRecovery:
    """
    Logical recovery for the 3-qubit repetition code.

    The code space is:

        |0>L = |000>
        |1>L = |111>

    After correction, majority voting determines
    the recovered logical state.

    Examples:

        000 -> logical 0
        001 -> logical 0
        010 -> logical 0
        100 -> logical 0

        111 -> logical 1
        110 -> logical 1
        101 -> logical 1
        011 -> logical 1
    """

    def validate_state(self, state):
        """
        Validate a 3-qubit physical state.
        """

        if len(state) != 3:
            raise ValueError(
                "state must contain exactly 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in state
        ):
            raise ValueError(
                "state must contain only 0 and 1"
            )

    def recover(self, state):
        """
        Recover the logical state using majority voting.
        """

        self.validate_state(state)

        number_of_ones = sum(state)

        if number_of_ones >= 2:
            return 1

        return 0

    def expected_state(self, logical_state):
        """
        Return the ideal physical codeword.
        """

        if logical_state == 0:
            return [0, 0, 0]

        if logical_state == 1:
            return [1, 1, 1]

        raise ValueError(
            "logical_state must be 0 or 1"
        )

    def is_logical_success(
        self,
        original_logical_state,
        corrected_state
    ):
        """
        Determine whether the recovered logical
        state matches the original logical state.
        """

        recovered_state = self.recover(
            corrected_state
        )

        return (
            recovered_state
            == original_logical_state
        )

    def recover_sample(
        self,
        original_logical_state,
        corrected_state
    ):
        """
        Complete logical recovery for one sample.
        """

        recovered_state = self.recover(
            corrected_state
        )

        success = (
            recovered_state
            == original_logical_state
        )

        return {
            "original_logical_state":
                original_logical_state,

            "recovered_logical_state":
                recovered_state,

            "logical_success":
                success,

            "logical_failure":
                not success
        }