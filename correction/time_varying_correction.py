class TimeVaryingCorrectionEngine:
    """
    Correction engine for the 3-qubit bit-flip code
    with a predicted final physical error pattern.

    Error pattern:

        [0, 0, 0] -> no correction
        [1, 0, 0] -> X on q0
        [0, 1, 0] -> X on q1
        [0, 0, 1] -> X on q2

        [1, 1, 0] -> X on q0 and q1
        [1, 0, 1] -> X on q0 and q2
        [0, 1, 1] -> X on q1 and q2
        [1, 1, 1] -> X on q0, q1 and q2

    Important:
        Applying the same X pattern again cancels
        the accumulated X errors because:

            X X = I
    """

    def validate_error_state(self, error_state):
        """
        Validate a 3-qubit binary error pattern.
        """

        if len(error_state) != 3:
            raise ValueError(
                "error_state must contain exactly 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in error_state
        ):
            raise ValueError(
                "error_state must contain only 0 and 1"
            )

    def apply_correction(
        self,
        error_state,
        predicted_error_state
    ):
        """
        Apply the predicted correction to the
        actual accumulated physical error state.

        Since X * X = I, applying the same error
        pattern removes the corresponding errors.

        Example:

            Actual:
                [1, 0, 1]

            Predicted:
                [1, 0, 1]

            Corrected:
                [0, 0, 0]
        """

        self.validate_error_state(
            error_state
        )

        self.validate_error_state(
            predicted_error_state
        )

        corrected_state = []

        for actual, predicted in zip(
            error_state,
            predicted_error_state
        ):
            corrected_bit = actual ^ predicted

            corrected_state.append(
                corrected_bit
            )

        return corrected_state

    def is_physically_correct(
        self,
        corrected_state
    ):
        """
        Check whether all physical X errors
        have been removed.
        """

        self.validate_error_state(
            corrected_state
        )

        return corrected_state == [0, 0, 0]

    def describe_correction(
        self,
        predicted_error_state
    ):
        """
        Convert a predicted error pattern into
        a human-readable correction description.
        """

        self.validate_error_state(
            predicted_error_state
        )

        active_qubits = [
            index
            for index, bit
            in enumerate(predicted_error_state)
            if bit == 1
        ]

        if len(active_qubits) == 0:
            return "No correction required"

        return (
            "Apply X correction on "
            + ", ".join(
                f"q{qubit}"
                for qubit in active_qubits
            )
        )

    def correct_sample(
        self,
        actual_error_state,
        predicted_error_state
    ):
        """
        Complete correction operation for one sample.
        """

        corrected_state = self.apply_correction(
            actual_error_state,
            predicted_error_state
        )

        return {
            "actual_error_state":
                list(actual_error_state),

            "predicted_error_state":
                list(predicted_error_state),

            "corrected_state":
                corrected_state,

            "physically_correct":
                self.is_physically_correct(
                    corrected_state
                ),

            "correction_description":
                self.describe_correction(
                    predicted_error_state
                )
        }