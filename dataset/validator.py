from typing import List, Tuple

from dataset.schema import QECSample


class QECDatasetValidator:
    """
    Validator for QEC dataset samples.

    This validator checks whether every sample
    is structurally and logically consistent.

    Current supported QEC code:

        3-qubit bit-flip code
    """

    def __init__(self):
        # ---------------------------------
        # Supported syndrome mapping
        # ---------------------------------

        self.syndrome_map = {
            None: "00",
            0: "10",
            1: "11",
            2: "01",
        }

        # ---------------------------------
        # Supported logical states
        # ---------------------------------

        self.logical_state_map = {
            0: "000",
            1: "111",
        }

    def validate_sample(
        self,
        sample: QECSample
    ) -> Tuple[bool, List[str]]:
        """
        Validate one QEC dataset sample.

        Returns:

            (True, [])
                if the sample is valid.

            (False, errors)
                if the sample contains errors.
        """

        errors = []

        # ---------------------------------
        # 1. Validate sample ID
        # ---------------------------------

        if not isinstance(
            sample.sample_id,
            int
        ):
            errors.append(
                "sample_id must be an integer"
            )

        # ---------------------------------
        # 2. Validate QEC code
        # ---------------------------------

        if sample.qec_code != "bit_flip_3":

            errors.append(
                "Unsupported QEC code"
            )

        # ---------------------------------
        # 3. Validate number of qubits
        # ---------------------------------

        if sample.num_qubits != 3:

            errors.append(
                "num_qubits must be 3"
            )

        # ---------------------------------
        # 4. Validate logical state
        # ---------------------------------

        if sample.logical_state not in (
            0,
            1
        ):

            errors.append(
                "logical_state must be 0 or 1"
            )

        # ---------------------------------
        # 5. Validate original state
        # ---------------------------------

        expected_original_state = (
            self.logical_state_map.get(
                sample.logical_state
            )
        )

        if (
            expected_original_state
            != sample.original_state
        ):

            errors.append(
                "original_state does not match "
                "logical_state"
            )

        # ---------------------------------
        # 6. Validate error qubit
        # ---------------------------------

        if sample.error_qubit not in (
            None,
            0,
            1,
            2
        ):

            errors.append(
                "error_qubit must be "
                "None, 0, 1, or 2"
            )

        # ---------------------------------
        # 7. Validate error type
        # ---------------------------------

        if sample.error_qubit is None:

            if sample.error_type != "none":

                errors.append(
                    "error_type must be 'none' "
                    "when error_qubit is None"
                )

        else:

            if sample.error_type != "bit_flip":

                errors.append(
                    "error_type must be "
                    "'bit_flip'"
                )

        # ---------------------------------
        # 8. Validate corrupted state
        # ---------------------------------

        expected_corrupted_state = (
            sample.original_state
        )

        if sample.error_qubit is not None:

            bits = list(
                sample.original_state
            )

            index = sample.error_qubit

            bits[index] = (
                "1"
                if bits[index] == "0"
                else "0"
            )

            expected_corrupted_state = (
                "".join(bits)
            )

        if (
            sample.corrupted_state
            != expected_corrupted_state
        ):

            errors.append(
                "corrupted_state does not match "
                "the injected error"
            )

        # ---------------------------------
        # 9. Validate syndrome
        # ---------------------------------

        expected_syndrome = (
            self.syndrome_map[
                sample.error_qubit
            ]
        )

        if sample.syndrome != expected_syndrome:

            errors.append(
                "syndrome does not match "
                "error_qubit"
            )

        # ---------------------------------
        # 10. Validate target
        # ---------------------------------

        if sample.target != sample.error_qubit:

            errors.append(
                "target must match error_qubit"
            )

        # ---------------------------------
        # 11. Validate target range
        # ---------------------------------

        if sample.target not in (
            None,
            0,
            1,
            2
        ):

            errors.append(
                "target must be "
                "None, 0, 1, or 2"
            )

        # ---------------------------------
        # Final result
        # ---------------------------------

        if len(errors) == 0:

            return True, []

        return False, errors

    def validate_dataset(
        self,
        dataset: List[QECSample]
    ) -> Tuple[bool, List[str]]:
        """
        Validate an entire dataset.

        Returns:

            (True, [])
                if every sample is valid.

            (False, errors)
                if one or more samples are invalid.
        """

        errors = []

        # ---------------------------------
        # Dataset must not be empty
        # ---------------------------------

        if len(dataset) == 0:

            return False, [
                "Dataset is empty"
            ]

        # ---------------------------------
        # Validate every sample
        # ---------------------------------

        for sample in dataset:

            valid, sample_errors = (
                self.validate_sample(
                    sample
                )
            )

            if not valid:

                for error in sample_errors:

                    errors.append(
                        f"Sample "
                        f"{sample.sample_id}: "
                        f"{error}"
                    )

        # ---------------------------------
        # Final result
        # ---------------------------------

        if len(errors) == 0:

            return True, []

        return False, errors