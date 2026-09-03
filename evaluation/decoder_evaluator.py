import time

from evaluation.logical_recovery import (
    LogicalRecovery
)


class DecoderEvaluator:
    """
    Reusable evaluator for QEC decoders.

    Evaluates:

        1. Exact error accuracy
        2. Physical recovery
        3. Bit accuracy
        4. Logical success
        5. Inference latency

    IMPORTANT:

    Logical success is evaluated using:

        encoded state
              +
        actual physical error
              +
        predicted correction
              ↓
        corrected physical state
              ↓
        logical recovery

    The decoder is NOT given the ground-truth
    physical error during prediction.
    """

    def __init__(self):
        self.recovery = LogicalRecovery()

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    @staticmethod
    def validate_samples(samples):
        if not samples:
            raise ValueError(
                "samples cannot be empty"
            )

    @staticmethod
    def validate_prediction(prediction):
        if not isinstance(
            prediction,
            (list, tuple)
        ):
            raise ValueError(
                "prediction must be a list or tuple"
            )

        if len(prediction) != 3:
            raise ValueError(
                "prediction must contain 3 bits"
            )

        if any(
            int(bit) not in (0, 1)
            for bit in prediction
        ):
            raise ValueError(
                "prediction must contain only 0 and 1"
            )

    # ========================================================
    # CORRUPTED STATE
    # ========================================================

    @staticmethod
    def corrupted_state(
        sample
    ):
        """
        Calculate:

            encoded_state XOR actual_error
        """

        encoded_state = [
            int(bit)
            for bit in sample[
                "encoded_state"
            ]
        ]

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        if len(encoded_state) != 3:
            raise ValueError(
                "encoded_state must contain 3 bits"
            )

        if len(actual_error) != 3:
            raise ValueError(
                "final_error_state must contain 3 bits"
            )

        return [
            encoded_state[index]
            ^ actual_error[index]
            for index in range(3)
        ]

    # ========================================================
    # CORRECTED STATE
    # ========================================================

    def corrected_state(
        self,
        sample,
        prediction
    ):
        """
        Calculate:

            corrupted_state XOR prediction
        """

        self.validate_prediction(
            prediction
        )

        corrupted = self.corrupted_state(
            sample
        )

        return [
            corrupted[index]
            ^ int(prediction[index])
            for index in range(3)
        ]

    # ========================================================
    # EXACT ERROR ACCURACY
    # ========================================================

    @staticmethod
    def exact_error_match(
        sample,
        prediction
    ):
        """
        True when predicted correction exactly
        matches the actual physical error pattern.
        """

        actual = tuple(
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        )

        predicted = tuple(
            int(bit)
            for bit in prediction
        )

        return actual == predicted

    # ========================================================
    # BIT ACCURACY
    # ========================================================

    def calculate_bit_accuracy(
        self,
        samples,
        predictions
    ):
        """
        Calculate per-bit prediction accuracy.
        """

        if len(samples) != len(predictions):
            raise ValueError(
                "samples and predictions "
                "must have the same length"
            )

        total_bits = 0
        correct_bits = 0

        for sample, prediction in zip(
            samples,
            predictions
        ):

            self.validate_prediction(
                prediction
            )

            actual = sample[
                "final_error_state"
            ]

            for actual_bit, predicted_bit in zip(
                actual,
                prediction
            ):

                total_bits += 1

                if (
                    int(actual_bit)
                    ==
                    int(predicted_bit)
                ):
                    correct_bits += 1

        if total_bits == 0:
            return 0.0

        return (
            correct_bits
            / total_bits
        )

    # ========================================================
    # LOGICAL SUCCESS
    # ========================================================

    def logical_success(
        self,
        sample,
        prediction
    ):
        """
        Determine whether the predicted
        correction preserves the logical state.
        """

        corrected = self.corrected_state(
            sample,
            prediction
        )

        recovered_logical = (
            self.recovery.recover(
                corrected
            )
        )

        expected_logical = int(
            sample["logical_state"]
        )

        return (
            recovered_logical
            ==
            expected_logical
        )

    # ========================================================
    # PHYSICAL RECOVERY
    # ========================================================

    def physical_recovery(
        self,
        sample,
        prediction
    ):
        """
        Physical recovery means the final
        corrected physical state exactly equals
        the encoded logical state.
        """

        corrected = self.corrected_state(
            sample,
            prediction
        )

        encoded = [
            int(bit)
            for bit in sample[
                "encoded_state"
            ]
        ]

        return corrected == encoded

    # ========================================================
    # EXACT ERROR ACCURACY
    # ========================================================

    def calculate_exact_accuracy(
        self,
        samples,
        predictions
    ):
        if len(samples) != len(predictions):
            raise ValueError(
                "samples and predictions "
                "must have the same length"
            )

        correct = 0

        for sample, prediction in zip(
            samples,
            predictions
        ):

            if self.exact_error_match(
                sample,
                prediction
            ):
                correct += 1

        return (
            correct / len(samples)
        )

    # ========================================================
    # PHYSICAL RECOVERY ACCURACY
    # ========================================================

    def calculate_physical_accuracy(
        self,
        samples,
        predictions
    ):
        if len(samples) != len(predictions):
            raise ValueError(
                "samples and predictions "
                "must have the same length"
            )

        correct = 0

        for sample, prediction in zip(
            samples,
            predictions
        ):

            if self.physical_recovery(
                sample,
                prediction
            ):
                correct += 1

        return (
            correct / len(samples)
        )

    # ========================================================
    # LOGICAL SUCCESS ACCURACY
    # ========================================================

    def calculate_logical_accuracy(
        self,
        samples,
        predictions
    ):
        if len(samples) != len(predictions):
            raise ValueError(
                "samples and predictions "
                "must have the same length"
            )

        correct = 0

        for sample, prediction in zip(
            samples,
            predictions
        ):

            if self.logical_success(
                sample,
                prediction
            ):
                correct += 1

        return (
            correct / len(samples)
        )

    # ========================================================
    # INFERENCE LATENCY
    # ========================================================

    @staticmethod
    def measure_inference(
        decoder,
        samples
    ):
        """
        Measure batch decoding time.

        The decoder's decode_batch()
        method is expected.
        """

        start = time.perf_counter()

        predictions = decoder.decode_batch(
            samples
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return predictions, elapsed

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    def evaluate(
        self,
        decoder,
        samples
    ):
        """
        Run complete decoder evaluation.

        Returns:

            exact
            physical
            bit
            logical
            inference_seconds
            samples_per_second
        """

        self.validate_samples(
            samples
        )

        predictions, inference_seconds = (
            self.measure_inference(
                decoder,
                samples
            )
        )

        exact = (
            self.calculate_exact_accuracy(
                samples,
                predictions
            )
        )

        physical = (
            self.calculate_physical_accuracy(
                samples,
                predictions
            )
        )

        bit = (
            self.calculate_bit_accuracy(
                samples,
                predictions
            )
        )

        logical = (
            self.calculate_logical_accuracy(
                samples,
                predictions
            )
        )

        if inference_seconds > 0:
            throughput = (
                len(samples)
                / inference_seconds
            )
        else:
            throughput = float("inf")

        return {
            "exact": exact,
            "physical": physical,
            "bit": bit,
            "logical": logical,
            "inference_seconds": (
                inference_seconds
            ),
            "samples_per_second": (
                throughput
            ),
            "sample_count": len(samples),
        }