from typing import List

from evaluation.decoder_metrics import DecoderMetrics


class MLEvaluator:
    """
    Evaluate an AI decoder.

    The evaluator compares:

        Actual error class
                vs
        AI predicted error class
    """

    def __init__(self):
        self.metrics = DecoderMetrics()

    def evaluate(
        self,
        decoder,
        X_test: List[List[int]],
        y_test: List[int]
    ):
        """
        Run the trained decoder on the test dataset
        and calculate evaluation metrics.
        """

        if len(X_test) == 0:
            raise ValueError(
                "X_test cannot be empty"
            )

        if len(y_test) == 0:
            raise ValueError(
                "y_test cannot be empty"
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                "X_test and y_test must "
                "have the same length"
            )

        predictions = decoder.predict(
            X_test
        )

        metrics = self.metrics.calculate(
            y_test,
            predictions
        )

        return predictions, metrics