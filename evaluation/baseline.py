from typing import List

from decoders.lookup import LookupDecoder
from evaluation.decoder_metrics import DecoderMetrics


class BaselineEvaluator:
    """
    Evaluate the traditional lookup-table decoder.

    Current QEC code:

        3-qubit bit-flip code

    Mapping:

        00 -> 0
        10 -> 1
        11 -> 2
        01 -> 3
    """

    def __init__(self):

        self.decoder = LookupDecoder()

        self.metrics = DecoderMetrics()

    def decode_syndrome(
        self,
        syndrome: str
    ) -> int:
        """
        Convert a syndrome into the
        encoded target class.

        Mapping:

            No error -> 0
            q0       -> 1
            q1       -> 2
            q2       -> 3
        """

        error_qubit = self.decoder.decode(
            syndrome
        )

        if error_qubit is None:
            return 0

        return error_qubit + 1

    def predict(
        self,
        syndromes: List[str]
    ) -> List[int]:
        """
        Decode a list of syndrome strings.
        """

        predictions = []

        for syndrome in syndromes:

            prediction = (
                self.decode_syndrome(
                    syndrome
                )
            )

            predictions.append(
                prediction
            )

        return predictions

    def evaluate(
        self,
        syndromes: List[str],
        y_true: List[int]
    ):
        """
        Evaluate the baseline decoder.
        """

        if len(syndromes) != len(y_true):
            raise ValueError(
                "syndromes and y_true must "
                "have the same length"
            )

        predictions = self.predict(
            syndromes
        )

        metrics = self.metrics.calculate(
            y_true,
            predictions
        )

        return predictions, metrics