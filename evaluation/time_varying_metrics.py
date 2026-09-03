from typing import List, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss
)


class TimeVaryingDecoderMetrics:
    """
    Metrics for multi-output QEC decoding.

    We evaluate both:

        1. Individual qubit prediction
        2. Complete error-pattern prediction
    """

    def calculate(
        self,
        y_true: List[List[int]],
        y_pred: List[List[int]]
    ) -> Dict:

        if len(y_true) == 0:
            raise ValueError(
                "y_true cannot be empty"
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have "
                "the same length"
            )

        # ---------------------------------------------
        # Flatten individual qubit predictions
        # ---------------------------------------------

        true_flat = [
            bit
            for target in y_true
            for bit in target
        ]

        pred_flat = [
            bit
            for target in y_pred
            for bit in target
        ]

        bit_accuracy = accuracy_score(
            true_flat,
            pred_flat
        )

        bit_precision = precision_score(
            true_flat,
            pred_flat,
            zero_division=0
        )

        bit_recall = recall_score(
            true_flat,
            pred_flat,
            zero_division=0
        )

        bit_f1 = f1_score(
            true_flat,
            pred_flat,
            zero_division=0
        )

        # ---------------------------------------------
        # Exact complete-pattern accuracy
        # ---------------------------------------------

        exact_matches = 0

        for true_target, pred_target in zip(
            y_true,
            y_pred
        ):

            if true_target == pred_target:
                exact_matches += 1

        exact_pattern_accuracy = (
            exact_matches
            / len(y_true)
        )

        # ---------------------------------------------
        # Hamming loss
        # ---------------------------------------------

        error_rate = hamming_loss(
            y_true,
            y_pred
        )

        return {
            "bit_accuracy": bit_accuracy,
            "bit_precision": bit_precision,
            "bit_recall": bit_recall,
            "bit_f1": bit_f1,
            "exact_pattern_accuracy":
                exact_pattern_accuracy,
            "hamming_loss": error_rate
        }

    def print_report(
        self,
        metrics: Dict,
        model_name: str
    ):

        print("\n===================================")
        print(
            f" {model_name.upper()} "
            "TIME-VARYING EVALUATION"
        )
        print("===================================")

        print(
            "\nBit Accuracy        : "
            f"{metrics['bit_accuracy']:.4f}"
        )

        print(
            "Bit Precision       : "
            f"{metrics['bit_precision']:.4f}"
        )

        print(
            "Bit Recall          : "
            f"{metrics['bit_recall']:.4f}"
        )

        print(
            "Bit F1              : "
            f"{metrics['bit_f1']:.4f}"
        )

        print(
            "Exact Pattern Acc.  : "
            f"{metrics['exact_pattern_accuracy']:.4f}"
        )

        print(
            "Hamming Loss        : "
            f"{metrics['hamming_loss']:.4f}"
        )