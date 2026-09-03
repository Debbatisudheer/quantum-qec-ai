from typing import List, Dict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class DecoderMetrics:
    """
    Evaluation metrics for QEC decoders.

    These metrics compare:

        Actual error class
                vs
        Predicted error class
    """

    def calculate(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict:

        if len(y_true) == 0:
            raise ValueError(
                "y_true cannot be empty"
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must "
                "have the same length"
            )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
                2,
                3
            ]
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": matrix
        }

    def print_report(
        self,
        metrics: Dict,
        model_name: str
    ):
        """
        Print decoder evaluation results.
        """

        print("\n===================================")
        print(
            f" {model_name.upper()} EVALUATION"
        )
        print("===================================")

        print(
            f"\nAccuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['f1']:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )