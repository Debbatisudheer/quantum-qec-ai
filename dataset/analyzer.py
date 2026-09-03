from collections import Counter
from typing import List, Dict

from dataset.schema import QECSample


class QECDatasetAnalyzer:
    """
    Analyze the contents of a QEC dataset.

    Current dataset:

        3-qubit bit-flip code

    Error classes:

        None
        q0
        q1
        q2
    """

    def analyze_error_distribution(
        self,
        dataset: List[QECSample]
    ) -> Dict[str, int]:
        """
        Count how many samples belong to
        each error class.
        """

        error_labels = []

        for sample in dataset:

            if sample.error_qubit is None:

                label = "none"

            else:

                label = (
                    f"q{sample.error_qubit}"
                )

            error_labels.append(label)

        counts = Counter(
            error_labels
        )

        # Always return all expected classes.
        # This makes the output predictable.

        return {
            "none": counts.get(
                "none",
                0
            ),

            "q0": counts.get(
                "q0",
                0
            ),

            "q1": counts.get(
                "q1",
                0
            ),

            "q2": counts.get(
                "q2",
                0
            ),
        }

    def analyze_logical_state_distribution(
        self,
        dataset: List[QECSample]
    ) -> Dict[int, int]:
        """
        Count logical |0>L and |1>L samples.
        """

        counts = Counter(
            sample.logical_state
            for sample in dataset
        )

        return {
            0: counts.get(0, 0),
            1: counts.get(1, 0),
        }

    def analyze_syndrome_distribution(
        self,
        dataset: List[QECSample]
    ) -> Dict[str, int]:
        """
        Count syndrome occurrences.
        """

        counts = Counter(
            sample.syndrome
            for sample in dataset
        )

        return {
            "00": counts.get(
                "00",
                0
            ),

            "10": counts.get(
                "10",
                0
            ),

            "11": counts.get(
                "11",
                0
            ),

            "01": counts.get(
                "01",
                0
            ),
        }

    def analyze(
        self,
        dataset: List[QECSample]
    ) -> Dict:
        """
        Perform complete dataset analysis.
        """

        if len(dataset) == 0:

            raise ValueError(
                "Dataset cannot be empty"
            )

        return {
            "total_samples": len(dataset),

            "error_distribution":
                self.analyze_error_distribution(
                    dataset
                ),

            "logical_state_distribution":
                self.analyze_logical_state_distribution(
                    dataset
                ),

            "syndrome_distribution":
                self.analyze_syndrome_distribution(
                    dataset
                ),
        }

    def print_report(
        self,
        analysis: Dict
    ):
        """
        Print a human-readable dataset report.
        """

        print("\n===================================")
        print(" QEC DATASET ANALYSIS")
        print("===================================")

        print(
            f"\nTotal samples: "
            f"{analysis['total_samples']}"
        )

        print("\nError distribution:")

        for label, count in (
            analysis[
                "error_distribution"
            ].items()
        ):

            print(
                f"  {label:>5} : {count}"
            )

        print(
            "\nLogical state distribution:"
        )

        for state, count in (
            analysis[
                "logical_state_distribution"
            ].items()
        ):

            print(
                f"  |{state}>L : {count}"
            )

        print(
            "\nSyndrome distribution:"
        )

        for syndrome, count in (
            analysis[
                "syndrome_distribution"
            ].items()
        ):

            print(
                f"  {syndrome} : {count}"
            )