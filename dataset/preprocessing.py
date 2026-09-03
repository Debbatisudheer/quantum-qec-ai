from typing import List, Tuple

from dataset.schema import QECSample


class QECDatasetPreprocessor:
    """
    Preprocess QEC dataset samples for machine learning.

    Current QEC code:

        3-qubit bit-flip code

    ML input:

        syndrome

    Example:

        "00" -> [0, 0]
        "10" -> [1, 0]
        "11" -> [1, 1]
        "01" -> [0, 1]

    ML target:

        None -> 0
        q0   -> 1
        q1   -> 2
        q2   -> 3
    """

    def __init__(self):

        self.target_map = {
            None: 0,
            0: 1,
            1: 2,
            2: 3,
        }

        self.reverse_target_map = {
            0: None,
            1: 0,
            2: 1,
            3: 2,
        }

    def encode_syndrome(
        self,
        syndrome: str
    ) -> List[int]:
        """
        Convert a two-bit syndrome string
        into numerical ML features.

        Examples:

            "00" -> [0, 0]
            "10" -> [1, 0]
            "11" -> [1, 1]
            "01" -> [0, 1]
        """

        if syndrome not in (
            "00",
            "10",
            "11",
            "01",
        ):
            raise ValueError(
                f"Invalid syndrome: {syndrome}"
            )

        return [
            int(syndrome[0]),
            int(syndrome[1]),
        ]

    def encode_target(
        self,
        error_qubit
    ) -> int:
        """
        Convert the physical error location
        into an ML class.

        Mapping:

            None -> 0
            q0   -> 1
            q1   -> 2
            q2   -> 3
        """

        if error_qubit not in (
            None,
            0,
            1,
            2,
        ):
            raise ValueError(
                "error_qubit must be "
                "None, 0, 1, or 2"
            )

        return self.target_map[
            error_qubit
        ]

    def decode_target(
        self,
        encoded_target: int
    ):
        """
        Convert an encoded ML class
        back into the physical error location.

        Mapping:

            0 -> None
            1 -> q0
            2 -> q1
            3 -> q2
        """

        if encoded_target not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError(
                "encoded_target must "
                "be 0, 1, 2, or 3"
            )

        return self.reverse_target_map[
            encoded_target
        ]

    def transform_sample(
        self,
        sample: QECSample
    ) -> Tuple[List[int], int]:
        """
        Transform one QECSample into:

            features
            target

        The decoder receives only the syndrome.

        Ground-truth information is NOT used
        as an input feature.
        """

        features = self.encode_syndrome(
            sample.syndrome
        )

        target = self.encode_target(
            sample.error_qubit
        )

        return features, target

    def transform_dataset(
        self,
        dataset: List[QECSample]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Transform an entire dataset into:

            X = ML features
            y = ML targets
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        features = []
        targets = []

        for sample in dataset:

            sample_features, sample_target = (
                self.transform_sample(
                    sample
                )
            )

            features.append(
                sample_features
            )

            targets.append(
                sample_target
            )

        return features, targets

    def print_sample_preview(
        self,
        dataset: List[QECSample],
        count: int = 10
    ):
        """
        Print a small preprocessing preview.
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        count = min(
            count,
            len(dataset)
        )

        print("\n===================================")
        print(" PREPROCESSING PREVIEW")
        print("===================================")

        print(
            "\nSyndrome → Features → Target"
        )

        for sample in dataset[:count]:

            features, target = (
                self.transform_sample(
                    sample
                )
            )

            print(
                f"  {sample.syndrome}"
                f" → {features}"
                f" → {target}"
            )