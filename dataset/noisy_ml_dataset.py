from dataclasses import dataclass
from typing import List

from dataset.noisy_generator import (
    NoisyQECDatasetGenerator
)


@dataclass
class NoisyMLDataset:
    """
    ML dataset created from noisy QEC observations.

    X contains ONLY observed syndrome features.

    y contains the actual error class.

    Ground-truth information such as:

        perfect_syndrome
        error_qubit
        corrupted_state

    is NOT included in X.
    """

    X_train: List[List[int]]
    y_train: List[int]

    X_validation: List[List[int]]
    y_validation: List[int]

    X_test: List[List[int]]
    y_test: List[int]


class NoisyMLDatasetBuilder:
    """
    Convert noisy QEC samples into ML-ready data.

    Observable input:

        observed_syndrome

    Target:

        actual error_qubit
    """

    def __init__(
        self,
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    ):
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def encode_syndrome(
        self,
        syndrome: str
    ) -> List[int]:
        """
        Convert observed syndrome into ML features.

        Example:

            "10" -> [1, 0]
        """

        if syndrome not in (
            "00",
            "10",
            "11",
            "01"
        ):
            raise ValueError(
                f"Invalid syndrome: {syndrome}"
            )

        return [
            int(syndrome[0]),
            int(syndrome[1])
        ]

    def encode_target(
        self,
        error_qubit
    ) -> int:
        """
        Convert actual error location into
        a classification target.

        Classes:

            0 -> No error
            1 -> q0
            2 -> q1
            3 -> q2
        """

        if error_qubit not in (
            None,
            0,
            1,
            2
        ):
            raise ValueError(
                "error_qubit must be "
                "None, 0, 1, or 2"
            )

        if error_qubit is None:
            return 0

        return error_qubit + 1

    def transform_sample(
        self,
        sample
    ):
        """
        Transform one noisy sample.

        IMPORTANT:

        Features come from observed_syndrome.

        Target comes from error_qubit.
        """

        features = self.encode_syndrome(
            sample["observed_syndrome"]
        )

        target = self.encode_target(
            sample["error_qubit"]
        )

        return features, target

    def transform_dataset(
        self,
        dataset
    ):
        """
        Transform all noisy samples.
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

    def _split_indices(
        self,
        dataset
    ):
        """
        Create deterministic stratified
        train/validation/test indices.
        """

        from sklearn.model_selection import (
            train_test_split
        )

        indices = list(
            range(len(dataset))
        )

        targets = [
            self.encode_target(
                sample["error_qubit"]
            )
            for sample in dataset
        ]

        train_indices, temporary_indices = (
            train_test_split(
                indices,
                test_size=(
                    self.test_size
                    + self.validation_size
                ),
                stratify=targets,
                random_state=self.random_state
            )
        )

        temporary_targets = [
            targets[index]
            for index in temporary_indices
        ]

        validation_ratio = (
            self.validation_size
            /
            (
                self.test_size
                + self.validation_size
            )
        )

        validation_indices, test_indices = (
            train_test_split(
                temporary_indices,
                test_size=(
                    1.0 - validation_ratio
                ),
                stratify=temporary_targets,
                random_state=self.random_state
            )
        )

        return (
            train_indices,
            validation_indices,
            test_indices
        )

    def build(
        self,
        dataset
    ) -> NoisyMLDataset:
        """
        Build the complete noisy ML dataset.
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        (
            train_indices,
            validation_indices,
            test_indices
        ) = self._split_indices(
            dataset
        )

        train_dataset = [
            dataset[index]
            for index in train_indices
        ]

        validation_dataset = [
            dataset[index]
            for index in validation_indices
        ]

        test_dataset = [
            dataset[index]
            for index in test_indices
        ]

        X_train, y_train = (
            self.transform_dataset(
                train_dataset
            )
        )

        X_validation, y_validation = (
            self.transform_dataset(
                validation_dataset
            )
        )

        X_test, y_test = (
            self.transform_dataset(
                test_dataset
            )
        )

        return NoisyMLDataset(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_test=X_test,
            y_test=y_test
        )

    def print_report(
        self,
        ml_dataset
    ):
        """
        Print a summary of the noisy ML dataset.
        """

        print("\n===================================")
        print(" NOISY ML DATASET")
        print("===================================")

        print(
            f"\nTraining samples   : "
            f"{len(ml_dataset.X_train)}"
        )

        print(
            f"Validation samples : "
            f"{len(ml_dataset.X_validation)}"
        )

        print(
            f"Test samples       : "
            f"{len(ml_dataset.X_test)}"
        )

        print("\nFeature source:")
        print(
            "  X = observed_syndrome"
        )

        print("\nTarget source:")
        print(
            "  y = actual error_qubit"
        )

        if len(ml_dataset.X_train) > 0:

            print("\nFeatures per sample:")
            print(
                f"  "
                f"{len(ml_dataset.X_train[0])}"
            )

        print("\nTarget classes:")
        print(
            "  0 = No error"
        )
        print(
            "  1 = X on q0"
        )
        print(
            "  2 = X on q1"
        )
        print(
            "  3 = X on q2"
        )