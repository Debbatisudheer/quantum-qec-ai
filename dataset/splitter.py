from typing import List, Tuple

from sklearn.model_selection import train_test_split

from dataset.schema import QECSample


class QECDatasetSplitter:
    """
    Split a QEC dataset into:

        Training
        Validation
        Test

    using stratified sampling.

    Default split:

        80% training
        10% validation
        10% test
    """

    def __init__(
        self,
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    ):
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                "test_size must be between 0 and 1"
            )

        if not 0.0 < validation_size < 1.0:
            raise ValueError(
                "validation_size must be between 0 and 1"
            )

        if test_size + validation_size >= 1.0:
            raise ValueError(
                "test_size + validation_size "
                "must be less than 1"
            )

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def _get_targets(
        self,
        dataset: List[QECSample]
    ) -> List[int]:
        """
        Extract target classes from the dataset.

        Mapping:

            None -> 0
            q0   -> 1
            q1   -> 2
            q2   -> 3
        """

        targets = []

        for sample in dataset:

            if sample.error_qubit is None:
                target = 0
            else:
                target = sample.error_qubit + 1

            targets.append(target)

        return targets

    def split(
        self,
        dataset: List[QECSample]
    ) -> Tuple[
        List[QECSample],
        List[QECSample],
        List[QECSample]
    ]:
        """
        Split the dataset into:

            train
            validation
            test
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        targets = self._get_targets(
            dataset
        )

        train_dataset, temporary_dataset, \
            train_targets, temporary_targets = (
                train_test_split(
                    dataset,
                    targets,
                    test_size=(
                        self.test_size
                        + self.validation_size
                    ),
                    stratify=targets,
                    random_state=self.random_state
                )
            )

        validation_ratio = (
            self.validation_size
            / (
                self.test_size
                + self.validation_size
            )
        )

        validation_dataset, test_dataset, \
            _, _ = (
                train_test_split(
                    temporary_dataset,
                    temporary_targets,
                    test_size=(
                        1.0 - validation_ratio
                    ),
                    stratify=temporary_targets,
                    random_state=self.random_state
                )
            )

        return (
            train_dataset,
            validation_dataset,
            test_dataset
        )

    def print_report(
        self,
        train_dataset: List[QECSample],
        validation_dataset: List[QECSample],
        test_dataset: List[QECSample]
    ):
        """
        Print dataset split statistics.
        """

        total = (
            len(train_dataset)
            + len(validation_dataset)
            + len(test_dataset)
        )

        print("\n===================================")
        print(" QEC DATASET SPLIT")
        print("===================================")

        print(
            f"\nTotal samples      : {total}"
        )

        print(
            f"Training samples   : "
            f"{len(train_dataset)}"
        )

        print(
            f"Validation samples : "
            f"{len(validation_dataset)}"
        )

        print(
            f"Test samples       : "
            f"{len(test_dataset)}"
        )

        print(
            "\nExpected split:"
        )

        print(
            "  Training   : 80%"
        )

        print(
            "  Validation : 10%"
        )

        print(
            "  Test       : 10%"
        )