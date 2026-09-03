from dataclasses import dataclass
from typing import List

from dataset.schema import QECSample
from dataset.preprocessing import QECDatasetPreprocessor
from dataset.splitter import QECDatasetSplitter


@dataclass
class MLDataset:
    """
    Machine-learning-ready QEC dataset.

    Contains:

        Training data
        Validation data
        Test data

    Each split contains:

        X = feature vectors
        y = target classes
    """

    X_train: List[List[int]]
    y_train: List[int]

    X_validation: List[List[int]]
    y_validation: List[int]

    X_test: List[List[int]]
    y_test: List[int]


class MLDatasetBuilder:
    """
    Build an ML-ready dataset from QECSample objects.

    Pipeline:

        QECSamples
             ↓
          Split
             ↓
        Preprocess
             ↓
        ML Dataset
    """

    def __init__(
        self,
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    ):
        self.splitter = QECDatasetSplitter(
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )

        self.preprocessor = (
            QECDatasetPreprocessor()
        )

    def _transform_split(
        self,
        dataset: List[QECSample]
    ):
        """
        Transform one dataset split into:

            X
            y
        """

        return self.preprocessor.transform_dataset(
            dataset
        )

    def build(
        self,
        dataset: List[QECSample]
    ) -> MLDataset:
        """
        Build the complete ML dataset.

        Args:
            dataset: List of QECSample objects

        Returns:
            MLDataset
        """

        if len(dataset) == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        # --------------------------------
        # 1. Split raw dataset
        # --------------------------------

        (
            train_dataset,
            validation_dataset,
            test_dataset
        ) = self.splitter.split(
            dataset
        )

        # --------------------------------
        # 2. Preprocess training set
        # --------------------------------

        X_train, y_train = (
            self._transform_split(
                train_dataset
            )
        )

        # --------------------------------
        # 3. Preprocess validation set
        # --------------------------------

        X_validation, y_validation = (
            self._transform_split(
                validation_dataset
            )
        )

        # --------------------------------
        # 4. Preprocess test set
        # --------------------------------

        X_test, y_test = (
            self._transform_split(
                test_dataset
            )
        )

        # --------------------------------
        # 5. Create ML dataset
        # --------------------------------

        return MLDataset(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            X_test=X_test,
            y_test=y_test
        )

    def print_report(
        self,
        ml_dataset: MLDataset
    ):
        """
        Print ML dataset statistics.
        """

        print("\n===================================")
        print(" ML DATASET")
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

        print(
            "\nFeature count:"
        )

        if len(ml_dataset.X_train) > 0:
            print(
                f"  Features per sample : "
                f"{len(ml_dataset.X_train[0])}"
            )

        print(
            "\nTarget classes:"
        )

        classes = sorted(
            set(ml_dataset.y_train)
        )

        print(
            f"  {classes}"
        )

        print(
            "\nML input format:"
        )

        print(
            "  X = syndrome features"
        )

        print(
            "  y = error class"
        )