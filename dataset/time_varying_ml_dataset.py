from dataclasses import dataclass
from typing import List, Dict

from sklearn.model_selection import train_test_split


@dataclass
class TimeVaryingMLDataset:
    """
    ML-ready dataset for time-dependent QEC decoding.

    X:
        Observed temporal syndrome information.

    y:
        Final physical error pattern.

    Error pattern:

        000 -> no X errors
        100 -> X on q0
        010 -> X on q1
        001 -> X on q2
        110 -> X on q0 and q1
        101 -> X on q0 and q2
        011 -> X on q1 and q2
        111 -> X on q0, q1 and q2
    """

    X_train: List[List[int]]
    X_validation: List[List[int]]
    X_test: List[List[int]]

    y_train: List[List[int]]
    y_validation: List[List[int]]
    y_test: List[List[int]]


class TimeVaryingMLDatasetBuilder:
    """
    Convert time-dependent QEC samples into ML data.

    The decoder receives only observable information:

        observed_syndrome_history
        detection_events

    Ground truth such as:

        physical_error_history
        final_error_state
        perfect syndrome

    is used only as the target or for validation.
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
                "test_size + validation_size must be less than 1"
            )

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    # -------------------------------------------------
    # Encode one syndrome
    # -------------------------------------------------

    def encode_syndrome(
        self,
        syndrome
    ):
        """
        Convert a 2-bit syndrome into two ML features.

        Examples:

            00 -> [0, 0]
            10 -> [1, 0]
            11 -> [1, 1]
            01 -> [0, 1]
        """

        if not isinstance(syndrome, str):
            raise ValueError(
                "syndrome must be a string"
            )

        if len(syndrome) != 2:
            raise ValueError(
                "syndrome must contain exactly 2 bits"
            )

        if any(
            bit not in "01"
            for bit in syndrome
        ):
            raise ValueError(
                "syndrome must contain only 0 and 1"
            )

        return [
            int(syndrome[0]),
            int(syndrome[1])
        ]

    # -------------------------------------------------
    # Encode syndrome history
    # -------------------------------------------------

    def encode_syndrome_history(
        self,
        syndrome_history
    ):
        """
        Flatten a temporal syndrome history.

        Example:

            ["00", "11", "10"]

        becomes:

            [0,0,1,1,1,0]
        """

        if len(syndrome_history) == 0:
            raise ValueError(
                "syndrome_history cannot be empty"
            )

        features = []

        for syndrome in syndrome_history:

            features.extend(
                self.encode_syndrome(
                    syndrome
                )
            )

        return features

    # -------------------------------------------------
    # Encode detection events
    # -------------------------------------------------

    def encode_detection_events(
        self,
        detection_events
    ):
        """
        Flatten detection events.

        Example:

            ["00", "11", "01"]

        becomes:

            [0,0,1,1,0,1]
        """

        if len(detection_events) == 0:
            raise ValueError(
                "detection_events cannot be empty"
            )

        features = []

        for event in detection_events:

            features.extend(
                self.encode_syndrome(
                    event
                )
            )

        return features

    # -------------------------------------------------
    # Combine temporal features
    # -------------------------------------------------

    def encode_temporal_features(
        self,
        observed_syndrome_history,
        detection_events
    ):
        """
        Combine observed syndrome history and
        detection events into one feature vector.

        For N rounds:

            syndrome history = N * 2 features
            detection events = N * 2 features

        Total:

            N * 4 features
        """

        syndrome_features = (
            self.encode_syndrome_history(
                observed_syndrome_history
            )
        )

        detection_features = (
            self.encode_detection_events(
                detection_events
            )
        )

        return (
            syndrome_features
            + detection_features
        )

    # -------------------------------------------------
    # Encode target
    # -------------------------------------------------

    def encode_target(
        self,
        error_state
    ):
        """
        Encode the final physical error state.

        Example:

            [0,0,0] -> [0,0,0]
            [1,0,0] -> [1,0,0]
            [1,0,1] -> [1,0,1]
            [1,1,1] -> [1,1,1]

        The target is multi-output rather than
        a single class.
        """

        if len(error_state) != 3:
            raise ValueError(
                "error_state must contain 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in error_state
        ):
            raise ValueError(
                "error_state must contain only 0 and 1"
            )

        return [
            int(error_state[0]),
            int(error_state[1]),
            int(error_state[2])
        ]

    # -------------------------------------------------
    # Transform one sample
    # -------------------------------------------------

    def transform_sample(
        self,
        sample
    ):
        """
        Convert one time-dependent QEC sample
        into X and y.
        """

        if (
            "observed_syndrome_history"
            not in sample
        ):
            raise ValueError(
                "Missing observed_syndrome_history"
            )

        if (
            "detection_events"
            not in sample
        ):
            raise ValueError(
                "Missing detection_events"
            )

        if (
            "final_error_state"
            not in sample
        ):
            raise ValueError(
                "Missing final_error_state"
            )

        X = self.encode_temporal_features(
            sample[
                "observed_syndrome_history"
            ],
            sample[
                "detection_events"
            ]
        )

        y = self.encode_target(
            sample[
                "final_error_state"
            ]
        )

        return X, y

    # -------------------------------------------------
    # Build dataset
    # -------------------------------------------------

    def build(
        self,
        samples
    ):
        """
        Build train/validation/test ML datasets.
        """

        if len(samples) == 0:
            raise ValueError(
                "samples cannot be empty"
            )

        X = []
        y = []

        for sample in samples:

            features, target = (
                self.transform_sample(
                    sample
                )
            )

            X.append(features)
            y.append(target)

        # -------------------------------------------------
        # Stratification
        # -------------------------------------------------
        #
        # Multi-output targets cannot directly be passed
        # to sklearn's stratify parameter in the same way
        # as a single class label.
        #
        # We therefore create a temporary string label
        # representing the complete 3-bit error pattern.
        # This is ONLY used for splitting.
        #
        # The actual ML target remains multi-output.
        # -------------------------------------------------

        stratify_labels = [
            "".join(
                str(bit)
                for bit in target
            )
            for target in y
        ]

        (
            X_train,
            X_temp,
            y_train,
            y_temp,
            labels_train,
            labels_temp
        ) = train_test_split(
            X,
            y,
            stratify_labels,
            test_size=(
                self.test_size
                + self.validation_size
            ),
            random_state=self.random_state,
            stratify=stratify_labels
        )

        # -------------------------------------------------
        # Validation / test split
        # -------------------------------------------------

        relative_test_size = (
            self.test_size
            / (
                self.test_size
                + self.validation_size
            )
        )

        (
            X_validation,
            X_test,
            y_validation,
            y_test
        ) = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_test_size,
            random_state=self.random_state,
            stratify=labels_temp
        )

        return TimeVaryingMLDataset(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,

            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test
        )

    # -------------------------------------------------
    # Decode target
    # -------------------------------------------------

    def decode_target(
        self,
        target
    ):
        """
        Convert an encoded target back into
        a human-readable error description.
        """

        if len(target) != 3:
            raise ValueError(
                "target must contain 3 bits"
            )

        active_qubits = [
            index
            for index, bit
            in enumerate(target)
            if bit == 1
        ]

        if len(active_qubits) == 0:
            return "No X errors"

        return (
            "X errors on "
            + ", ".join(
                f"q{qubit}"
                for qubit in active_qubits
            )
        )

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    def print_report(
        self,
        ml_dataset
    ):
        """
        Print dataset information.
        """

        print("\n===================================")
        print(" TIME-VARYING ML DATASET")
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

        if len(ml_dataset.X_train) > 0:

            print(
                f"\nFeatures per sample: "
                f"{len(ml_dataset.X_train[0])}"
            )

        print(
            "\nInput:"
        )

        print(
            "  observed_syndrome_history"
        )

        print(
            "  detection_events"
        )

        print(
            "\nTarget:"
        )

        print(
            "  final_error_state"
        )

        print(
            "\nTarget format:"
        )

        print(
            "  [q0, q1, q2]"
        )