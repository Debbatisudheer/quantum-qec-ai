import numpy as np

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier
)

from decoders.logical_target import (
    LogicalTargetBuilder
)


class LogicalTargetGRUDecoder:
    """
    Temporal GRU decoder trained to predict
    a logical-preserving correction.

    Input:

        observed syndrome history
        +
        detection-event history

    Output:

        3-bit physical correction pattern

    The model is NOT trained to reconstruct
    the exact physical error.

    It is trained to produce a correction that
    maximizes logical-state preservation.
    """

    def __init__(
        self,
        rounds=5,
        hidden_size=64,
        learning_rate=0.003,
        epochs=100,
        random_seed=42
    ):

        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        self.rounds = rounds

        self.target_builder = (
            LogicalTargetBuilder()
        )

        self.model = TemporalGRUClassifier(
            input_size=4,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            epochs=epochs,
            random_seed=random_seed
        )

        self.targets = {}

    @staticmethod
    def encode_features(sample):

        syndrome_history = sample[
            "observed_syndrome_history"
        ]

        detection_events = sample[
            "detection_events"
        ]

        if len(syndrome_history) != len(
            detection_events
        ):
            raise ValueError(
                "syndrome history and detection "
                "event history must have the same length"
            )

        features = []

        for syndrome, detection in zip(
            syndrome_history,
            detection_events
        ):

            if len(syndrome) != 2:
                raise ValueError(
                    "syndrome must contain 2 bits"
                )

            if len(detection) != 2:
                raise ValueError(
                    "detection event must contain 2 bits"
                )

            features.append([
                int(syndrome[0]),
                int(syndrome[1]),
                int(detection[0]),
                int(detection[1])
            ])

        return features

    def train(
        self,
        training_samples,
        verbose=True
    ):
        """
        Build logical targets from training data,
        then train the GRU on those targets.
        """

        (
            self.targets,
            self.target_scores
        ) = self.target_builder.build(
            training_samples
        )

        X = []

        y = []

        for sample in training_samples:

            observation = (
                self.target_builder.observation_key(
                    sample[
                        "observed_syndrome_history"
                    ]
                )
            )

            X.append(
                self.encode_features(
                    sample
                )
            )

            y.append(
                self.targets[
                    observation
                ]
            )

        X = np.array(
            X,
            dtype=np.float32
        )

        y = np.array(
            y,
            dtype=np.int64
        )

        self.model.train(
            X,
            y,
            verbose=verbose
        )

        return self

    def predict(
        self,
        sample
    ):

        features = self.encode_features(
            sample
        )

        return self.model.decode(
            features
        )

    def decode(
        self,
        sample
    ):

        return self.predict(
            sample
        )

    def predict_batch(
        self,
        samples
    ):

        X = np.array(
            [
                self.encode_features(sample)
                for sample in samples
            ],
            dtype=np.float32
        )

        return self.model.predict(
            X
        )

    def get_training_target(
        self,
        sample
    ):

        observation = (
            self.target_builder.observation_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        return self.target_builder.get_target(
            observation,
            self.targets
        )