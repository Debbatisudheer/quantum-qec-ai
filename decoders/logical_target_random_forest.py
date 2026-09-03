import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from decoders.logical_target import LogicalTargetBuilder


class LogicalTargetRandomForestDecoder:
    """
    Random Forest decoder trained to predict a
    logical-preserving physical correction.

    Input:
        observed syndrome history
        +
        detection event history

    Output:
        3-bit correction pattern

        [q0, q1, q2]
    """

    def __init__(
        self,
        rounds=5,
        n_estimators=100,
        random_seed=42
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than 0"
            )

        self.rounds = rounds
        self.n_estimators = n_estimators
        self.random_seed = random_seed

        self.target_builder = (
            LogicalTargetBuilder()
        )

        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_seed
        )

        self.model = MultiOutputClassifier(
            base_model
        )

        self.targets = {}
        self.target_scores = {}

        self.is_trained = False

    # ========================================================
    # FEATURE ENCODING
    # ========================================================

    @staticmethod
    def encode_features(sample):
        """
        Convert each round into:

            [syndrome_1,
             syndrome_2,
             detection_1,
             detection_2]

        Example for 5 rounds:

            5 x 4 = 20 features
        """

        syndrome_history = (
            sample["observed_syndrome_history"]
        )

        detection_events = (
            sample["detection_events"]
        )

        if len(syndrome_history) != len(
            detection_events
        ):
            raise ValueError(
                "syndrome history and detection "
                "event history must have the "
                "same length"
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
                    "detection event must contain "
                    "2 bits"
                )

            features.extend([
                int(syndrome[0]),
                int(syndrome[1]),
                int(detection[0]),
                int(detection[1]),
            ])

        return features

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        training_samples
    ):
        """
        Build logical-preserving targets
        from training data and train the
        Random Forest decoder.
        """

        if not training_samples:
            raise ValueError(
                "training_samples cannot be empty"
            )

        # Build logical targets ONLY from
        # training data.
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

            features = self.encode_features(
                sample
            )

            target = self.targets[
                observation
            ]

            X.append(features)
            y.append(target)

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.int64
        )

        self.model.fit(
            X,
            y
        )

        self.is_trained = True

        return self

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):
        """
        Predict correction patterns
        for a batch of feature vectors.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before prediction"
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        predictions = self.model.predict(
            X
        )

        return predictions.tolist()

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(self, X):
        """
        Return per-qubit class probabilities.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before prediction"
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        return self.model.predict_proba(
            X
        )

    # ========================================================
    # SINGLE SAMPLE DECODE
    # ========================================================

    def decode(self, sample):
        """
        Decode one complete QEC sample.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before decoding"
            )

        features = self.encode_features(
            sample
        )

        prediction = self.predict(
            [features]
        )

        return prediction[0]

    # ========================================================
    # BATCH DECODE
    # ========================================================

    def decode_batch(self, samples):
        """
        Decode multiple QEC samples.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before decoding"
            )

        X = np.asarray(
            [
                self.encode_features(sample)
                for sample in samples
            ],
            dtype=np.float32
        )

        return self.predict(X)

    # ========================================================
    # TRAINING TARGET
    # ========================================================

    def get_training_target(
        self,
        sample
    ):
        """
        Return the logical-preserving target
        assigned to a training observation.
        """

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

    # ========================================================
    # TARGET SCORE
    # ========================================================

    def get_target_score(
        self,
        sample
    ):
        """
        Return the empirical logical-success
        score of the training target for
        this observation.
        """

        observation = (
            self.target_builder.observation_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        if observation not in self.target_scores:
            return 0.0

        return self.target_scores[
            observation
        ]