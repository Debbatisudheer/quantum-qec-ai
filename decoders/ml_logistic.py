from typing import List

from sklearn.linear_model import LogisticRegression


class LogisticRegressionDecoder:
    """
    AI decoder using Logistic Regression.

    Input:
        Syndrome features

    Output:
        Predicted error class

    Error classes:

        0 -> No error
        1 -> X on q0
        2 -> X on q1
        3 -> X on q2
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        self.is_trained = False

    def train(
        self,
        X_train: List[List[int]],
        y_train: List[int]
    ):
        """
        Train the Logistic Regression decoder.
        """

        if len(X_train) == 0:
            raise ValueError(
                "X_train cannot be empty"
            )

        if len(y_train) == 0:
            raise ValueError(
                "y_train cannot be empty"
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must "
                "have the same length"
            )

        self.model.fit(
            X_train,
            y_train
        )

        self.is_trained = True

    def predict(
        self,
        X: List[List[int]]
    ) -> List[int]:
        """
        Predict error classes.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained before prediction"
            )

        if len(X) == 0:
            raise ValueError(
                "X cannot be empty"
            )

        predictions = self.model.predict(X)

        return predictions.tolist()

    def predict_proba(
        self,
        X: List[List[int]]
    ):
        """
        Return prediction probabilities.

        These probabilities will later be useful
        for decoder confidence.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained before prediction"
            )

        if len(X) == 0:
            raise ValueError(
                "X cannot be empty"
            )

        return self.model.predict_proba(X)

    def decode(
        self,
        features: List[int]
    ) -> int:
        """
        Decode one syndrome feature vector.

        Example:

            [1, 0]
                ↓
            predicted class
                ↓
            1 = X on q0
        """

        predictions = self.predict(
            [features]
        )

        return predictions[0]

    def decode_with_confidence(
        self,
        features: List[int]
    ):
        """
        Decode one sample and return:

            predicted class
            confidence
        """

        probabilities = self.predict_proba(
            [features]
        )

        prediction = int(
            self.model.predict(
                [features]
            )[0]
        )

        confidence = float(
            probabilities[0][prediction]
        )

        return prediction, confidence