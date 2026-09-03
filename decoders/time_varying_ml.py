from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class TimeVaryingDecoderBase:
    """
    Base interface for time-varying QEC decoders.

    The decoder predicts three independent binary
    outputs:

        q0 -> 0 or 1
        q1 -> 0 or 1
        q2 -> 0 or 1

    Together they form the predicted physical
    error pattern.
    """

    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def decode(self, features):
        prediction = self.predict(
            [features]
        )

        return list(
            prediction[0]
        )


class TimeVaryingLogisticDecoder(
    TimeVaryingDecoderBase
):
    """
    Multi-output Logistic Regression decoder.
    """

    def __init__(
        self,
        random_state=42
    ):
        base_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )

        self.model = MultiOutputClassifier(
            base_model
        )

    def train(self, X, y):
        self.model.fit(
            X,
            y
        )

        return self

    def predict(self, X):
        predictions = (
            self.model.predict(X)
        )

        return predictions.tolist()

    def predict_proba(self, X):
        return self.model.predict_proba(
            X
        )

    def decode(self, features):
        return super().decode(
            features
        )


class TimeVaryingRandomForestDecoder(
    TimeVaryingDecoderBase
):
    """
    Multi-output Random Forest decoder.
    """

    def __init__(
        self,
        n_estimators=100,
        random_state=42
    ):
        base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        self.model = MultiOutputClassifier(
            base_model
        )

    def train(self, X, y):
        self.model.fit(
            X,
            y
        )

        return self

    def predict(self, X):
        predictions = (
            self.model.predict(X)
        )

        return predictions.tolist()

    def predict_proba(self, X):
        return self.model.predict_proba(
            X
        )

    def decode(self, features):
        return super().decode(
            features
        )


class TimeVaryingMLPDecoder(
    TimeVaryingDecoderBase
):
    """
    Multi-output MLP neural-network decoder.
    """

    def __init__(
        self,
        hidden_layer_sizes=(32, 16),
        max_iter=1000,
        random_state=42
    ):
        base_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state
        )

        self.model = MultiOutputClassifier(
            base_model
        )

    def train(self, X, y):
        self.model.fit(
            X,
            y
        )

        return self

    def predict(self, X):
        predictions = (
            self.model.predict(X)
        )

        return predictions.tolist()

    def predict_proba(self, X):
        return self.model.predict_proba(
            X
        )

    def decode(self, features):
        return super().decode(
            features
        )