import torch
import torch.nn as nn


ERROR_PATTERNS = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]


PATTERN_TO_CLASS = {
    pattern: index
    for index, pattern in enumerate(
        ERROR_PATTERNS
    )
}


class GRUClassifierNetwork(nn.Module):
    """
    Temporal GRU classifier.

    Input:

        batch × rounds × 2

    Example:

        batch × 5 × 2

    Output:

        batch × 8

    The 8 outputs correspond to the
    eight possible 3-qubit error patterns.
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=32,
        num_layers=1,
        output_size=8
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        sequence_output, _ = self.gru(x)

        final_output = (
            sequence_output[:, -1, :]
        )

        return self.output(
            final_output
        )


class TemporalGRUClassifier:
    """
    8-class temporal GRU decoder.

    Target mapping:

        class 0 -> 000
        class 1 -> 001
        class 2 -> 010
        class 3 -> 011
        class 4 -> 100
        class 5 -> 101
        class 6 -> 110
        class 7 -> 111
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=32,
        num_layers=1,
        learning_rate=0.001,
        epochs=50,
        random_seed=42
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed

        torch.manual_seed(
            random_seed
        )

        self.model = (
            GRUClassifierNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=8
            )
        )

    @staticmethod
    def pattern_to_class(pattern):

        pattern = tuple(
            int(bit)
            for bit in pattern
        )

        if pattern not in PATTERN_TO_CLASS:

            raise ValueError(
                f"Unknown error pattern: "
                f"{pattern}"
            )

        return PATTERN_TO_CLASS[
            pattern
        ]

    @staticmethod
    def class_to_pattern(class_index):

        if not 0 <= class_index < 8:

            raise ValueError(
                "class_index must be "
                "between 0 and 7"
            )

        return list(
            ERROR_PATTERNS[
                class_index
            ]
        )

    def train(
        self,
        X,
        y,
        verbose=True
    ):

        torch.manual_seed(
            self.random_seed
        )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        y_classes = [
            self.pattern_to_class(
                target
            )
            for target in y
        ]

        y_tensor = torch.tensor(
            y_classes,
            dtype=torch.long
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        loss_function = (
            nn.CrossEntropyLoss()
        )

        self.model.train()

        for epoch in range(
            1,
            self.epochs + 1
        ):

            optimizer.zero_grad()

            logits = self.model(
                X_tensor
            )

            loss = loss_function(
                logits,
                y_tensor
            )

            loss.backward()

            optimizer.step()

            if verbose and (
                epoch == 1
                or epoch % 5 == 0
                or epoch == self.epochs
            ):

                predictions = (
                    torch.argmax(
                        logits,
                        dim=1
                    )
                )

                accuracy = (
                    (
                        predictions
                        == y_tensor
                    )
                    .float()
                    .mean()
                    .item()
                )

                print(
                    f"Epoch {epoch:3d}/"
                    f"{self.epochs} "
                    f"Loss={loss.item():.6f} "
                    f"Train exact="
                    f"{accuracy:.4f}"
                )

        return self

    def predict_classes(self, X):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                X_tensor
            )

            classes = torch.argmax(
                logits,
                dim=1
            )

        return classes.tolist()

    def predict(self, X):

        classes = (
            self.predict_classes(X)
        )

        return [
            self.class_to_pattern(
                class_index
            )
            for class_index in classes
        ]

    def predict_proba(self, X):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                X_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        return probabilities.tolist()

    def decode(self, features):

        prediction = self.predict(
            [features]
        )

        return prediction[0]