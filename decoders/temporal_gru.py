import torch
import torch.nn as nn


class GRUNetwork(nn.Module):
    """
    Small GRU network for repeated QEC decoding.

    Input shape:

        batch × rounds × features_per_round

    Example:

        100 × 5 × 2

    Output:

        batch × 3

    The three outputs represent:

        q0
        q1
        q2
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=32,
        num_layers=1,
        output_size=3
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

        # Use the final round's hidden representation.
        final_output = sequence_output[:, -1, :]

        return self.output(
            final_output
        )


class TemporalGRUDecoder:
    """
    Temporal GRU decoder for repeated
    syndrome history.

    Expected input:

        [
            [r1_bit1, r1_bit2],
            [r2_bit1, r2_bit2],
            ...
            [r5_bit1, r5_bit2]
        ]

    Target:

        [q0, q1, q2]
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=32,
        num_layers=1,
        learning_rate=0.001,
        epochs=30,
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

        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3
        )

    def train(
        self,
        X,
        y
    ):

        torch.manual_seed(
            self.random_seed
        )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.float32
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        loss_function = (
            nn.BCEWithLogitsLoss()
        )

        self.model.train()

        for epoch in range(
            self.epochs
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

        return self

    def predict(self, X):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.model.eval()

        with torch.no_grad():

            logits = self.model(
                X_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).int()

        return predictions.tolist()

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

            probabilities = torch.sigmoid(
                logits
            )

        return probabilities.tolist()

    def decode(self, features):

        prediction = self.predict(
            [features]
        )

        return prediction[0]