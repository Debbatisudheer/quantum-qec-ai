from dataset.repeated_generator import (
    RepeatedSyndromeDatasetGenerator
)

from dataset.repeated_ml_dataset import (
    RepeatedMLDatasetBuilder
)

from decoders.ml_logistic import (
    LogisticRegressionDecoder
)

from decoders.ml_random_forest import (
    RandomForestDecoder
)

from decoders.ml_mlp import (
    MLPDecoder
)

from evaluation.decoder_metrics import (
    DecoderMetrics
)


class RoundSweepExperiment:
    """
    Measure AI decoder performance as the number
    of repeated syndrome rounds changes.

    Example:

        1 round  -> 2 features
        2 rounds -> 4 features
        3 rounds -> 6 features
        5 rounds -> 10 features
        7 rounds -> 14 features
        10 rounds -> 20 features

    The physical error model and measurement-noise
    probability remain fixed.

    Only the number of syndrome observations changes.
    """

    def __init__(
        self,
        rounds_list=None,
        measurement_noise_probability=0.10,
        dataset_size=5000,
        random_state=42
    ):
        if rounds_list is None:
            rounds_list = [
                1,
                2,
                3,
                5,
                7,
                10
            ]

        if len(rounds_list) == 0:
            raise ValueError(
                "rounds_list cannot be empty"
            )

        if any(
            rounds <= 0
            for rounds in rounds_list
        ):
            raise ValueError(
                "All rounds must be greater than 0"
            )

        if not 0.0 <= (
            measurement_noise_probability
        ) <= 1.0:
            raise ValueError(
                "measurement_noise_probability "
                "must be between 0 and 1"
            )

        if dataset_size <= 0:
            raise ValueError(
                "dataset_size must be greater than 0"
            )

        self.rounds_list = rounds_list

        self.measurement_noise_probability = (
            measurement_noise_probability
        )

        self.dataset_size = dataset_size

        self.random_state = random_state

    def generate_dataset(
        self,
        rounds
    ):
        """
        Generate a repeated-syndrome dataset
        for a specific number of rounds.
        """

        generator = (
            RepeatedSyndromeDatasetGenerator(
                rounds=rounds,
                measurement_noise_probability=(
                    self.measurement_noise_probability
                ),
                seed=self.random_state
            )
        )

        return generator.generate_dataset(
            num_samples=self.dataset_size
        )

    def build_ml_dataset(
        self,
        dataset
    ):
        """
        Convert generated samples into
        ML-ready train/validation/test data.
        """

        builder = RepeatedMLDatasetBuilder(
            test_size=0.10,
            validation_size=0.10,
            random_state=self.random_state
        )

        return builder.build(
            dataset
        )

    def evaluate_decoder(
        self,
        decoder,
        ml_dataset
    ):
        """
        Train and evaluate one decoder.
        """

        decoder.train(
            ml_dataset.X_train,
            ml_dataset.y_train
        )

        predictions = decoder.predict(
            ml_dataset.X_test
        )

        metrics = DecoderMetrics()

        return metrics.calculate(
            ml_dataset.y_test,
            predictions
        )

    def run_round(
        self,
        rounds
    ):
        """
        Run one round-count experiment.
        """

        print("\n-----------------------------------")

        print(
            f" ROUNDS = {rounds}"
        )

        print("-----------------------------------")

        # -------------------------------------------------
        # Generate dataset
        # -------------------------------------------------

        dataset = self.generate_dataset(
            rounds
        )

        print(
            f"Dataset generated: "
            f"{len(dataset)}"
        )

        # -------------------------------------------------
        # Build ML dataset
        # -------------------------------------------------

        ml_dataset = self.build_ml_dataset(
            dataset
        )

        expected_features = (
            rounds * 2
        )

        actual_features = len(
            ml_dataset.X_train[0]
        )

        assert (
            actual_features
            == expected_features
        )

        print(
            f"Features per sample: "
            f"{actual_features}"
        )

        # -------------------------------------------------
        # Logistic Regression
        # -------------------------------------------------

        logistic_decoder = (
            LogisticRegressionDecoder()
        )

        logistic_metrics = (
            self.evaluate_decoder(
                logistic_decoder,
                ml_dataset
            )
        )

        # -------------------------------------------------
        # Random Forest
        # -------------------------------------------------

        random_forest_decoder = (
            RandomForestDecoder(
                n_estimators=100,
                random_state=self.random_state
            )
        )

        random_forest_metrics = (
            self.evaluate_decoder(
                random_forest_decoder,
                ml_dataset
            )
        )

        # -------------------------------------------------
        # MLP
        # -------------------------------------------------

        mlp_decoder = MLPDecoder(
            hidden_layer_sizes=(16, 16),
            max_iter=1000,
            random_state=self.random_state
        )

        mlp_metrics = (
            self.evaluate_decoder(
                mlp_decoder,
                ml_dataset
            )
        )

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print(
            "\nAccuracy:"
        )

        print(
            "  Logistic Regression : "
            f"{logistic_metrics['accuracy']:.4f}"
        )

        print(
            "  Random Forest       : "
            f"{random_forest_metrics['accuracy']:.4f}"
        )

        print(
            "  MLP                 : "
            f"{mlp_metrics['accuracy']:.4f}"
        )

        return {
            "rounds": rounds,
            "features": actual_features,

            "logistic_accuracy":
                logistic_metrics[
                    "accuracy"
                ],

            "logistic_f1":
                logistic_metrics[
                    "f1"
                ],

            "random_forest_accuracy":
                random_forest_metrics[
                    "accuracy"
                ],

            "random_forest_f1":
                random_forest_metrics[
                    "f1"
                ],

            "mlp_accuracy":
                mlp_metrics[
                    "accuracy"
                ],

            "mlp_f1":
                mlp_metrics[
                    "f1"
                ],
        }

    def run(self):
        """
        Run the complete round sweep.
        """

        print("\n===================================")
        print(" QEC ROUND SWEEP EXPERIMENT")
        print("===================================")

        print(
            f"\nMeasurement noise : "
            f"{self.measurement_noise_probability * 100:.0f}%"
        )

        print(
            f"Dataset size      : "
            f"{self.dataset_size}"
        )

        print(
            f"Rounds tested     : "
            f"{self.rounds_list}"
        )

        results = []

        for rounds in self.rounds_list:

            result = self.run_round(
                rounds
            )

            results.append(
                result
            )

        return results

    def print_summary(
        self,
        results
    ):
        """
        Print a compact summary of all
        round-sweep results.
        """

        print("\n===================================")
        print(" ROUND SWEEP SUMMARY")
        print("===================================")

        print(
            "\nRounds | Features | "
            "Logistic | Random Forest | MLP"
        )

        print(
            "-----------------------------------"
        )

        for result in results:

            print(
                f"{result['rounds']:>6} | "
                f"{result['features']:>8} | "
                f"{result['logistic_accuracy']:.4f}   | "
                f"{result['random_forest_accuracy']:.4f}         | "
                f"{result['mlp_accuracy']:.4f}"
            )

        print(
            "\n==================================="
        )

        print(
            " ROUND SWEEP COMPLETE"
        )

        print(
            "==================================="
        )


if __name__ == "__main__":

    experiment = RoundSweepExperiment(
        rounds_list=[
            1,
            2,
            3,
            5,
            7,
            10
        ],
        measurement_noise_probability=0.10,
        dataset_size=5000,
        random_state=42
    )

    results = experiment.run()

    experiment.print_summary(
        results
    )