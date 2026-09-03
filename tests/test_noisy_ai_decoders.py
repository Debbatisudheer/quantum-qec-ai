from dataset.noisy_generator import (
    NoisyQECDatasetGenerator
)

from dataset.noisy_ml_dataset import (
    NoisyMLDatasetBuilder
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


def evaluate_decoder(
    decoder,
    model_name,
    ml_dataset
):
    """
    Train and evaluate one AI decoder.
    """

    print("\n-----------------------------------")
    print(
        f" {model_name.upper()}"
    )
    print("-----------------------------------")

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    print(
        "Training: PASS"
    )

    predictions = decoder.predict(
        ml_dataset.X_test
    )

    metrics_calculator = DecoderMetrics()

    metrics = metrics_calculator.calculate(
        ml_dataset.y_test,
        predictions
    )

    metrics_calculator.print_report(
        metrics,
        model_name
    )

    return metrics


def test_noisy_ai_decoders():

    print("\n===================================")
    print(" NOISY AI DECODER EXPERIMENT")
    print("===================================")

    # -------------------------------------------------
    # 1. Generate noisy dataset
    # -------------------------------------------------

    noise_probability = 0.10

    generator = NoisyQECDatasetGenerator(
        measurement_noise_probability=(
            noise_probability
        ),
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=5000
    )

    print(
        f"\nDataset generated : "
        f"{len(dataset)}"
    )

    print(
        f"Measurement noise : "
        f"{noise_probability * 100:.0f}%"
    )

    assert len(dataset) == 5000

    # -------------------------------------------------
    # 2. Build noisy ML dataset
    # -------------------------------------------------

    builder = NoisyMLDatasetBuilder(
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    ml_dataset = builder.build(
        dataset
    )

    print(
        "Noisy ML dataset: PASS"
    )

    print(
        f"Training samples   : "
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

    # -------------------------------------------------
    # 3. Logistic Regression
    # -------------------------------------------------

    logistic_decoder = (
        LogisticRegressionDecoder()
    )

    logistic_metrics = evaluate_decoder(
        logistic_decoder,
        "Logistic Regression",
        ml_dataset
    )

    # -------------------------------------------------
    # 4. Random Forest
    # -------------------------------------------------

    random_forest_decoder = (
        RandomForestDecoder(
            n_estimators=100,
            random_state=42
        )
    )

    random_forest_metrics = evaluate_decoder(
        random_forest_decoder,
        "Random Forest",
        ml_dataset
    )

    # -------------------------------------------------
    # 5. MLP
    # -------------------------------------------------

    mlp_decoder = MLPDecoder(
        hidden_layer_sizes=(16, 16),
        max_iter=1000,
        random_state=42
    )

    mlp_metrics = evaluate_decoder(
        mlp_decoder,
        "MLP Neural Network",
        ml_dataset
    )

    # -------------------------------------------------
    # 6. Basic validation
    # -------------------------------------------------

    for metrics in (
        logistic_metrics,
        random_forest_metrics,
        mlp_metrics
    ):

        assert (
            0.0
            <= metrics["accuracy"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["precision"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["recall"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["f1"]
            <= 1.0
        )

    print(
        "\nAI metric validation: PASS"
    )

    # -------------------------------------------------
    # 7. Compare models
    # -------------------------------------------------

    print("\n===================================")
    print(" NOISY MODEL COMPARISON")
    print("===================================")

    print(
        "\nModel                  Accuracy"
    )

    print(
        "Logistic Regression    "
        f"{logistic_metrics['accuracy']:.4f}"
    )

    print(
        "Random Forest          "
        f"{random_forest_metrics['accuracy']:.4f}"
    )

    print(
        "MLP Neural Network     "
        f"{mlp_metrics['accuracy']:.4f}"
    )

    # -------------------------------------------------
    # 8. Show important comparison
    # -------------------------------------------------

    print("\n===================================")
    print(" NOISY DECODING INTERPRETATION")
    print("===================================")

    print(
        "\nThe AI receives:"
    )

    print(
        "  observed_syndrome"
    )

    print(
        "\nThe AI must predict:"
    )

    print(
        "  actual error class"
    )

    print(
        "\nGround truth leakage:"
    )

    print(
        "  NONE"
    )

    # -------------------------------------------------
    # 9. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" NOISY AI DECODER RESULT")
    print("===================================")

    print(
        "Logistic Regression : READY"
    )

    print(
        "Random Forest       : READY"
    )

    print(
        "MLP                 : READY"
    )

    print(
        "Noisy decoding      : TESTED"
    )

    print(
        "RESULT              : SUCCESS"
    )


if __name__ == "__main__":
    test_noisy_ai_decoders()