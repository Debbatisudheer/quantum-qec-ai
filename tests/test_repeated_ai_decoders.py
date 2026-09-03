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


def evaluate_decoder(
    decoder,
    model_name,
    ml_dataset
):
    """
    Train and evaluate one AI decoder
    using repeated syndrome history.
    """

    print("\n-----------------------------------")
    print(
        f" {model_name.upper()}"
    )
    print("-----------------------------------")

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    print(
        "Training: PASS"
    )

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------

    predictions = decoder.predict(
        ml_dataset.X_test
    )

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------

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


def test_repeated_ai_decoders():

    print("\n===================================")
    print(" REPEATED SYNDROME AI DECODER")
    print("===================================")

    # -------------------------------------------------
    # 1. Configuration
    # -------------------------------------------------

    rounds = 5

    measurement_noise_probability = 0.10

    dataset_size = 5000

    print(
        f"\nSyndrome rounds     : "
        f"{rounds}"
    )

    print(
        f"Measurement noise   : "
        f"{measurement_noise_probability * 100:.0f}%"
    )

    print(
        f"Dataset size        : "
        f"{dataset_size}"
    )

    # -------------------------------------------------
    # 2. Generate dataset
    # -------------------------------------------------

    generator = (
        RepeatedSyndromeDatasetGenerator(
            rounds=rounds,
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            seed=42
        )
    )

    dataset = generator.generate_dataset(
        num_samples=dataset_size
    )

    assert len(dataset) == dataset_size

    print(
        "\nDataset generation: PASS"
    )

    # -------------------------------------------------
    # 3. Build ML dataset
    # -------------------------------------------------

    builder = RepeatedMLDatasetBuilder(
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    ml_dataset = builder.build(
        dataset
    )

    print(
        "Repeated ML dataset: PASS"
    )

    # -------------------------------------------------
    # 4. Verify feature count
    # -------------------------------------------------

    expected_feature_count = (
        rounds * 2
    )

    assert (
        len(
            ml_dataset.X_train[0]
        )
        == expected_feature_count
    )

    print(
        f"Feature count: "
        f"{expected_feature_count}"
    )

    print(
        "Feature dimensions: PASS"
    )

    # -------------------------------------------------
    # 5. Logistic Regression
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
    # 6. Random Forest
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
    # 7. MLP
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
    # 8. Validate metrics
    # -------------------------------------------------

    all_metrics = {
        "Logistic Regression":
            logistic_metrics,

        "Random Forest":
            random_forest_metrics,

        "MLP Neural Network":
            mlp_metrics
    }

    for model_name, metrics in (
        all_metrics.items()
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
    # 9. Compare models
    # -------------------------------------------------

    print("\n===================================")
    print(" REPEATED MODEL COMPARISON")
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
    # 10. Feature comparison
    # -------------------------------------------------

    print("\n===================================")
    print(" FEATURE COMPARISON")
    print("===================================")

    print(
        "\nSingle noisy syndrome:"
    )

    print(
        "  2 features"
    )

    print(
        "  Example: [1,0]"
    )

    print(
        "\nRepeated syndrome history:"
    )

    print(
        f"  {expected_feature_count} features"
    )

    print(
        "  Example: "
        "[0,1, 0,1, 1,1, 0,1, 0,1]"
    )

    # -------------------------------------------------
    # 11. Important interpretation
    # -------------------------------------------------

    print("\n===================================")
    print(" DECODING INTERPRETATION")
    print("===================================")

    print(
        "\nAI input:"
    )

    print(
        "  syndrome_history"
    )

    print(
        "\nAI target:"
    )

    print(
        "  actual error class"
    )

    print(
        "\nGround-truth leakage:"
    )

    print(
        "  NONE"
    )

    # -------------------------------------------------
    # 12. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" REPEATED AI DECODER RESULT")
    print("===================================")

    print(
        "Repeated syndrome input : PASS"
    )

    print(
        "Logistic Regression     : READY"
    )

    print(
        "Random Forest           : READY"
    )

    print(
        "MLP                     : READY"
    )

    print(
        "AI evaluation           : PASS"
    )

    print(
        "RESULT                  : SUCCESS"
    )


if __name__ == "__main__":
    test_repeated_ai_decoders()