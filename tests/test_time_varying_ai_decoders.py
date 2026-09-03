from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from dataset.time_varying_ml_dataset import (
    TimeVaryingMLDatasetBuilder
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)

from evaluation.time_varying_metrics import (
    TimeVaryingDecoderMetrics
)


def evaluate_decoder(
    decoder,
    ml_dataset
):
    """
    Train and evaluate one time-varying decoder.
    """

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    predictions = decoder.predict(
        ml_dataset.X_test
    )

    metrics = TimeVaryingDecoderMetrics()

    return metrics.calculate(
        ml_dataset.y_test,
        predictions
    )


def test_time_varying_ai_decoders():

    print("\n===================================")
    print(" TIME-VARYING AI DECODER TEST")
    print("===================================")

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    rounds = 5
    dataset_size = 5000

    physical_error_probability = 0.10
    measurement_noise_probability = 0.10

    random_state = 42

    print(
        f"\nRounds              : {rounds}"
    )

    print(
        f"Physical noise      : "
        f"{physical_error_probability * 100:.0f}%"
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
    # Generate dataset
    # -------------------------------------------------

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=(
                physical_error_probability
            ),
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            seed=random_state
        )
    )

    samples = generator.generate_dataset(
        num_samples=dataset_size
    )

    assert len(samples) == dataset_size

    print(
        "\nDataset generation: PASS"
    )

    # -------------------------------------------------
    # Build ML dataset
    # -------------------------------------------------

    builder = (
        TimeVaryingMLDatasetBuilder(
            test_size=0.10,
            validation_size=0.10,
            random_state=random_state
        )
    )

    ml_dataset = builder.build(
        samples
    )

    print(
        "ML dataset construction: PASS"
    )

    # -------------------------------------------------
    # Verify dimensions
    # -------------------------------------------------

    expected_features = rounds * 4

    assert len(
        ml_dataset.X_train[0]
    ) == expected_features

    for target in ml_dataset.y_train:

        assert len(target) == 3

    print(
        f"Features per sample: "
        f"{expected_features}"
    )

    print(
        "Multi-output target: 3 bits"
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    metrics = TimeVaryingDecoderMetrics()

    # -------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic = (
        TimeVaryingLogisticDecoder(
            random_state=random_state
        )
    )

    logistic_metrics = evaluate_decoder(
        logistic,
        ml_dataset
    )

    metrics.print_report(
        logistic_metrics,
        "Logistic Regression"
    )

    # -------------------------------------------------
    # Random Forest
    # -------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest = (
        TimeVaryingRandomForestDecoder(
            n_estimators=100,
            random_state=random_state
        )
    )

    random_forest_metrics = (
        evaluate_decoder(
            random_forest,
            ml_dataset
        )
    )

    metrics.print_report(
        random_forest_metrics,
        "Random Forest"
    )

    # -------------------------------------------------
    # MLP
    # -------------------------------------------------

    print(
        "\nTraining MLP..."
    )

    mlp = (
        TimeVaryingMLPDecoder(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=random_state
        )
    )

    mlp_metrics = evaluate_decoder(
        mlp,
        ml_dataset
    )

    metrics.print_report(
        mlp_metrics,
        "MLP Neural Network"
    )

    # -------------------------------------------------
    # Verify metric ranges
    # -------------------------------------------------

    all_metrics = [
        logistic_metrics,
        random_forest_metrics,
        mlp_metrics
    ]

    for result in all_metrics:

        assert (
            0.0
            <= result["bit_accuracy"]
            <= 1.0
        )

        assert (
            0.0
            <= result["bit_precision"]
            <= 1.0
        )

        assert (
            0.0
            <= result["bit_recall"]
            <= 1.0
        )

        assert (
            0.0
            <= result["bit_f1"]
            <= 1.0
        )

        assert (
            0.0
            <= result["exact_pattern_accuracy"]
            <= 1.0
        )

        assert (
            0.0
            <= result["hamming_loss"]
            <= 1.0
        )

    print(
        "\nMetric ranges: PASS"
    )

    # -------------------------------------------------
    # Single decoding example
    # -------------------------------------------------

    example_features = (
        ml_dataset.X_test[0]
    )

    example_target = (
        ml_dataset.y_test[0]
    )

    logistic_prediction = (
        logistic.decode(
            example_features
        )
    )

    print(
        "\n==================================="
    )

    print(
        " SINGLE DECODING EXAMPLE"
    )

    print(
        "==================================="
    )

    print(
        "\nFeatures:"
    )

    print(
        example_features
    )

    print(
        "\nActual error:"
    )

    print(
        example_target
    )

    print(
        "\nPredicted error:"
    )

    print(
        logistic_prediction
    )

    print(
        "\nActual error description:"
    )

    print(
        builder.decode_target(
            example_target
        )
    )

    print(
        "\nPredicted error description:"
    )

    print(
        builder.decode_target(
            logistic_prediction
        )
    )

    # -------------------------------------------------
    # Final
    # -------------------------------------------------

    print("\n===================================")
    print(" TIME-VARYING AI DECODER RESULT")
    print("===================================")

    print(
        "Logistic Regression : PASS"
    )

    print(
        "Random Forest       : PASS"
    )

    print(
        "MLP Neural Network  : PASS"
    )

    print(
        "Multi-output decode : PASS"
    )

    print(
        "RESULT              : SUCCESS"
    )


if __name__ == "__main__":
    test_time_varying_ai_decoders()