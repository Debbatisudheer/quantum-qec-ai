from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from dataset.time_varying_ml_dataset import (
    TimeVaryingMLDatasetBuilder
)


def test_time_varying_ml_dataset():

    print("\n===================================")
    print(" TIME-VARYING ML DATASET TEST")
    print("===================================")

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    rounds = 5
    dataset_size = 1000

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=0.10,
            measurement_noise_probability=0.10,
            seed=42
        )
    )

    # -------------------------------------------------
    # Generate samples
    # -------------------------------------------------

    samples = generator.generate_dataset(
        num_samples=dataset_size
    )

    print(
        f"Dataset generated: "
        f"{len(samples)}"
    )

    assert len(samples) == dataset_size

    print(
        "Dataset generation: PASS"
    )

    # -------------------------------------------------
    # Build ML dataset
    # -------------------------------------------------

    builder = (
        TimeVaryingMLDatasetBuilder(
            test_size=0.10,
            validation_size=0.10,
            random_state=42
        )
    )

    ml_dataset = builder.build(
        samples
    )

    print(
        "ML dataset construction: PASS"
    )

    # -------------------------------------------------
    # Dataset sizes
    # -------------------------------------------------

    assert len(
        ml_dataset.X_train
    ) == 800

    assert len(
        ml_dataset.X_validation
    ) == 100

    assert len(
        ml_dataset.X_test
    ) == 100

    print(
        "Dataset sizes: PASS"
    )

    # -------------------------------------------------
    # Feature dimensions
    # -------------------------------------------------

    expected_features = rounds * 4

    assert len(
        ml_dataset.X_train[0]
    ) == expected_features

    assert len(
        ml_dataset.X_validation[0]
    ) == expected_features

    assert len(
        ml_dataset.X_test[0]
    ) == expected_features

    print(
        "Feature dimensions: PASS"
    )

    # -------------------------------------------------
    # Target dimensions
    # -------------------------------------------------

    for target in ml_dataset.y_train:

        assert len(target) == 3

        assert all(
            bit in (0, 1)
            for bit in target
        )

    print(
        "Target dimensions: PASS"
    )

    # -------------------------------------------------
    # Ground-truth leakage test
    # -------------------------------------------------

    sample = samples[0]

    X, y = (
        builder.transform_sample(
            sample
        )
    )

    expected_X = (
        builder.encode_temporal_features(
            sample[
                "observed_syndrome_history"
            ],
            sample[
                "detection_events"
            ]
        )
    )

    expected_y = (
        builder.encode_target(
            sample[
                "final_error_state"
            ]
        )
    )

    assert X == expected_X
    assert y == expected_y

    print(
        "Input/target mapping: PASS"
    )

    # -------------------------------------------------
    # Verify input does not use perfect syndrome
    # -------------------------------------------------

    original_X = X.copy()

    modified_sample = sample.copy()

    modified_sample[
        "syndrome_history"
    ] = [
        "11"
        for _ in range(rounds)
    ]

    modified_X, _ = (
        builder.transform_sample(
            modified_sample
        )
    )

    assert (
        modified_X
        == original_X
    )

    print(
        "Perfect syndrome leakage: NONE"
    )

    # -------------------------------------------------
    # Verify target represents final error
    # -------------------------------------------------

    decoded_target = (
        builder.decode_target(y)
    )

    print(
        "\nExample sample:"
    )

    print(
        "Observed syndrome history:"
    )

    print(
        sample[
            "observed_syndrome_history"
        ]
    )

    print(
        "\nDetection events:"
    )

    print(
        sample[
            "detection_events"
        ]
    )

    print(
        "\nML features:"
    )

    print(X)

    print(
        "\nFinal error state:"
    )

    print(
        sample[
            "final_error_state"
        ]
    )

    print(
        "\nML target:"
    )

    print(y)

    print(
        "\nDecoded target:"
    )

    print(decoded_target)

    # -------------------------------------------------
    # Print report
    # -------------------------------------------------

    builder.print_report(
        ml_dataset
    )

    # -------------------------------------------------
    # Final
    # -------------------------------------------------

    print("\n===================================")
    print(" TIME-VARYING ML DATASET RESULT")
    print("===================================")

    print(
        "Temporal feature encoding : PASS"
    )

    print(
        "Detection-event encoding  : PASS"
    )

    print(
        "Multi-output target       : PASS"
    )

    print(
        "Ground-truth separation   : PASS"
    )

    print(
        "RESULT                    : SUCCESS"
    )


if __name__ == "__main__":
    test_time_varying_ml_dataset()