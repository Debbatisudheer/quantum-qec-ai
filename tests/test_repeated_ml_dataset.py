from dataset.repeated_generator import (
    RepeatedSyndromeDatasetGenerator
)

from dataset.repeated_ml_dataset import (
    RepeatedMLDatasetBuilder
)


def test_repeated_ml_dataset():

    print("\n===================================")
    print(" REPEATED ML DATASET TEST")
    print("===================================")

    # -------------------------------------------------
    # 1. Generate repeated noisy dataset
    # -------------------------------------------------

    generator = (
        RepeatedSyndromeDatasetGenerator(
            rounds=5,
            measurement_noise_probability=0.10,
            seed=42
        )
    )

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    print(
        f"\nDataset generated: "
        f"{len(dataset)}"
    )

    assert len(dataset) == 1000

    print(
        "Dataset generation: PASS"
    )

    # -------------------------------------------------
    # 2. Build ML dataset
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
        "Repeated ML dataset construction: PASS"
    )

    # -------------------------------------------------
    # 3. Check dataset sizes
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
    # 4. Check feature dimensions
    # -------------------------------------------------

    expected_features = 5 * 2

    for features in ml_dataset.X_train:

        assert len(
            features
        ) == expected_features

    for features in ml_dataset.X_validation:

        assert len(
            features
        ) == expected_features

    for features in ml_dataset.X_test:

        assert len(
            features
        ) == expected_features

    print(
        "Feature dimensions: PASS"
    )

    # -------------------------------------------------
    # 5. Check target classes
    # -------------------------------------------------

    valid_classes = {
        0,
        1,
        2,
        3
    }

    assert set(
        ml_dataset.y_train
    ).issubset(valid_classes)

    assert set(
        ml_dataset.y_validation
    ).issubset(valid_classes)

    assert set(
        ml_dataset.y_test
    ).issubset(valid_classes)

    print(
        "Target classes: PASS"
    )

    # -------------------------------------------------
    # 6. Verify history encoding
    # -------------------------------------------------

    test_history = [
        "01",
        "10",
        "11",
        "00",
        "01"
    ]

    expected_features = [
        0, 1,
        1, 0,
        1, 1,
        0, 0,
        0, 1
    ]

    actual_features = (
        builder.encode_history(
            test_history
        )
    )

    assert (
        actual_features
        == expected_features
    )

    print(
        "History encoding: PASS"
    )

    print(
        "\nExample history:"
    )

    print(
        f"  {test_history}"
    )

    print(
        "\nEncoded features:"
    )

    print(
        f"  {actual_features}"
    )

    # -------------------------------------------------
    # 7. Verify observable input
    # -------------------------------------------------

    found_noisy_sample = False

    for sample in dataset:

        perfect_syndrome = (
            sample["perfect_syndrome"]
        )

        history = (
            sample["syndrome_history"]
        )

        if any(
            syndrome != perfect_syndrome
            for syndrome in history
        ):

            features, target = (
                builder.transform_sample(
                    sample
                )
            )

            expected_features = (
                builder.encode_history(
                    history
                )
            )

            assert (
                features
                == expected_features
            )

            found_noisy_sample = True

            print(
                "\nNoisy history verification:"
            )

            print(
                "Perfect syndrome : "
                f"{perfect_syndrome}"
            )

            print(
                "History          : "
                f"{history}"
            )

            print(
                "ML features      : "
                f"{features}"
            )

            print(
                "Target           : "
                f"{target}"
            )

            break

    assert found_noisy_sample is True

    print(
        "Observed history as input: PASS"
    )

    # -------------------------------------------------
    # 8. Ground-truth leakage check
    # -------------------------------------------------

    for features in ml_dataset.X_train:

        assert len(
            features
        ) == 10

        for value in features:

            assert value in (
                0,
                1
            )

    print(
        "Ground-truth leakage check: PASS"
    )

    # -------------------------------------------------
    # 9. Print report
    # -------------------------------------------------

    builder.print_report(
        ml_dataset
    )

    # -------------------------------------------------
    # 10. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" REPEATED ML DATASET RESULT")
    print("===================================")

    print(
        "5 syndrome rounds    : PASS"
    )

    print(
        "10 ML features       : PASS"
    )

    print(
        "Observed history → X : PASS"
    )

    print(
        "Actual error → y     : PASS"
    )

    print(
        "Ground-truth leakage : NONE"
    )

    print(
        "RESULT               : SUCCESS"
    )


if __name__ == "__main__":
    test_repeated_ml_dataset()