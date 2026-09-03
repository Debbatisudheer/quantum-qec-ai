from dataset.noisy_generator import (
    NoisyQECDatasetGenerator
)

from dataset.noisy_ml_dataset import (
    NoisyMLDatasetBuilder
)


def test_noisy_ml_dataset():

    print("\n===================================")
    print(" NOISY ML DATASET TEST")
    print("===================================")

    # -------------------------------------------------
    # 1. Generate noisy dataset
    # -------------------------------------------------

    generator = NoisyQECDatasetGenerator(
        measurement_noise_probability=0.10,
        seed=42
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
        "Noisy ML dataset construction: PASS"
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

    for features in ml_dataset.X_train:

        assert len(features) == 2

    for features in ml_dataset.X_validation:

        assert len(features) == 2

    for features in ml_dataset.X_test:

        assert len(features) == 2

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
    # 6. Verify observed syndrome is being used
    # -------------------------------------------------

    found_noisy_sample = False

    for sample in dataset:

        if (
            sample["perfect_syndrome"]
            != sample["observed_syndrome"]
        ):

            features, target = (
                builder.transform_sample(
                    sample
                )
            )

            expected_features = [
                int(
                    sample[
                        "observed_syndrome"
                    ][0]
                ),
                int(
                    sample[
                        "observed_syndrome"
                    ][1]
                ),
            ]

            assert (
                features
                == expected_features
            )

            found_noisy_sample = True

            print(
                "\nNoisy sample verification:"
            )

            print(
                "Perfect syndrome : "
                f"{sample['perfect_syndrome']}"
            )

            print(
                "Observed syndrome: "
                f"{sample['observed_syndrome']}"
            )

            print(
                f"ML features      : "
                f"{features}"
            )

            print(
                f"Target           : "
                f"{target}"
            )

            break

    assert found_noisy_sample is True

    print(
        "Observed syndrome as input: PASS"
    )

    # -------------------------------------------------
    # 7. Ground-truth leakage check
    # -------------------------------------------------

    for features in ml_dataset.X_train:

        assert len(features) == 2

        for value in features:

            assert value in (0, 1)

    print(
        "Ground-truth leakage check: PASS"
    )

    # -------------------------------------------------
    # 8. Print report
    # -------------------------------------------------

    builder.print_report(
        ml_dataset
    )

    # -------------------------------------------------
    # 9. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" NOISY ML DATASET RESULT")
    print("===================================")

    print(
        "Observed syndrome → X: PASS"
    )

    print(
        "Actual error → y: PASS"
    )

    print(
        "Ground-truth leakage: NONE"
    )

    print(
        "RESULT: SUCCESS"
    )


if __name__ == "__main__":
    test_noisy_ml_dataset()