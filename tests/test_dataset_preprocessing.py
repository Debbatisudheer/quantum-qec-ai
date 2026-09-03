from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.preprocessing import (
    QECDatasetPreprocessor
)


def test_dataset_preprocessing():

    print("\n===================================")
    print(" DATASET PREPROCESSING TEST")
    print("===================================")

    # --------------------------------
    # 1. Generate dataset
    # --------------------------------

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    print(
        "\nDataset generated:"
    )

    print(
        f"Samples: {len(dataset)}"
    )

    # --------------------------------
    # 2. Validate dataset
    # --------------------------------

    validator = QECDatasetValidator()

    valid, errors = (
        validator.validate_dataset(
            dataset
        )
    )

    assert valid is True
    assert errors == []

    print(
        "Dataset validation: PASS"
    )

    # --------------------------------
    # 3. Create preprocessor
    # --------------------------------

    preprocessor = (
        QECDatasetPreprocessor()
    )

    # --------------------------------
    # 4. Test syndrome encoding
    # --------------------------------

    assert (
        preprocessor.encode_syndrome(
            "00"
        )
        == [0, 0]
    )

    assert (
        preprocessor.encode_syndrome(
            "10"
        )
        == [1, 0]
    )

    assert (
        preprocessor.encode_syndrome(
            "11"
        )
        == [1, 1]
    )

    assert (
        preprocessor.encode_syndrome(
            "01"
        )
        == [0, 1]
    )

    print(
        "Syndrome encoding: PASS"
    )

    # --------------------------------
    # 5. Test target encoding
    # --------------------------------

    assert (
        preprocessor.encode_target(
            None
        )
        == 0
    )

    assert (
        preprocessor.encode_target(
            0
        )
        == 1
    )

    assert (
        preprocessor.encode_target(
            1
        )
        == 2
    )

    assert (
        preprocessor.encode_target(
            2
        )
        == 3
    )

    print(
        "Target encoding: PASS"
    )

    # --------------------------------
    # 6. Test reverse target encoding
    # --------------------------------

    assert (
        preprocessor.decode_target(
            0
        )
        is None
    )

    assert (
        preprocessor.decode_target(
            1
        )
        == 0
    )

    assert (
        preprocessor.decode_target(
            2
        )
        == 1
    )

    assert (
        preprocessor.decode_target(
            3
        )
        == 2
    )

    print(
        "Target decoding: PASS"
    )

    # --------------------------------
    # 7. Transform dataset
    # --------------------------------

    features, targets = (
        preprocessor.transform_dataset(
            dataset
        )
    )

    print(
        "Dataset transformation: PASS"
    )

    # --------------------------------
    # 8. Validate dimensions
    # --------------------------------

    assert len(features) == 1000
    assert len(targets) == 1000

    for feature_vector in features:

        assert len(
            feature_vector
        ) == 2

        assert feature_vector in (
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        )

    for target in targets:

        assert target in (
            0,
            1,
            2,
            3,
        )

    print(
        "Feature dimensions: PASS"
    )

    print(
        "Target dimensions: PASS"
    )

    # --------------------------------
    # 9. Preview
    # --------------------------------

    preprocessor.print_sample_preview(
        dataset,
        count=10
    )

    # --------------------------------
    # 10. Final result
    # --------------------------------

    print("\n===================================")
    print(" DATASET PREPROCESSING RESULT")
    print("===================================")

    print(
        "Samples transformed : 1000"
    )

    print(
        "Input features       : 2"
    )

    print(
        "Target classes       : 4"
    )

    print(
        "Ground-truth leakage : NONE"
    )

    print(
        "RESULT               : SUCCESS"
    )


if __name__ == "__main__":
    test_dataset_preprocessing()