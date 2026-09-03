from collections import Counter

from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.ml_dataset import MLDatasetBuilder


def test_ml_dataset():

    print("\n===================================")
    print(" ML DATASET BUILDER TEST")
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
    # 3. Build ML dataset
    # --------------------------------

    builder = MLDatasetBuilder(
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    ml_dataset = builder.build(
        dataset
    )

    print(
        "ML dataset construction: PASS"
    )

    # --------------------------------
    # 4. Print report
    # --------------------------------

    builder.print_report(
        ml_dataset
    )

    # --------------------------------
    # 5. Validate dataset sizes
    # --------------------------------

    assert (
        len(ml_dataset.X_train)
        == 800
    )

    assert (
        len(ml_dataset.y_train)
        == 800
    )

    assert (
        len(ml_dataset.X_validation)
        == 100
    )

    assert (
        len(ml_dataset.y_validation)
        == 100
    )

    assert (
        len(ml_dataset.X_test)
        == 100
    )

    assert (
        len(ml_dataset.y_test)
        == 100
    )

    print(
        "\nML dataset sizes: PASS"
    )

    # --------------------------------
    # 6. Validate feature dimensions
    # --------------------------------

    for features in (
        ml_dataset.X_train
        + ml_dataset.X_validation
        + ml_dataset.X_test
    ):

        assert len(features) == 2

        assert features in (
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        )

    print(
        "Feature dimensions: PASS"
    )

    # --------------------------------
    # 7. Validate target classes
    # --------------------------------

    all_targets = (
        ml_dataset.y_train
        + ml_dataset.y_validation
        + ml_dataset.y_test
    )

    assert all(
        target in (
            0,
            1,
            2,
            3
        )
        for target in all_targets
    )

    print(
        "Target classes: PASS"
    )

    # --------------------------------
    # 8. Check all classes
    # --------------------------------

    train_classes = set(
        ml_dataset.y_train
    )

    validation_classes = set(
        ml_dataset.y_validation
    )

    test_classes = set(
        ml_dataset.y_test
    )

    expected_classes = {
        0,
        1,
        2,
        3
    }

    assert (
        train_classes
        == expected_classes
    )

    assert (
        validation_classes
        == expected_classes
    )

    assert (
        test_classes
        == expected_classes
    )

    print(
        "Training classes: PASS"
    )

    print(
        "Validation classes: PASS"
    )

    print(
        "Test classes: PASS"
    )

    # --------------------------------
    # 9. Display distributions
    # --------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        "Training   : "
        f"{dict(Counter(ml_dataset.y_train))}"
    )

    print(
        "Validation : "
        f"{dict(Counter(ml_dataset.y_validation))}"
    )

    print(
        "Test       : "
        f"{dict(Counter(ml_dataset.y_test))}"
    )

    # --------------------------------
    # 10. Preview
    # --------------------------------

    print(
        "\nML sample preview:"
    )

    for index in range(5):

        print(
            f"  X = "
            f"{ml_dataset.X_train[index]}"
            f"   y = "
            f"{ml_dataset.y_train[index]}"
        )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n===================================")
    print(" ML DATASET BUILDER RESULT")
    print("===================================")

    print(
        "Training samples   : 800"
    )

    print(
        "Validation samples : 100"
    )

    print(
        "Test samples       : 100"
    )

    print(
        "Input features     : 2"
    )

    print(
        "Target classes     : 4"
    )

    print(
        "ML dataset         : READY"
    )

    print(
        "RESULT             : SUCCESS"
    )


if __name__ == "__main__":
    test_ml_dataset()