from collections import Counter

from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.splitter import QECDatasetSplitter


def get_target(sample):

    if sample.error_qubit is None:
        return 0

    return sample.error_qubit + 1


def test_dataset_splitter():

    print("\n===================================")
    print(" DATASET SPLITTER TEST")
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
    # 3. Split dataset
    # --------------------------------

    splitter = QECDatasetSplitter(
        test_size=0.10,
        validation_size=0.10,
        random_state=42
    )

    train_dataset, \
        validation_dataset, \
        test_dataset = splitter.split(
            dataset
        )

    # --------------------------------
    # 4. Print report
    # --------------------------------

    splitter.print_report(
        train_dataset,
        validation_dataset,
        test_dataset
    )

    # --------------------------------
    # 5. Validate sizes
    # --------------------------------

    assert len(
        train_dataset
    ) == 800

    assert len(
        validation_dataset
    ) == 100

    assert len(
        test_dataset
    ) == 100

    print(
        "\nDataset sizes: PASS"
    )

    # --------------------------------
    # 6. Validate total
    # --------------------------------

    assert (
        len(train_dataset)
        + len(validation_dataset)
        + len(test_dataset)
        == 1000
    )

    print(
        "Total sample count: PASS"
    )

    # --------------------------------
    # 7. Check for duplicate samples
    # --------------------------------

    train_ids = {
        sample.sample_id
        for sample in train_dataset
    }

    validation_ids = {
        sample.sample_id
        for sample in validation_dataset
    }

    test_ids = {
        sample.sample_id
        for sample in test_dataset
    }

    assert (
        train_ids.isdisjoint(
            validation_ids
        )
    )

    assert (
        train_ids.isdisjoint(
            test_ids
        )
    )

    assert (
        validation_ids.isdisjoint(
            test_ids
        )
    )

    print(
        "Dataset separation: PASS"
    )

    # --------------------------------
    # 8. Check all classes
    # --------------------------------

    train_classes = set(
        get_target(sample)
        for sample in train_dataset
    )

    validation_classes = set(
        get_target(sample)
        for sample in validation_dataset
    )

    test_classes = set(
        get_target(sample)
        for sample in test_dataset
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
        "All classes in training set: PASS"
    )

    print(
        "All classes in validation set: PASS"
    )

    print(
        "All classes in test set: PASS"
    )

    # --------------------------------
    # 9. Print class distributions
    # --------------------------------

    train_distribution = Counter(
        get_target(sample)
        for sample in train_dataset
    )

    validation_distribution = Counter(
        get_target(sample)
        for sample in validation_dataset
    )

    test_distribution = Counter(
        get_target(sample)
        for sample in test_dataset
    )

    print(
        "\nClass distribution:"
    )

    print(
        f"Training   : "
        f"{dict(train_distribution)}"
    )

    print(
        f"Validation : "
        f"{dict(validation_distribution)}"
    )

    print(
        f"Test       : "
        f"{dict(test_distribution)}"
    )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n===================================")
    print(" DATASET SPLITTER RESULT")
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
        "Classes preserved  : YES"
    )

    print(
        "Dataset overlap    : NONE"
    )

    print(
        "RESULT             : SUCCESS"
    )


if __name__ == "__main__":
    test_dataset_splitter()