import os

from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.analyzer import QECDatasetAnalyzer
from dataset.storage import QECDatasetStorage


def test_dataset_storage():

    print("\n===================================")
    print(" DATASET STORAGE TEST")
    print("===================================")

    # --------------------------------
    # 1. Generate dataset
    # --------------------------------

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=100
    )

    print("\nDataset generated:")
    print(f"Samples: {len(dataset)}")

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

    print("Dataset validation: PASS")

    # --------------------------------
    # 3. Analyze dataset
    # --------------------------------

    analyzer = QECDatasetAnalyzer()

    analysis = analyzer.analyze(
        dataset
    )

    print(
        "Dataset analysis: PASS"
    )

    assert (
        analysis["total_samples"]
        == 100
    )

    # --------------------------------
    # 4. Create storage engine
    # --------------------------------

    storage = QECDatasetStorage()

    # --------------------------------
    # 5. CSV storage
    # --------------------------------

    csv_filepath = (
        "test_qec_dataset.csv"
    )

    storage.save_csv(
        dataset,
        csv_filepath
    )

    assert os.path.exists(
        csv_filepath
    )

    print(
        "CSV save: PASS"
    )

    # Load CSV back

    loaded_csv_dataset = (
        storage.load_csv(
            csv_filepath
        )
    )

    assert len(
        loaded_csv_dataset
    ) == len(dataset)

    # Verify samples

    for original, loaded in zip(
        dataset,
        loaded_csv_dataset
    ):

        assert original == loaded

    print(
        "CSV load: PASS"
    )

    # --------------------------------
    # 6. JSON storage
    # --------------------------------

    json_filepath = (
        "test_qec_dataset.json"
    )

    storage.save_json(
        dataset,
        json_filepath
    )

    assert os.path.exists(
        json_filepath
    )

    print(
        "JSON save: PASS"
    )

    # Load JSON back

    loaded_json_dataset = (
        storage.load_json(
            json_filepath
        )
    )

    assert len(
        loaded_json_dataset
    ) == len(dataset)

    # Verify samples

    for original, loaded in zip(
        dataset,
        loaded_json_dataset
    ):

        assert original == loaded

    print(
        "JSON load: PASS"
    )

    # --------------------------------
    # 7. Validate loaded datasets
    # --------------------------------

    csv_valid, csv_errors = (
        validator.validate_dataset(
            loaded_csv_dataset
        )
    )

    assert csv_valid is True
    assert csv_errors == []

    json_valid, json_errors = (
        validator.validate_dataset(
            loaded_json_dataset
        )
    )

    assert json_valid is True
    assert json_errors == []

    print(
        "Loaded dataset validation: PASS"
    )

    # --------------------------------
    # 8. Clean up test files
    # --------------------------------

    if os.path.exists(
        csv_filepath
    ):
        os.remove(
            csv_filepath
        )

    if os.path.exists(
        json_filepath
    ):
        os.remove(
            json_filepath
        )

    print(
        "Test file cleanup: PASS"
    )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n===================================")
    print(" DATASET STORAGE RESULT")
    print("===================================")

    print("Samples stored       : 100")
    print("CSV save             : PASS")
    print("CSV load             : PASS")
    print("JSON save            : PASS")
    print("JSON load            : PASS")
    print("Loaded validation    : PASS")
    print("RESULT               : SUCCESS")


if __name__ == "__main__":
    test_dataset_storage()