from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator


def test_valid_dataset():

    print("\n===================================")
    print(" DATASET VALIDATOR TEST")
    print("===================================")

    # ---------------------------------
    # Generate dataset
    # ---------------------------------

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=100
    )

    # ---------------------------------
    # Validate dataset
    # ---------------------------------

    validator = QECDatasetValidator()

    valid, errors = (
        validator.validate_dataset(
            dataset
        )
    )

    print("\nSamples checked:")
    print(len(dataset))

    print("\nValidation result:")

    if valid:
        print("VALID")
    else:
        print("INVALID")

        for error in errors:
            print(error)

    # ---------------------------------
    # Test must pass
    # ---------------------------------

    assert valid is True
    assert errors == []

    print("\n===================================")
    print(" VALIDATION RESULT")
    print("===================================")

    print("Samples checked : 100")
    print("Invalid samples : 0")
    print("RESULT          : SUCCESS")


def test_invalid_syndrome():

    print("\n===================================")
    print(" INVALID SAMPLE TEST")
    print("===================================")

    generator = QECDatasetGenerator(
        seed=42
    )

    sample = generator.generate_sample(
        sample_id=0
    )

    # Intentionally corrupt the syndrome.
    sample.syndrome = "11"

    validator = QECDatasetValidator()

    valid, errors = (
        validator.validate_sample(
            sample
        )
    )

    print("\nIntentionally corrupted sample:")
    print(
        f"Sample ID : {sample.sample_id}"
    )

    print(
        f"Error     : {sample.error_description}"
    )

    print(
        f"Syndrome  : {sample.syndrome}"
    )

    print("\nValidator result:")

    print(
        f"Valid     : {valid}"
    )

    print("\nDetected errors:")

    for error in errors:
        print(
            f"- {error}"
        )

    # The validator MUST reject it.
    assert valid is False

    print("\nRESULT: INVALID DATA DETECTED")


if __name__ == "__main__":

    test_valid_dataset()

    test_invalid_syndrome()