from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.ml_dataset import MLDatasetBuilder
from evaluation.baseline import BaselineEvaluator


def test_baseline_decoder():

    print("\n===================================")
    print(" BASELINE DECODER TEST")
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
    # 4. Get test syndromes
    # --------------------------------

    test_samples = []

    # Recreate the same split so that
    # the baseline receives exactly
    # the same test samples.

    (
        train_samples,
        validation_samples,
        test_samples
    ) = builder.splitter.split(
        dataset
    )

    test_syndromes = [
        sample.syndrome
        for sample in test_samples
    ]

    y_test = ml_dataset.y_test

    # --------------------------------
    # 5. Evaluate baseline
    # --------------------------------

    evaluator = BaselineEvaluator()

    predictions, metrics = (
        evaluator.evaluate(
            test_syndromes,
            y_test
        )
    )

    evaluator.metrics.print_report(
        metrics,
        "Traditional Lookup Decoder"
    )

    # --------------------------------
    # 6. Verify predictions
    # --------------------------------

    assert len(
        predictions
    ) == len(y_test)

    assert (
        metrics["accuracy"]
        == 1.0
    )

    assert (
        metrics["precision"]
        == 1.0
    )

    assert (
        metrics["recall"]
        == 1.0
    )

    assert (
        metrics["f1"]
        == 1.0
    )

    # --------------------------------
    # Final result
    # --------------------------------

    print("\n===================================")
    print(" BASELINE DECODER RESULT")
    print("===================================")

    print(
        "Test samples       : 100"
    )

    print(
        "Accuracy            : 100%"
    )

    print(
        "Precision           : 100%"
    )

    print(
        "Recall              : 100%"
    )

    print(
        "F1 Score            : 100%"
    )

    print(
        "Baseline status     : PERFECT"
    )

    print(
        "RESULT              : SUCCESS"
    )


if __name__ == "__main__":
    test_baseline_decoder()