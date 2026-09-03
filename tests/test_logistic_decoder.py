from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.ml_dataset import MLDatasetBuilder

from decoders.ml_logistic import (
    LogisticRegressionDecoder
)

from evaluation.ml_evaluator import (
    MLEvaluator
)


def test_logistic_regression_decoder():

    print("\n===================================")
    print(" LOGISTIC REGRESSION AI DECODER")
    print("===================================")

    # -------------------------------------------------
    # 1. Generate dataset
    # -------------------------------------------------

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    print(
        f"\nDataset generated: "
        f"{len(dataset)}"
    )

    # -------------------------------------------------
    # 2. Validate dataset
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 3. Build ML dataset
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 4. Create AI decoder
    # -------------------------------------------------

    decoder = LogisticRegressionDecoder()

    print(
        "AI decoder creation: PASS"
    )

    # -------------------------------------------------
    # 5. Train AI decoder
    # -------------------------------------------------

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    print(
        "AI model training: PASS"
    )

    # -------------------------------------------------
    # 6. Evaluate AI decoder
    # -------------------------------------------------

    evaluator = MLEvaluator()

    predictions, metrics = (
        evaluator.evaluate(
            decoder,
            ml_dataset.X_test,
            ml_dataset.y_test
        )
    )

    evaluator.metrics.print_report(
        metrics,
        "Logistic Regression"
    )

    # -------------------------------------------------
    # 7. Validate predictions
    # -------------------------------------------------

    assert len(predictions) == (
        len(ml_dataset.y_test)
    )

    assert metrics["accuracy"] >= 0.90

    assert metrics["precision"] >= 0.90

    assert metrics["recall"] >= 0.90

    assert metrics["f1"] >= 0.90

    # -------------------------------------------------
    # 8. Test single-sample decoding
    # -------------------------------------------------

    prediction = decoder.decode(
        [1, 0]
    )

    print(
        "\nSingle syndrome test:"
    )

    print(
        "Syndrome features : [1, 0]"
    )

    print(
        f"Predicted class   : {prediction}"
    )

    # [1,0] corresponds to X on q0
    assert prediction == 1

    print(
        "Single prediction : PASS"
    )

    # -------------------------------------------------
    # 9. Test confidence
    # -------------------------------------------------

    prediction, confidence = (
        decoder.decode_with_confidence(
            [1, 0]
        )
    )

    print(
        "\nConfidence test:"
    )

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.4f}"
    )

    assert prediction == 1

    assert 0.0 <= confidence <= 1.0

    print(
        "Confidence output : PASS"
    )

    # -------------------------------------------------
    # 10. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" LOGISTIC REGRESSION RESULT")
    print("===================================")

    print(
        f"Test samples       : "
        f"{len(ml_dataset.X_test)}"
    )

    print(
        f"Accuracy            : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision           : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall              : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score            : "
        f"{metrics['f1']:.4f}"
    )

    print(
        "AI decoder status   : READY"
    )

    print(
        "RESULT              : SUCCESS"
    )


if __name__ == "__main__":
    test_logistic_regression_decoder()