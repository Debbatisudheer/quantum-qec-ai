from dataset.generator import QECDatasetGenerator
from dataset.validator import QECDatasetValidator
from dataset.analyzer import QECDatasetAnalyzer


def test_dataset_analysis():

    print("\n===================================")
    print(" DATASET ANALYZER TEST")
    print("===================================")

    # ---------------------------------
    # Generate dataset
    # ---------------------------------

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    # ---------------------------------
    # Validate dataset first
    # ---------------------------------

    validator = QECDatasetValidator()

    valid, errors = (
        validator.validate_dataset(
            dataset
        )
    )

    assert valid is True
    assert errors == []

    print("\nDataset validation: PASS")

    # ---------------------------------
    # Analyze dataset
    # ---------------------------------

    analyzer = QECDatasetAnalyzer()

    analysis = analyzer.analyze(
        dataset
    )

    analyzer.print_report(
        analysis
    )

    # ---------------------------------
    # Basic checks
    # ---------------------------------

    assert (
        analysis["total_samples"]
        == 1000
    )

    error_distribution = (
        analysis[
            "error_distribution"
        ]
    )

    # All four classes must exist.
    assert error_distribution["none"] > 0
    assert error_distribution["q0"] > 0
    assert error_distribution["q1"] > 0
    assert error_distribution["q2"] > 0

    # Check total error-class count.
    assert sum(
        error_distribution.values()
    ) == 1000

    # ---------------------------------
    # Logical-state checks
    # ---------------------------------

    logical_distribution = (
        analysis[
            "logical_state_distribution"
        ]
    )

    assert logical_distribution[0] > 0
    assert logical_distribution[1] > 0

    assert sum(
        logical_distribution.values()
    ) == 1000

    # ---------------------------------
    # Syndrome checks
    # ---------------------------------

    syndrome_distribution = (
        analysis[
            "syndrome_distribution"
        ]
    )

    assert syndrome_distribution["00"] > 0
    assert syndrome_distribution["10"] > 0
    assert syndrome_distribution["11"] > 0
    assert syndrome_distribution["01"] > 0

    assert sum(
        syndrome_distribution.values()
    ) == 1000

    print("\n===================================")
    print(" DATASET ANALYSIS RESULT")
    print("===================================")

    print("Samples analyzed : 1000")
    print("All error classes: PRESENT")
    print("Logical states   : PRESENT")
    print("All syndromes    : PRESENT")
    print("RESULT           : SUCCESS")


if __name__ == "__main__":
    test_dataset_analysis()