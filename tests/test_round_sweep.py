from experiments.round_sweep import (
    RoundSweepExperiment
)


def test_round_sweep():

    print("\n===================================")
    print(" ROUND SWEEP TEST")
    print("===================================")

    experiment = RoundSweepExperiment(
        rounds_list=[
            1,
            2,
            3,
            5,
            7,
            10
        ],
        measurement_noise_probability=0.10,
        dataset_size=5000,
        random_state=42
    )

    # -------------------------------------------------
    # 1. Run experiment
    # -------------------------------------------------

    results = experiment.run()

    # -------------------------------------------------
    # 2. Check result count
    # -------------------------------------------------

    assert len(results) == 6

    print(
        "\nResult count: PASS"
    )

    # -------------------------------------------------
    # 3. Check expected rounds
    # -------------------------------------------------

    actual_rounds = [
        result["rounds"]
        for result in results
    ]

    expected_rounds = [
        1,
        2,
        3,
        5,
        7,
        10
    ]

    assert (
        actual_rounds
        == expected_rounds
    )

    print(
        "Round configurations: PASS"
    )

    # -------------------------------------------------
    # 4. Check feature dimensions
    # -------------------------------------------------

    for result in results:

        expected_features = (
            result["rounds"] * 2
        )

        assert (
            result["features"]
            == expected_features
        )

    print(
        "Feature dimensions: PASS"
    )

    # -------------------------------------------------
    # 5. Check metrics
    # -------------------------------------------------

    metric_names = [
        "logistic_accuracy",
        "logistic_f1",
        "random_forest_accuracy",
        "random_forest_f1",
        "mlp_accuracy",
        "mlp_f1"
    ]

    for result in results:

        for metric_name in metric_names:

            value = result[
                metric_name
            ]

            assert (
                0.0
                <= value
                <= 1.0
            )

    print(
        "Metric ranges: PASS"
    )

    # -------------------------------------------------
    # 6. Print summary
    # -------------------------------------------------

    experiment.print_summary(
        results
    )

    # -------------------------------------------------
    # 7. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" ROUND SWEEP TEST RESULT")
    print("===================================")

    print(
        "1 → 10 rounds tested : PASS"
    )

    print(
        "Feature scaling      : PASS"
    )

    print(
        "AI evaluation        : PASS"
    )

    print(
        "RESULT               : SUCCESS"
    )


if __name__ == "__main__":
    test_round_sweep()