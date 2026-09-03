from experiments.config import (
    ExperimentConfig
)

from experiments.engine import (
    ExperimentEngine
)

from experiments.result_storage import (
    ExperimentResultStorage
)

from experiments.result_query import (
    ExperimentResultQuery
)


def create_experiment(
    physical_noise,
    measurement_noise,
    seed
):
    """
    Run a small experiment so the query
    layer has real stored results to inspect.
    """

    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        rounds=5,

        physical_noise_probability=(
            physical_noise
        ),

        measurement_noise_probability=(
            measurement_noise
        ),

        training_samples=500,

        test_samples=200,

        decoder_type=(
            "logical_target_random_forest"
        ),

        random_forest_estimators=20,

        seed=seed
    )

    storage = (
        ExperimentResultStorage()
    )

    engine = ExperimentEngine(
        config=config,
        storage=storage
    )

    return engine.run()


def main():

    print()
    print(
        "TESTING EXPERIMENT RESULT QUERY"
    )
    print()

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    storage = (
        ExperimentResultStorage()
    )

    query = (
        ExperimentResultQuery(
            storage
        )
    )

    print(
        "ResultQuery creation      : PASS"
    )

    # --------------------------------------------------------
    # CREATE TEST EXPERIMENTS
    # --------------------------------------------------------

    print()
    print(
        "Creating test experiments..."
    )

    result_1 = create_experiment(
        physical_noise=0.05,
        measurement_noise=0.10,
        seed=100
    )

    result_2 = create_experiment(
        physical_noise=0.10,
        measurement_noise=0.10,
        seed=200
    )

    result_3 = create_experiment(
        physical_noise=0.20,
        measurement_noise=0.10,
        seed=300
    )

    print()
    print(
        "Test experiments created : PASS"
    )

    # --------------------------------------------------------
    # ALL RESULTS
    # --------------------------------------------------------

    all_results = (
        query.all_results()
    )

    assert len(
        all_results
    ) >= 3

    print(
        "Load all results         : PASS"
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    filtered = query.filter(
        rounds=5,
        decoder_type=(
            "logical_target_random_forest"
        )
    )

    assert len(
        filtered
    ) >= 3

    print(
        "Result filtering         : PASS"
    )

    # --------------------------------------------------------
    # FILTER BY NOISE
    # --------------------------------------------------------

    noise_results = (
        query.filter(
            physical_noise_probability=0.10
        )
    )

    assert len(
        noise_results
    ) >= 1

    for result in noise_results:

        assert (
            result["config"][
                "physical_noise_probability"
            ]
            == 0.10
        )

    print(
        "Noise filtering         : PASS"
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    sorted_results = (
        query.sort_by(
            all_results,
            "logical_accuracy"
        )
    )

    assert len(
        sorted_results
    ) >= 3

    for index in range(
        len(sorted_results) - 1
    ):

        assert (
            sorted_results[index][
                "logical_accuracy"
            ]
            >=
            sorted_results[index + 1][
                "logical_accuracy"
            ]
        )

    print(
        "Result sorting           : PASS"
    )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best = query.best(
        all_results,
        "logical_accuracy"
    )

    assert best is not None

    print(
        "Best result              : PASS"
    )

    # --------------------------------------------------------
    # WORST
    # --------------------------------------------------------

    worst = query.worst(
        all_results,
        "logical_accuracy"
    )

    assert worst is not None

    print(
        "Worst result             : PASS"
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    comparison = query.compare(
        [
            result_1.experiment_id,
            result_2.experiment_id,
            result_3.experiment_id,
        ]
    )

    assert len(
        comparison
    ) == 3

    assert (
        comparison[0][
            "logical_accuracy"
        ]
        >=
        comparison[1][
            "logical_accuracy"
        ]
    )

    assert (
        comparison[1][
            "logical_accuracy"
        ]
        >=
        comparison[2][
            "logical_accuracy"
        ]
    )

    print(
        "Experiment comparison    : PASS"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        query.summary(
            comparison
        )
    )

    assert (
        summary["count"]
        == 3
    )

    assert (
        summary[
            "best_logical_accuracy"
        ]
        is not None
    )

    assert (
        summary[
            "average_logical_accuracy"
        ]
        is not None
    )

    print(
        "Result summary            : PASS"
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "Comparison summary:"
    )

    print(
        f"Experiments              : "
        f"{summary['count']}"
    )

    print(
        f"Best logical success     : "
        f"{summary['best_logical_accuracy']:.4f}"
    )

    print(
        f"Average logical success  : "
        f"{summary['average_logical_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " EXPERIMENT RESULT QUERY TEST : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()