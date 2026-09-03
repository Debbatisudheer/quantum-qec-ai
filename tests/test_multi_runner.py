import tempfile

from experiments.config import (
    ExperimentConfig
)

from experiments.result_storage import (
    ExperimentResultStorage
)

from experiments.result_query import (
    ExperimentResultQuery
)

from experiments.multi_runner import (
    MultiExperimentRunner
)


def main():

    print()
    print(
        "TESTING MULTI-EXPERIMENT RUNNER"
    )
    print()

    # --------------------------------------------------------
    # TEMPORARY STORAGE
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        storage = ExperimentResultStorage(
            storage_directory=temp_dir
        )

        runner = MultiExperimentRunner(
            storage=storage
        )

        print(
            "MultiExperimentRunner creation : PASS"
        )

        # ----------------------------------------------------
        # BUILD SMALL GRID
        # ----------------------------------------------------

        configs = runner.build_grid(
            qec_code="bit_flip_3",

            num_qubits=3,

            rounds_list=[
                3,
                5
            ],

            physical_noise_list=[
                0.05,
                0.10
            ],

            measurement_noise_list=[
                0.10
            ],

            training_samples=100,

            test_samples=50,

            decoder_type=(
                "logical_target_random_forest"
            ),

            random_forest_estimators=10,

            seed=42
        )

        # 2 rounds × 2 physical noise
        # × 1 measurement noise = 4

        assert len(configs) == 4

        for config in configs:

            assert isinstance(
                config,
                ExperimentConfig
            )

            config.validate()

        print(
            "Configuration grid          : PASS"
        )

        # ----------------------------------------------------
        # RUN GRID
        # ----------------------------------------------------

        results = runner.run(
            configs
        )

        assert len(
            results
        ) == 4

        print()
        print(
            "Multi-experiment execution : PASS"
        )

        # ----------------------------------------------------
        # RESULT VALIDATION
        # ----------------------------------------------------

        for result in results:

            assert (
                result.training_samples
                == 100
            )

            assert (
                result.test_samples
                == 50
            )

            assert (
                result.decoder_type
                == "logical_target_random_forest"
            )

            assert (
                0.0
                <= result.logical_accuracy
                <= 1.0
            )

        print(
            "Result validation           : PASS"
        )

        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        stored_ids = (
            storage.list_results()
        )

        assert len(
            stored_ids
        ) == 4

        print(
            "All results stored          : PASS"
        )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        query = (
            ExperimentResultQuery(
                storage
            )
        )

        all_results = (
            query.all_results()
        )

        assert len(
            all_results
        ) == 4

        print(
            "Query integration           : PASS"
        )

        # ----------------------------------------------------
        # FILTER BY ROUNDS
        # ----------------------------------------------------

        round_5_results = (
            query.filter(
                rounds=5
            )
        )

        assert len(
            round_5_results
        ) == 2

        print(
            "Round filtering             : PASS"
        )

        # ----------------------------------------------------
        # FILTER BY PHYSICAL NOISE
        # ----------------------------------------------------

        noise_010_results = (
            query.filter(
                physical_noise_probability=0.10
            )
        )

        assert len(
            noise_010_results
        ) == 2

        print(
            "Noise filtering             : PASS"
        )

        # ----------------------------------------------------
        # BEST RESULT
        # ----------------------------------------------------

        best = query.best(
            all_results,
            "logical_accuracy"
        )

        assert best is not None

        assert (
            0.0
            <= best["logical_accuracy"]
            <= 1.0
        )

        print(
            "Best experiment             : PASS"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = (
            query.summary(
                all_results
            )
        )

        assert (
            summary["count"]
            == 4
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
            "Experiment summary          : PASS"
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            " MULTI-EXPERIMENT SUMMARY"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Experiments executed : "
            f"{summary['count']}"
        )

        print(
            f"Best logical success : "
            f"{summary['best_logical_accuracy']:.4f}"
        )

        print(
            f"Average logical      : "
            f"{summary['average_logical_accuracy']:.4f}"
        )

        print()

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        print(
            "=" * 70
        )

        print(
            " MULTI-EXPERIMENT RUNNER TEST : SUCCESS"
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()