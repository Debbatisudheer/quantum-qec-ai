import tempfile

from experiments.result_storage import (
    ExperimentResultStorage
)

from experiments.result_query import (
    ExperimentResultQuery
)

from experiments.multi_runner import (
    MultiExperimentRunner
)

from experiments.analysis import (
    ExperimentAnalysis
)


def main():

    print()
    print(
        "TESTING EXPERIMENT ANALYSIS"
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

        # ----------------------------------------------------
        # CREATE SMALL EXPERIMENT GRID
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

        print(
            "Analysis experiment grid : PASS"
        )

        # ----------------------------------------------------
        # RUN EXPERIMENTS
        # ----------------------------------------------------

        results = runner.run(
            configs
        )

        assert len(
            results
        ) == 4

        print(
            "Analysis test experiments : PASS"
        )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        query = ExperimentResultQuery(
            storage
        )

        analysis = ExperimentAnalysis(
            query
        )

        print(
            "ExperimentAnalysis creation : PASS"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = analysis.summary(
            results
        )

        assert (
            summary["count"]
            == 4
        )

        assert (
            0.0
            <= summary[
                "average_logical_accuracy"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= summary[
                "best_logical_accuracy"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= summary[
                "worst_logical_accuracy"
            ]
            <= 1.0
        )

        print(
            "Summary analysis          : PASS"
        )

        # ----------------------------------------------------
        # BASIC STATISTICS
        # ----------------------------------------------------

        average = (
            analysis.average(
                results,
                "logical_accuracy"
            )
        )

        minimum = (
            analysis.minimum(
                results,
                "logical_accuracy"
            )
        )

        maximum = (
            analysis.maximum(
                results,
                "logical_accuracy"
            )
        )

        assert (
            minimum
            <= average
            <= maximum
        )

        print(
            "Basic statistics          : PASS"
        )

        # ----------------------------------------------------
        # ROUND ANALYSIS
        # ----------------------------------------------------

        rounds = (
            analysis.analyze_rounds(
                results
            )
        )

        assert set(
            rounds.keys()
        ) == {
            3,
            5
        }

        print(
            "Round analysis            : PASS"
        )

        # ----------------------------------------------------
        # PHYSICAL NOISE ANALYSIS
        # ----------------------------------------------------

        physical_noise = (
            analysis.analyze_physical_noise(
                results
            )
        )

        assert set(
            physical_noise.keys()
        ) == {
            0.05,
            0.10
        }

        print(
            "Physical noise analysis   : PASS"
        )

        # ----------------------------------------------------
        # MEASUREMENT NOISE ANALYSIS
        # ----------------------------------------------------

        measurement_noise = (
            analysis.analyze_measurement_noise(
                results
            )
        )

        assert set(
            measurement_noise.keys()
        ) == {
            0.10
        }

        print(
            "Measurement noise analysis : PASS"
        )

        # ----------------------------------------------------
        # DECODER ANALYSIS
        # ----------------------------------------------------

        decoders = (
            analysis.analyze_decoders(
                results
            )
        )

        assert set(
            decoders.keys()
        ) == {
            "logical_target_random_forest"
        }

        print(
            "Decoder analysis           : PASS"
        )

        # ----------------------------------------------------
        # BEST / WORST
        # ----------------------------------------------------

        best = (
            analysis.best_experiment(
                results
            )
        )

        worst = (
            analysis.worst_experiment(
                results
            )
        )

        assert best is not None
        assert worst is not None

        # ExperimentResult is an object,
        # so use attribute access.

        assert (
            best.logical_accuracy
            >=
            worst.logical_accuracy
        )

        print(
            "Best/worst analysis        : PASS"
        )

        # ----------------------------------------------------
        # GAIN
        # ----------------------------------------------------

        gain = (
            analysis.calculate_gain(
                0.70,
                0.80
            )
        )

        assert abs(
            gain - 0.10
        ) < 1e-9

        print(
            "Gain calculation           : PASS"
        )

        # ----------------------------------------------------
        # RESULT ROWS
        # ----------------------------------------------------

        rows = (
            analysis.result_rows(
                results
            )
        )

        assert len(
            rows
        ) == 4

        required_fields = {
            "experiment_id",
            "qec_code",
            "rounds",
            "physical_noise",
            "measurement_noise",
            "decoder",
            "logical_accuracy",
            "physical_accuracy",
            "bit_accuracy",
            "training_seconds",
            "inference_seconds",
            "samples_per_second",
        }

        assert required_fields.issubset(
            rows[0].keys()
        )

        print(
            "Result table generation   : PASS"
        )

        # ----------------------------------------------------
        # PRINT ANALYSIS
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            " EXPERIMENT ANALYSIS SUMMARY"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Experiments              : "
            f"{summary['count']}"
        )

        print(
            f"Average logical success  : "
            f"{summary['average_logical_accuracy']:.4f}"
        )

        print(
            f"Best logical success     : "
            f"{summary['best_logical_accuracy']:.4f}"
        )

        print(
            f"Worst logical success    : "
            f"{summary['worst_logical_accuracy']:.4f}"
        )

        print()

        print(
            "Logical success by rounds:"
        )

        for key in sorted(
            rounds
        ):
            print(
                f"  Rounds {key}: "
                f"{rounds[key]:.4f}"
            )

        print()

        print(
            "Logical success by physical noise:"
        )

        for key in sorted(
            physical_noise
        ):
            print(
                f"  Noise {key:.2f}: "
                f"{physical_noise[key]:.4f}"
            )

        print()

        print(
            "=" * 70
        )

        print(
            " EXPERIMENT ANALYSIS TEST : SUCCESS"
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()