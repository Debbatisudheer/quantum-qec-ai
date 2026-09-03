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

from experiments.report import (
    ExperimentReportGenerator
)


def main():

    print()
    print(
        "TESTING EXPERIMENT REPORT GENERATOR"
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
        # CREATE SMALL GRID
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
            "Report experiment grid : PASS"
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
            "Report test experiments : PASS"
        )

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        query = ExperimentResultQuery(
            storage
        )

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        analysis = ExperimentAnalysis(
            query
        )

        print(
            "Analysis integration     : PASS"
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        report_generator = (
            ExperimentReportGenerator(
                analysis
            )
        )

        print(
            "Report generator creation : PASS"
        )

        # ----------------------------------------------------
        # BUILD STRUCTURED REPORT
        # ----------------------------------------------------

        report = (
            report_generator.build(
                results
            )
        )

        assert isinstance(
            report,
            dict
        )

        assert (
            report[
                "experiment_count"
            ]
            == 4
        )

        assert (
            "summary"
            in report
        )

        assert (
            "round_analysis"
            in report
        )

        assert (
            "physical_noise_analysis"
            in report
        )

        assert (
            "measurement_noise_analysis"
            in report
        )

        assert (
            "decoder_analysis"
            in report
        )

        assert (
            "best_experiment"
            in report
        )

        assert (
            "worst_experiment"
            in report
        )

        assert (
            "results"
            in report
        )

        print(
            "Structured report        : PASS"
        )

        # ----------------------------------------------------
        # SUMMARY VALIDATION
        # ----------------------------------------------------

        summary = report[
            "summary"
        ]

        assert (
            summary[
                "count"
            ]
            == 4
        )

        assert (
            0.0
            <= summary[
                "average_logical_accuracy"
            ]
            <= 1.0
        )

        print(
            "Report summary            : PASS"
        )

        # ----------------------------------------------------
        # ROUND VALIDATION
        # ----------------------------------------------------

        round_analysis = report[
            "round_analysis"
        ]

        assert set(
            round_analysis.keys()
        ) == {
            3,
            5
        }

        print(
            "Round report              : PASS"
        )

        # ----------------------------------------------------
        # NOISE VALIDATION
        # ----------------------------------------------------

        physical_noise = report[
            "physical_noise_analysis"
        ]

        assert set(
            physical_noise.keys()
        ) == {
            0.05,
            0.10
        }

        measurement_noise = report[
            "measurement_noise_analysis"
        ]

        assert set(
            measurement_noise.keys()
        ) == {
            0.10
        }

        print(
            "Noise report              : PASS"
        )

        # ----------------------------------------------------
        # DECODER VALIDATION
        # ----------------------------------------------------

        decoder_analysis = report[
            "decoder_analysis"
        ]

        assert set(
            decoder_analysis.keys()
        ) == {
            "logical_target_random_forest"
        }

        print(
            "Decoder report            : PASS"
        )

        # ----------------------------------------------------
        # BEST / WORST
        # ----------------------------------------------------

        assert (
            report[
                "best_experiment"
            ]
            is not None
        )

        assert (
            report[
                "worst_experiment"
            ]
            is not None
        )

        best_accuracy = (
            report[
                "best_experiment"
            ][
                "logical_accuracy"
            ]
        )

        worst_accuracy = (
            report[
                "worst_experiment"
            ][
                "logical_accuracy"
            ]
        )

        assert (
            best_accuracy
            >=
            worst_accuracy
        )

        print(
            "Best/worst report        : PASS"
        )

        # ----------------------------------------------------
        # RESULT ROWS
        # ----------------------------------------------------

        rows = (
            report_generator.result_rows(
                results
            )
        )

        assert len(
            rows
        ) == 4

        print(
            "Report result rows       : PASS"
        )

        # ----------------------------------------------------
        # TEXT REPORT
        # ----------------------------------------------------

        text_report = (
            report_generator.to_text(
                results
            )
        )

        assert isinstance(
            text_report,
            str
        )

        assert (
            "QEC EXPERIMENT REPORT"
            in text_report
        )

        assert (
            "ROUND ANALYSIS"
            in text_report
        )

        assert (
            "PHYSICAL NOISE ANALYSIS"
            in text_report
        )

        assert (
            "MEASUREMENT NOISE ANALYSIS"
            in text_report
        )

        assert (
            "DECODER ANALYSIS"
            in text_report
        )

        assert (
            "BEST EXPERIMENT"
            in text_report
        )

        assert (
            "WORST EXPERIMENT"
            in text_report
        )

        print(
            "Text report generation   : PASS"
        )

        # ----------------------------------------------------
        # PRINT REPORT
        # ----------------------------------------------------

        print()
        print(
            text_report
        )

        print()
        print(
            "EXPERIMENT REPORT TEST : SUCCESS"
        )


if __name__ == "__main__":
    main()