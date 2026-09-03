import tempfile
from pathlib import Path

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

from experiments.visualization import (
    ExperimentVisualization
)


def main():

    print()
    print(
        "TESTING EXPERIMENT VISUALIZATION"
    )
    print()

    # ========================================================
    # TEMPORARY STORAGE
    # ========================================================

    with tempfile.TemporaryDirectory() as temp_dir:

        storage = ExperimentResultStorage(
            storage_directory=temp_dir
        )

        print(
            "Result storage creation   : PASS"
        )

        # ====================================================
        # RUN SMALL EXPERIMENT GRID
        # ====================================================

        runner = MultiExperimentRunner(
            storage=storage
        )

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
            "Visualization grid        : PASS"
        )

        results = runner.run(
            configs
        )

        assert len(
            results
        ) == 4

        print(
            "Visualization experiments : PASS"
        )

        # ====================================================
        # QUERY
        # ====================================================

        query = ExperimentResultQuery(
            storage
        )

        print(
            "Result query creation     : PASS"
        )

        # ====================================================
        # ANALYSIS
        # ====================================================

        analysis = ExperimentAnalysis(
            query
        )

        print(
            "Analysis creation         : PASS"
        )

        # ====================================================
        # VISUALIZATION
        # ====================================================

        visualization = ExperimentVisualization(
            analysis
        )

        print(
            "Visualization creation    : PASS"
        )

        # ====================================================
        # ROUND DATA
        # ====================================================

        round_data = (
            visualization.logical_success_by_rounds(
                results
            )
        )

        assert len(
            round_data
        ) == 2

        assert (
            round_data[0]["rounds"]
            == 3
        )

        assert (
            round_data[1]["rounds"]
            == 5
        )

        assert all(
            0.0 <= row["logical_success"] <= 1.0
            for row in round_data
        )

        print(
            "Round visualization data  : PASS"
        )

        # ====================================================
        # PHYSICAL NOISE DATA
        # ====================================================

        physical_noise_data = (
            visualization.logical_success_by_physical_noise(
                results
            )
        )

        assert len(
            physical_noise_data
        ) == 2

        assert (
            physical_noise_data[0][
                "physical_noise"
            ]
            == 0.05
        )

        assert (
            physical_noise_data[1][
                "physical_noise"
            ]
            == 0.10
        )

        print(
            "Physical-noise data       : PASS"
        )

        # ====================================================
        # MEASUREMENT NOISE DATA
        # ====================================================

        measurement_noise_data = (
            visualization.logical_success_by_measurement_noise(
                results
            )
        )

        assert len(
            measurement_noise_data
        ) == 1

        assert (
            measurement_noise_data[0][
                "measurement_noise"
            ]
            == 0.10
        )

        print(
            "Measurement-noise data    : PASS"
        )

        # ====================================================
        # DECODER DATA
        # ====================================================

        decoder_data = (
            visualization.decoder_comparison(
                results
            )
        )

        assert len(
            decoder_data
        ) == 1

        assert (
            decoder_data[0]["decoder"]
            == "logical_target_random_forest"
        )

        assert (
            0.0
            <= decoder_data[0][
                "logical_success"
            ]
            <= 1.0
        )

        print(
            "Decoder comparison data   : PASS"
        )

        # ====================================================
        # PERFORMANCE DATA
        # ====================================================

        performance_data = (
            visualization.performance_comparison(
                results
            )
        )

        assert len(
            performance_data
        ) == 4

        required_fields = {
            "experiment_id",
            "decoder",
            "rounds",
            "physical_noise",
            "measurement_noise",
            "logical_success",
            "physical_recovery",
            "bit_accuracy",
            "training_seconds",
            "inference_seconds",
            "samples_per_second",
        }

        assert required_fields.issubset(
            performance_data[0].keys()
        )

        print(
            "Performance data          : PASS"
        )

        # ====================================================
        # COMPLETE BUILD
        # ====================================================

        visualization_data = (
            visualization.build(
                results
            )
        )

        expected_keys = {
            "logical_success_by_rounds",
            "logical_success_by_physical_noise",
            "logical_success_by_measurement_noise",
            "decoder_comparison",
            "performance_comparison",
        }

        assert set(
            visualization_data.keys()
        ) == expected_keys

        print(
            "Complete visualization    : PASS"
        )

        # ====================================================
        # FRONTEND DATA
        # ====================================================

        frontend_data = (
            visualization.frontend_data(
                results
            )
        )

        assert "charts" in frontend_data

        assert "performance" in frontend_data

        assert "rounds" in frontend_data["charts"]

        assert (
            "physical_noise"
            in frontend_data["charts"]
        )

        assert (
            "measurement_noise"
            in frontend_data["charts"]
        )

        assert (
            "decoders"
            in frontend_data["charts"]
        )

        assert len(
            frontend_data["performance"]
        ) == 4

        print(
            "Frontend visualization    : PASS"
        )

        # ====================================================
        # EMPTY RESULTS
        # ====================================================

        empty_data = (
            visualization.build(
                []
            )
        )

        assert (
            empty_data[
                "logical_success_by_rounds"
            ]
            == []
        )

        assert (
            empty_data[
                "performance_comparison"
            ]
            == []
        )

        print(
            "Empty-result handling     : PASS"
        )

        # ====================================================
        # STORED RESULTS
        # ====================================================

        stored_results = (
            query.all_results()
        )

        stored_round_data = (
            visualization.logical_success_by_rounds(
                stored_results
            )
        )

        assert len(
            stored_round_data
        ) == 2

        print(
            "Stored-result visualization: PASS"
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            " VISUALIZATION SUMMARY"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Experiments analyzed : "
            f"{len(results)}"
        )

        print()

        print(
            "Logical success by rounds:"
        )

        for row in round_data:

            print(
                f"  Rounds "
                f"{row['rounds']}: "
                f"{row['logical_success']:.4f}"
            )

        print()

        print(
            "Logical success by physical noise:"
        )

        for row in physical_noise_data:

            print(
                f"  Noise "
                f"{row['physical_noise']}: "
                f"{row['logical_success']:.4f}"
            )

        print()

        print(
            "Decoder comparison:"
        )

        for row in decoder_data:

            print(
                f"  {row['decoder']}: "
                f"{row['logical_success']:.4f}"
            )

        print()

        print(
            "=" * 70
        )

        print(
            " EXPERIMENT VISUALIZATION TEST : SUCCESS"
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()