import csv
import json
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

from experiments.export import (
    ExperimentExporter
)


def main():

    print()
    print(
        "TESTING EXPERIMENT EXPORT"
    )
    print()

    with tempfile.TemporaryDirectory() as temp_dir:

        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        storage = ExperimentResultStorage(
            storage_directory=temp_dir
        )

        runner = MultiExperimentRunner(
            storage=storage
        )

        # ----------------------------------------------------
        # EXPERIMENT GRID
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
            "Export experiment grid : PASS"
        )

        # ----------------------------------------------------
        # RUN
        # ----------------------------------------------------

        results = runner.run(
            configs
        )

        assert len(
            results
        ) == 4

        print(
            "Export test experiments : PASS"
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
            "Analysis integration      : PASS"
        )

        # ----------------------------------------------------
        # EXPORTER
        # ----------------------------------------------------

        exporter = ExperimentExporter(
            analysis
        )

        print(
            "ExperimentExporter creation : PASS"
        )

        # ----------------------------------------------------
        # OUTPUT DIRECTORY
        # ----------------------------------------------------

        output_directory = (
            Path(temp_dir)
            / "exports"
        )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        json_path = (
            exporter.export_json(
                results,
                output_directory
                / "test.json"
            )
        )

        assert json_path.exists()

        with json_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            json_data = json.load(
                file
            )

        assert isinstance(
            json_data,
            list
        )

        assert len(
            json_data
        ) == 4

        print(
            "JSON export               : PASS"
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_path = (
            exporter.export_csv(
                results,
                output_directory
                / "test.csv"
            )
        )

        assert csv_path.exists()

        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            csv_rows = list(
                reader
            )

        assert len(
            csv_rows
        ) == 4

        assert (
            "experiment_id"
            in csv_rows[0]
        )

        assert (
            "logical_accuracy"
            in csv_rows[0]
        )

        print(
            "CSV export                : PASS"
        )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_path = (
            exporter.export_text(
                results,
                output_directory
                / "test.txt"
            )
        )

        assert text_path.exists()

        text = (
            text_path.read_text(
                encoding="utf-8"
            )
        )

        assert (
            "QEC EXPERIMENT REPORT"
            in text
        )

        assert (
            "ROUND ANALYSIS"
            in text
        )

        assert (
            "PHYSICAL NOISE ANALYSIS"
            in text
        )

        assert (
            "MEASUREMENT NOISE ANALYSIS"
            in text
        )

        assert (
            "DECODER ANALYSIS"
            in text
        )

        print(
            "Text export               : PASS"
        )

        # ----------------------------------------------------
        # EXPORT ALL
        # ----------------------------------------------------

        all_directory = (
            Path(temp_dir)
            / "all_exports"
        )

        exported = (
            exporter.export_all(
                results,
                all_directory
            )
        )

        assert set(
            exported.keys()
        ) == {
            "json",
            "csv",
            "text"
        }

        for path in exported.values():

            assert path.exists()

        print(
            "Export all                : PASS"
        )

        # ----------------------------------------------------
        # LOADED RESULTS
        # ----------------------------------------------------

        loaded_results = (
            query.all_results()
        )

        assert len(
            loaded_results
        ) == 4

        # ----------------------------------------------------
        # EXPORT LOADED RESULTS
        # ----------------------------------------------------

        loaded_json_path = (
            exporter.export_json(
                loaded_results,
                output_directory
                / "loaded.json"
            )
        )

        assert (
            loaded_json_path.exists()
        )

        print(
            "Stored-result export      : PASS"
        )

        # ----------------------------------------------------
        # EMPTY EXPORT
        # ----------------------------------------------------

        empty_directory = (
            Path(temp_dir)
            / "empty"
        )

        empty_json = (
            exporter.export_json(
                [],
                empty_directory
                / "empty.json"
            )
        )

        assert empty_json.exists()

        with empty_json.open(
            "r",
            encoding="utf-8"
        ) as file:

            empty_data = json.load(
                file
            )

        assert empty_data == []

        print(
            "Empty-result export       : PASS"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            " EXPORT SUMMARY"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Experiments exported : "
            f"{len(results)}"
        )

        print(
            f"JSON file            : "
            f"{json_path}"
        )

        print(
            f"CSV file             : "
            f"{csv_path}"
        )

        print(
            f"Text file            : "
            f"{text_path}"
        )

        print()

        print(
            "=" * 70
        )

        print(
            " EXPERIMENT EXPORT TEST : SUCCESS"
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()