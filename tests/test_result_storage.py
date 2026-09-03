import tempfile
from pathlib import Path

from experiments.config import (
    ExperimentConfig
)

from experiments.engine import (
    ExperimentEngine
)

from experiments.result import (
    ExperimentResult
)

from experiments.result_storage import (
    ExperimentResultStorage
)


def create_test_result():

    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        rounds=5,
        physical_noise_probability=0.10,
        measurement_noise_probability=0.10,
        training_samples=100,
        test_samples=50,
        decoder_type=(
            "logical_target_random_forest"
        ),
        random_forest_estimators=10,
        seed=42
    )

    return ExperimentResult(
        experiment_id="storage_test_001",

        config={
            "qec_code": "bit_flip_3",
            "num_qubits": 3,
            "rounds": 5,
            "physical_noise_probability": 0.10,
            "measurement_noise_probability": 0.10,
            "training_samples": 100,
            "test_samples": 50,
            "decoder_type": (
                "logical_target_random_forest"
            ),
            "random_forest_estimators": 10,
            "seed": 42,
        },

        training_samples=100,
        test_samples=50,

        logical_targets_learned=25,
        average_target_score=0.85,

        exact_accuracy=0.33,
        physical_accuracy=0.33,
        bit_accuracy=0.69,
        logical_accuracy=0.78,

        training_seconds=1.25,
        inference_seconds=0.05,
        samples_per_second=1000.0,

        decoder_type=(
            "logical_target_random_forest"
        )
    )


def main():

    print()
    print(
        "TESTING EXPERIMENT RESULT STORAGE"
    )
    print()

    # --------------------------------------------------------
    # TEMPORARY DIRECTORY
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        storage = ExperimentResultStorage(
            storage_directory=temp_dir
        )

        assert Path(
            temp_dir
        ).exists()

        print(
            "Storage creation       : PASS"
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = create_test_result()

        assert isinstance(
            result,
            ExperimentResult
        )

        print(
            "Test result creation   : PASS"
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        saved_path = storage.save(
            result
        )

        assert saved_path.exists()

        assert (
            saved_path.name
            == "storage_test_001.json"
        )

        print(
            "Result save            : PASS"
        )

        # ----------------------------------------------------
        # EXISTS
        # ----------------------------------------------------

        assert storage.exists(
            "storage_test_001"
        )

        print(
            "Result exists check    : PASS"
        )

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        assert (
            storage.count()
            == 1
        )

        print(
            "Result count           : PASS"
        )

        # ----------------------------------------------------
        # LIST
        # ----------------------------------------------------

        result_ids = (
            storage.list_results()
        )

        assert (
            result_ids
            == ["storage_test_001"]
        )

        print(
            "Result listing         : PASS"
        )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        loaded = storage.load(
            "storage_test_001"
        )

        assert isinstance(
            loaded,
            dict
        )

        assert (
            loaded["experiment_id"]
            == "storage_test_001"
        )

        assert (
            loaded["logical_accuracy"]
            == 0.78
        )

        assert (
            loaded["decoder_type"]
            == "logical_target_random_forest"
        )

        print(
            "Result load            : PASS"
        )

        # ----------------------------------------------------
        # LOAD ALL
        # ----------------------------------------------------

        all_results = (
            storage.load_all()
        )

        assert (
            len(all_results)
            == 1
        )

        assert (
            all_results[0][
                "experiment_id"
            ]
            == "storage_test_001"
        )

        print(
            "Load all results       : PASS"
        )

        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        deleted = storage.delete(
            "storage_test_001"
        )

        assert deleted is True

        assert not storage.exists(
            "storage_test_001"
        )

        assert (
            storage.count()
            == 0
        )

        print(
            "Result deletion        : PASS"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " EXPERIMENT RESULT STORAGE TEST : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()