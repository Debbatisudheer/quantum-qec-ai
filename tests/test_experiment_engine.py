from experiments.config import (
    ExperimentConfig
)

from experiments.engine import (
    ExperimentEngine
)

from experiments.result_storage import (
    ExperimentResultStorage
)


def main():

    print()
    print(
        "TESTING UNIFIED EXPERIMENT ENGINE"
    )
    print()

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        rounds=5,
        physical_noise_probability=0.10,
        measurement_noise_probability=0.10,
        training_samples=5000,
        test_samples=1000,
        decoder_type=(
            "logical_target_random_forest"
        ),
        random_forest_estimators=100,
        seed=42
    )

    config.validate()

    print(
        "ExperimentConfig validation : PASS"
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    storage = (
        ExperimentResultStorage()
    )

    print(
        "ResultStorage creation      : PASS"
    )

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = ExperimentEngine(
        config=config,
        storage=storage
    )

    print(
        "ExperimentEngine creation   : PASS"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = engine.run()

    assert result is not None

    # --------------------------------------------------------
    # RESULT VALIDATION
    # --------------------------------------------------------

    assert (
        result.training_samples
        == 5000
    )

    assert (
        result.test_samples
        == 1000
    )

    assert (
        result.decoder_type
        == "logical_target_random_forest"
    )

    assert (
        0.0
        <= result.exact_accuracy
        <= 1.0
    )

    assert (
        0.0
        <= result.physical_accuracy
        <= 1.0
    )

    assert (
        0.0
        <= result.bit_accuracy
        <= 1.0
    )

    assert (
        0.0
        <= result.logical_accuracy
        <= 1.0
    )

    print()
    print(
        "Experiment execution        : PASS"
    )

    # --------------------------------------------------------
    # STORAGE VALIDATION
    # --------------------------------------------------------

    assert storage.exists(
        result.experiment_id
    )

    print(
        "Result persisted            : PASS"
    )

    loaded = storage.load(
        result.experiment_id
    )

    assert (
        loaded["experiment_id"]
        == result.experiment_id
    )

    assert (
        loaded["decoder_type"]
        == result.decoder_type
    )

    assert (
        loaded["logical_accuracy"]
        == result.logical_accuracy
    )

    print(
        "Stored result validation    : PASS"
    )

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    stored_results = (
        storage.list_results()
    )

    assert (
        result.experiment_id
        in stored_results
    )

    print(
        "Stored result listing       : PASS"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " UNIFIED EXPERIMENT + STORAGE TEST : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()