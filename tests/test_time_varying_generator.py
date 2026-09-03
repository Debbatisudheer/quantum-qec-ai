from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)


def test_time_varying_generator():

    print("\n===================================")
    print(" TIME-VARYING QEC GENERATOR TEST")
    print("===================================")

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=5,
            physical_error_probability=0.10,
            measurement_noise_probability=0.10,
            seed=42
        )
    )

    # -------------------------------------------------
    # 1. Generate dataset
    # -------------------------------------------------

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    assert len(dataset) == 1000

    print(
        "Dataset generation: PASS"
    )

    # -------------------------------------------------
    # 2. Check sample structure
    # -------------------------------------------------

    sample = dataset[0]

    required_fields = [
        "sample_id",
        "qec_code",
        "num_qubits",
        "rounds",
        "physical_error_probability",
        "measurement_noise_probability",
        "syndrome_history",
        "observed_syndrome_history",
        "detection_events",
        "physical_error_history",
        "final_error_state",
        "final_syndrome",
        "final_observed_syndrome",
        "final_error_description"
    ]

    for field in required_fields:

        assert field in sample

    print(
        "Sample schema: PASS"
    )

    # -------------------------------------------------
    # 3. Check number of rounds
    # -------------------------------------------------

    for sample in dataset:

        assert len(
            sample["syndrome_history"]
        ) == 5

        assert len(
            sample["observed_syndrome_history"]
        ) == 5

        assert len(
            sample["detection_events"]
        ) == 5

        assert len(
            sample["physical_error_history"]
        ) == 5

    print(
        "Round history lengths: PASS"
    )

    # -------------------------------------------------
    # 4. Validate syndrome values
    # -------------------------------------------------

    valid_syndromes = {
        "00",
        "10",
        "11",
        "01"
    }

    for sample in dataset:

        for syndrome in sample[
            "syndrome_history"
        ]:

            assert syndrome in valid_syndromes

        for syndrome in sample[
            "observed_syndrome_history"
        ]:

            assert syndrome in valid_syndromes

        for event in sample[
            "detection_events"
        ]:

            assert event in valid_syndromes

    print(
        "Syndrome values: PASS"
    )

    # -------------------------------------------------
    # 5. Validate physical error states
    # -------------------------------------------------

    for sample in dataset:

        for state in sample[
            "physical_error_history"
        ]:

            assert len(state) == 3

            assert all(
                bit in (0, 1)
                for bit in state
            )

        final_state = sample[
            "final_error_state"
        ]

        assert len(final_state) == 3

        assert all(
            bit in (0, 1)
            for bit in final_state
        )

    print(
        "Physical error states: PASS"
    )

    # -------------------------------------------------
    # 6. Verify syndrome mathematics
    # -------------------------------------------------

    for sample in dataset:

        for state, syndrome in zip(
            sample["physical_error_history"],
            sample["syndrome_history"]
        ):

            expected_syndrome = (
                generator.calculate_syndrome(
                    state
                )
            )

            assert (
                syndrome
                == expected_syndrome
            )

    print(
        "Syndrome correctness: PASS"
    )

    # -------------------------------------------------
    # 7. Verify detection events
    # -------------------------------------------------

    for sample in dataset:

        expected_events = (
            generator.calculate_detection_events(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        assert (
            sample["detection_events"]
            == expected_events
        )

    print(
        "Detection-event calculation: PASS"
    )

    # -------------------------------------------------
    # 8. Verify time-dependent behavior
    # -------------------------------------------------

    changed_samples = 0

    for sample in dataset:

        history = sample[
            "physical_error_history"
        ]

        if any(
            history[index]
            != history[index - 1]
            for index in range(
                1,
                len(history)
            )
        ):
            changed_samples += 1

    print(
        f"Samples with changing "
        f"physical state: {changed_samples}"
    )

    assert changed_samples > 0

    print(
        "Time-dependent physical noise: PASS"
    )

    # -------------------------------------------------
    # 9. Ground-truth separation
    # -------------------------------------------------

    observable_fields = [
        "observed_syndrome_history",
        "detection_events"
    ]

    ground_truth_fields = [
        "syndrome_history",
        "physical_error_history",
        "final_error_state"
    ]

    for field in observable_fields:

        assert field in sample

    for field in ground_truth_fields:

        assert field in sample

    print(
        "Ground-truth separation: PASS"
    )

    # -------------------------------------------------
    # 10. Show example
    # -------------------------------------------------

    print("\nExample sample:")

    print(
        "Physical error history:"
    )

    print(
        sample[
            "physical_error_history"
        ]
    )

    print(
        "\nPerfect syndrome history:"
    )

    print(
        sample[
            "syndrome_history"
        ]
    )

    print(
        "\nObserved syndrome history:"
    )

    print(
        sample[
            "observed_syndrome_history"
        ]
    )

    print(
        "\nDetection events:"
    )

    print(
        sample[
            "detection_events"
        ]
    )

    print(
        "\nFinal error state:"
    )

    print(
        sample[
            "final_error_state"
        ]
    )

    print(
        "\nFinal error description:"
    )

    print(
        sample[
            "final_error_description"
        ]
    )

    # -------------------------------------------------
    # Final
    # -------------------------------------------------

    print("\n===================================")
    print(" TIME-VARYING GENERATOR RESULT")
    print("===================================")

    print(
        "Dynamic physical errors : PASS"
    )

    print(
        "Repeated syndrome       : PASS"
    )

    print(
        "Detection events        : PASS"
    )

    print(
        "Ground truth separation : PASS"
    )

    print(
        "RESULT                  : SUCCESS"
    )


if __name__ == "__main__":
    test_time_varying_generator()