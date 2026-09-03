from dataset.repeated_generator import (
    RepeatedSyndromeDatasetGenerator
)


def test_repeated_syndrome_generator():

    print("\n===================================")
    print(" REPEATED SYNDROME GENERATOR TEST")
    print("===================================")

    # -------------------------------------------------
    # 1. Create generator
    # -------------------------------------------------

    generator = (
        RepeatedSyndromeDatasetGenerator(
            rounds=5,
            measurement_noise_probability=0.10,
            seed=42
        )
    )

    print(
        "\nRounds configured: "
        f"{generator.rounds}"
    )

    print(
        "Measurement noise: 10%"
    )

    # -------------------------------------------------
    # 2. Generate dataset
    # -------------------------------------------------

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    print(
        "\nDataset generated: "
        f"{len(dataset)}"
    )

    assert len(dataset) == 1000

    print(
        "Dataset generation: PASS"
    )

    # -------------------------------------------------
    # 3. Check syndrome history
    # -------------------------------------------------

    for sample in dataset:

        history = (
            sample["syndrome_history"]
        )

        assert len(history) == 5

        for syndrome in history:

            assert syndrome in (
                "00",
                "10",
                "11",
                "01"
            )

    print(
        "Syndrome history length: PASS"
    )

    print(
        "Syndrome values: PASS"
    )

    # -------------------------------------------------
    # 4. Check perfect syndrome
    # -------------------------------------------------

    syndrome_map = {
        None: "00",
        0: "10",
        1: "11",
        2: "01",
    }

    for sample in dataset:

        expected_syndrome = (
            syndrome_map[
                sample["error_qubit"]
            ]
        )

        assert (
            sample["perfect_syndrome"]
            == expected_syndrome
        )

    print(
        "Perfect syndrome mapping: PASS"
    )

    # -------------------------------------------------
    # 5. Find a noisy history
    # -------------------------------------------------

    found_noisy_history = False

    for sample in dataset:

        perfect_syndrome = (
            sample["perfect_syndrome"]
        )

        history = (
            sample["syndrome_history"]
        )

        if any(
            syndrome != perfect_syndrome
            for syndrome in history
        ):

            found_noisy_history = True

            print(
                "\nExample noisy history:"
            )

            print(
                "Error              : "
                f"{sample['error_description']}"
            )

            print(
                "Perfect syndrome   : "
                f"{perfect_syndrome}"
            )

            print(
                "Syndrome history   : "
                f"{history}"
            )

            break

    assert found_noisy_history is True

    print(
        "Measurement noise effect: PASS"
    )

    # -------------------------------------------------
    # 6. Ground-truth separation
    # -------------------------------------------------

    for sample in dataset:

        history = (
            sample["syndrome_history"]
        )

        assert isinstance(
            history,
            list
        )

        assert len(history) == 5

        for syndrome in history:

            assert len(syndrome) == 2

    print(
        "Ground-truth separation: PASS"
    )

    # -------------------------------------------------
    # 7. Check all error classes
    # -------------------------------------------------

    error_classes = set(
        sample["error_qubit"]
        for sample in dataset
    )

    assert error_classes == {
        None,
        0,
        1,
        2
    }

    print(
        "All error classes present: PASS"
    )

    # -------------------------------------------------
    # 8. Final result
    # -------------------------------------------------

    print("\n===================================")
    print(" REPEATED SYNDROME RESULT")
    print("===================================")

    print(
        "Repeated rounds       : PASS"
    )

    print(
        "Noisy syndrome history: PASS"
    )

    print(
        "Perfect syndrome      : SEPARATED"
    )

    print(
        "Ground truth          : SEPARATED"
    )

    print(
        "RESULT                : SUCCESS"
    )


if __name__ == "__main__":
    test_repeated_syndrome_generator()