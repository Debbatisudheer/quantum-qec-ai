from dataset.noisy_generator import (
    NoisyQECDatasetGenerator
)


def test_noisy_dataset():

    print("\n===================================")
    print(" NOISY SYNDROME DATASET TEST")
    print("===================================")

    # -------------------------------------------------
    # 1. Generate dataset with 10% measurement noise
    # -------------------------------------------------

    generator = NoisyQECDatasetGenerator(
        measurement_noise_probability=0.10,
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=1000
    )

    print(
        f"\nDataset generated: "
        f"{len(dataset)}"
    )

    assert len(dataset) == 1000

    print(
        "Dataset size: PASS"
    )

    # -------------------------------------------------
    # 2. Check required fields
    # -------------------------------------------------

    required_fields = [
        "sample_id",
        "qec_code",
        "num_qubits",
        "logical_state",
        "original_state",
        "corrupted_state",
        "error_type",
        "error_qubit",
        "error_description",
        "perfect_syndrome",
        "observed_syndrome",
        "target",
    ]

    for sample in dataset:

        for field in required_fields:

            assert field in sample

    print(
        "Dataset schema: PASS"
    )

    # -------------------------------------------------
    # 3. Check syndrome values
    # -------------------------------------------------

    valid_syndromes = {
        "00",
        "10",
        "11",
        "01"
    }

    for sample in dataset:

        assert (
            sample["perfect_syndrome"]
            in valid_syndromes
        )

        assert (
            sample["observed_syndrome"]
            in valid_syndromes
        )

    print(
        "Syndrome values: PASS"
    )

    # -------------------------------------------------
    # 4. Check ground truth mapping
    # -------------------------------------------------

    expected_mapping = {
        None: "00",
        0: "10",
        1: "11",
        2: "01",
    }

    for sample in dataset:

        error_qubit = sample[
            "error_qubit"
        ]

        assert (
            sample["perfect_syndrome"]
            == expected_mapping[
                error_qubit
            ]
        )

    print(
        "Perfect syndrome mapping: PASS"
    )

    # -------------------------------------------------
    # 5. Check that noise actually changes
    #    some observations
    # -------------------------------------------------

    changed_samples = 0

    for sample in dataset:

        if (
            sample["perfect_syndrome"]
            != sample["observed_syndrome"]
        ):
            changed_samples += 1

    print(
        f"Samples affected by measurement noise: "
        f"{changed_samples}"
    )

    assert changed_samples > 0

    print(
        "Measurement noise effect: PASS"
    )

    # -------------------------------------------------
    # 6. Check ground truth separation
    # -------------------------------------------------

    for sample in dataset:

        assert (
            sample["perfect_syndrome"]
            == expected_mapping[
                sample["error_qubit"]
            ]
        )

    print(
        "Ground truth separation: PASS"
    )

    # -------------------------------------------------
    # 7. Show examples
    # -------------------------------------------------

    print(
        "\nExample noisy samples:"
    )

    shown = 0

    for sample in dataset:

        if (
            sample["perfect_syndrome"]
            != sample["observed_syndrome"]
        ):

            print(
                f"  Error: "
                f"{sample['error_description']}"
            )

            print(
                f"  Perfect syndrome : "
                f"{sample['perfect_syndrome']}"
            )

            print(
                f"  Observed syndrome: "
                f"{sample['observed_syndrome']}"
            )

            print()

            shown += 1

            if shown == 5:
                break

    # -------------------------------------------------
    # 8. Final result
    # -------------------------------------------------

    print("===================================")
    print(" NOISY DATASET RESULT")
    print("===================================")

    print(
        f"Samples generated : "
        f"{len(dataset)}"
    )

    print(
        f"Noise probability : "
        f"10%"
    )

    print(
        f"Changed samples   : "
        f"{changed_samples}"
    )

    print(
        "Ground truth      : SEPARATED"
    )

    print(
        "Observed input     : AVAILABLE"
    )

    print(
        "RESULT             : SUCCESS"
    )


if __name__ == "__main__":
    test_noisy_dataset()