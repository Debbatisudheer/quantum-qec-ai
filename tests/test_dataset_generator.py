from dataset.generator import QECDatasetGenerator


def test_dataset_generation():

    print("\n===================================")
    print(" DATASET GENERATOR TEST")
    print("===================================")

    generator = QECDatasetGenerator(
        seed=42
    )

    dataset = generator.generate_dataset(
        num_samples=100
    )

    print("\nNumber of samples:")
    print(len(dataset))

    print("\nFirst 10 samples:")

    for sample in dataset[:10]:

        print("\n-------------------------------")

        print(
            f"Sample ID        : "
            f"{sample.sample_id}"
        )

        print(
            f"QEC code         : "
            f"{sample.qec_code}"
        )

        print(
            f"Physical qubits  : "
            f"{sample.num_qubits}"
        )

        print(
            f"Logical state    : "
            f"|{sample.logical_state}>L"
        )

        print(
            f"Original state   : "
            f"{sample.original_state}"
        )

        print(
            f"Error            : "
            f"{sample.error_description}"
        )

        print(
            f"Corrupted state  : "
            f"{sample.corrupted_state}"
        )

        print(
            f"Syndrome         : "
            f"{sample.syndrome}"
        )

        print(
            f"Target           : "
            f"{sample.target}"
        )

    # ---------------------------------
    # Validation
    # ---------------------------------

    assert len(dataset) == 100

    for sample in dataset:

        assert sample.qec_code == "bit_flip_3"

        assert sample.num_qubits == 3

        assert sample.logical_state in [
            0,
            1,
        ]

        assert sample.error_qubit in [
            None,
            0,
            1,
            2,
        ]

        assert sample.syndrome in [
            "00",
            "10",
            "11",
            "01",
        ]

        assert sample.target in [
            None,
            0,
            1,
            2,
        ]

    print("\n===================================")
    print(" DATASET TEST RESULT")
    print("===================================")

    print("Samples generated : 100")
    print("Schema validation : PASS")
    print("RESULT            : SUCCESS")


if __name__ == "__main__":
    test_dataset_generation()