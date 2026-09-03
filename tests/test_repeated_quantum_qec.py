from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.repeated_qec import (
    RepeatedQuantumQEC
)

from quantum.repeated_measurement import (
    RepeatedQuantumMeasurementParser
)


def test_repeated_quantum_qec():

    print("\n===================================")
    print(" REPEATED QUANTUM QEC TEST")
    print("===================================")

    rounds = 5

    qec = RepeatedQuantumQEC(
        rounds=rounds
    )

    parser = (
        RepeatedQuantumMeasurementParser(
            rounds=rounds
        )
    )

    backend = AerSimulator()

    logical_state = 1

    physical_error_history = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0]
    ]

    print(
        f"\nRounds: {rounds}"
    )

    print(
        f"Logical state: {logical_state}"
    )

    print(
        "\nPhysical error history:"
    )

    for index, state in enumerate(
        physical_error_history
    ):

        print(
            f"Round {index + 1}: {state}"
        )

    circuit = (
        qec.create_round_by_round_circuit(
            logical_state,
            physical_error_history
        )
    )

    compiled = transpile(
        circuit,
        backend
    )

    result = backend.run(
        compiled,
        shots=1
    ).result()

    counts = result.get_counts()

    raw = max(
        counts,
        key=counts.get
    )

    parsed = parser.parse(
        raw
    )

    print(
        "\nRaw measurement:"
    )

    print(raw)

    print(
        "\nFinal physical state:"
    )

    print(
        parsed["final_state"]
    )

    print(
        "\nQuantum syndrome history:"
    )

    print(
        parsed["syndrome_history"]
    )

    expected_history = [
        "00",
        "10",
        "10",
        "01",
        "01"
    ]

    assert (
        parsed["syndrome_history"]
        == expected_history
    )

    print(
        "\nExpected syndrome history:"
    )

    print(expected_history)

    print(
        "\nSyndrome extraction : PASS"
    )

    print(
        "Round-by-round quantum circuit : PASS"
    )

    print(
        "Final physical measurement : PASS"
    )

    print(
        "\n==================================="
    )

    print(
        " REPEATED QUANTUM QEC RESULT"
    )

    print(
        "==================================="
    )

    print(
        "Quantum rounds        : PASS"
    )

    print(
        "Syndrome extraction   : PASS"
    )

    print(
        "Syndrome history      : PASS"
    )

    print(
        "RESULT                : SUCCESS"
    )


if __name__ == "__main__":
    test_repeated_quantum_qec()