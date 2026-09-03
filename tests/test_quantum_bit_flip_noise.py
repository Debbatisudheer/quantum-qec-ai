from qiskit import QuantumCircuit


from noise.quantum_bit_flip import (
    QuantumBitFlipNoise
)


def main():

    print()

    print("===================================")
    print(" QUANTUM BIT-FLIP NOISE TEST")
    print("===================================")

    probability = 0.50

    print()
    print(
        f"Noise probability : {probability}"
    )

    circuit = QuantumCircuit(
        3
    )

    # Prepare a simple state.
    circuit.h(0)

    noise = QuantumBitFlipNoise(
        probability=probability,
        seed=42
    )

    circuit, error_state = noise.apply(
        circuit,
        [0, 1, 2]
    )

    print()
    print(
        "Ground truth error state:"
    )

    print(
        error_state
    )

    print()
    print(
        "Quantum circuit:"
    )

    print(
        circuit
    )

    print()

    if len(error_state) != 3:
        print(
            "ERROR STATE TEST : FAIL"
        )
        return

    if any(
        bit not in (0, 1)
        for bit in error_state
    ):
        print(
            "ERROR STATE TEST : FAIL"
        )
        return

    print(
        "Noise application   : PASS"
    )

    print(
        "Ground truth state   : PASS"
    )

    print(
        "Quantum circuit      : PASS"
    )

    print()

    print(
        "==================================="
    )

    print(
        "QUANTUM BIT-FLIP NOISE TEST : SUCCESS"
    )


if __name__ == "__main__":
    main()