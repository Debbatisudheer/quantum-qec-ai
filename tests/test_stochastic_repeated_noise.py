from noise.stochastic_repeated_noise import (
    StochasticRepeatedBitFlipNoise
)

from quantum.repeated_qec import (
    RepeatedQuantumQEC
)


def main():

    print()

    print("===================================")
    print(" STOCHASTIC REPEATED QEC TEST")
    print("===================================")

    rounds = 5
    probability = 0.30
    seed = 42

    print()
    print(
        f"Rounds            : {rounds}"
    )

    print(
        f"Noise probability : {probability}"
    )

    print(
        f"Random seed       : {seed}"
    )

    noise = StochasticRepeatedBitFlipNoise(
        rounds=rounds,
        probability=probability,
        seed=seed
    )

    error_history = (
        noise.generate_error_history()
    )

    print()
    print(
        "Stochastic physical error history:"
    )

    for index, state in enumerate(
        error_history,
        start=1
    ):
        print(
            f"Round {index}: {state}"
        )

    # Build the real repeated quantum QEC circuit.
    quantum_qec = RepeatedQuantumQEC(
        rounds=rounds
    )

    circuit = (
        quantum_qec.create_round_by_round_circuit(
            logical_state=1,
            physical_error_history=error_history
        )
    )

    print()
    print(
        "Quantum QEC circuit created : PASS"
    )

    print()
    print(
        "Quantum circuit:"
    )

    print(
        circuit
    )

    print()

    # Basic validation.
    if len(error_history) != rounds:
        print(
            "ERROR HISTORY TEST : FAIL"
        )
        return

    if any(
        len(state) != 3
        for state in error_history
    ):
        print(
            "ERROR STATE SIZE TEST : FAIL"
        )
        return

    if any(
        bit not in (0, 1)
        for state in error_history
        for bit in state
    ):
        print(
            "ERROR STATE VALUE TEST : FAIL"
        )
        return

    print(
        "Error history       : PASS"
    )

    print(
        "Round count         : PASS"
    )

    print(
        "Error-state format  : PASS"
    )

    print(
        "Quantum QEC bridge  : PASS"
    )

    print()
    print(
        "==================================="
    )

    print(
        "STOCHASTIC REPEATED QEC TEST : SUCCESS"
    )


if __name__ == "__main__":
    main()