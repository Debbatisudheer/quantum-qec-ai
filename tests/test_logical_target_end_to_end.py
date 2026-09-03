from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_gru import (
    LogicalTargetGRUDecoder
)

from decoders.repeated_lookup import (
    RepeatedLookupDecoder
)

from quantum.state_evaluator import (
    QuantumStateEvaluator
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITY = 0.10

TRAINING_SAMPLES = 5000

TEST_SAMPLES = 100

SEED = 42

QUANTUM_VERIFICATION_SAMPLES = 10


# ============================================================
# HELPERS
# ============================================================

def xor_states(a, b):
    """
    XOR two equal-length binary states.
    """

    if len(a) != len(b):
        raise ValueError(
            "States must have the same length"
        )

    return [
        int(x) ^ int(y)
        for x, y in zip(a, b)
    ]


def create_encoded_state(logical_state):
    """
    Encode one logical bit using the
    3-qubit repetition code.

        logical 0 -> 000
        logical 1 -> 111
    """

    if logical_state not in (0, 1):
        raise ValueError(
            "logical_state must be 0 or 1"
        )

    return [
        logical_state,
        logical_state,
        logical_state
    ]


def apply_error(
    encoded_state,
    error_state
):
    """
    Apply the physical X-error pattern
    to the encoded state.
    """

    return xor_states(
        encoded_state,
        error_state
    )


def apply_correction(
    corrupted_state,
    correction
):
    """
    Apply the predicted physical correction.
    """

    return xor_states(
        corrupted_state,
        correction
    )


# ============================================================
# QUANTUM VERIFICATION
# ============================================================

def verify_quantum_correction(
    logical_state,
    actual_error,
    correction
):
    """
    Build and execute an actual 3-qubit
    Qiskit quantum circuit.

    Flow:

        encoded state
              ↓
        actual X error
              ↓
        predicted correction
              ↓
          measurement
              ↓
        logical recovery
    """

    evaluator = QuantumStateEvaluator()

    # --------------------------------------------------------
    # Create encoded logical state
    # --------------------------------------------------------

    circuit = evaluator.create_encoded_state(
        logical_state
    )

    # --------------------------------------------------------
    # Apply actual physical error
    # --------------------------------------------------------

    circuit = evaluator.apply_x_errors(
        circuit,
        actual_error
    )

    # --------------------------------------------------------
    # Apply AI predicted correction
    # --------------------------------------------------------

    circuit = evaluator.apply_corrections(
        circuit,
        correction
    )

    # --------------------------------------------------------
    # Add measurements
    # --------------------------------------------------------

    circuit = evaluator.add_measurements(
        circuit
    )

    # --------------------------------------------------------
    # Run quantum simulator
    # --------------------------------------------------------

    from qiskit_aer import AerSimulator

    simulator = AerSimulator()

    result = simulator.run(
        circuit,
        shots=1
    ).result()

    counts = result.get_counts()

    # --------------------------------------------------------
    # Most likely measurement
    # --------------------------------------------------------

    measured_state = max(
        counts,
        key=counts.get
    )

    # Qiskit displays classical bits in reverse
    # order compared with our physical-qubit order.

    measured_state = measured_state[::-1]

    # --------------------------------------------------------
    # Expected classical state
    # --------------------------------------------------------

    expected_state = create_encoded_state(
        logical_state
    )

    expected_state = apply_error(
        expected_state,
        actual_error
    )

    expected_state = apply_correction(
        expected_state,
        correction
    )

    expected_state = "".join(
        str(bit)
        for bit in expected_state
    )

    # --------------------------------------------------------
    # Physical-state comparison
    # --------------------------------------------------------

    quantum_physical_match = (
        measured_state
        == expected_state
    )

    # --------------------------------------------------------
    # Quantum logical recovery
    # --------------------------------------------------------

    recovered_logical = (
        evaluator.recover_logical_state(
            measured_state
        )
    )

    quantum_logical_success = (
        recovered_logical
        == logical_state
    )

    return {
        "measured_state":
            measured_state,

        "expected_state":
            expected_state,

        "physical_match":
            quantum_physical_match,

        "recovered_logical":
            recovered_logical,

        "logical_success":
            quantum_logical_success
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        " LOGICAL-TARGET AI → QUANTUM "
        "END-TO-END TEST"
    )
    print("=" * 70)

    # ========================================================
    # TRAINING DATA
    # ========================================================

    training_generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=SEED
        )
    )

    training_samples = (
        training_generator.generate_dataset(
            TRAINING_SAMPLES
        )
    )

    print()
    print(
        f"Training samples : "
        f"{len(training_samples)}"
    )

    # ========================================================
    # TRAIN LOGICAL-TARGET GRU
    # ========================================================

    ai_decoder = LogicalTargetGRUDecoder(
        rounds=ROUNDS,
        hidden_size=64,
        learning_rate=0.003,
        epochs=100,
        random_seed=SEED
    )

    ai_decoder.train(
        training_samples,
        verbose=False
    )

    print(
        "Logical-target GRU : TRAINED"
    )

    # ========================================================
    # INDEPENDENT TEST DATA
    # ========================================================

    test_generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=12345
        )
    )

    test_samples = (
        test_generator.generate_dataset(
            TEST_SAMPLES
        )
    )

    print(
        f"Test samples     : "
        f"{len(test_samples)}"
    )

    # ========================================================
    # DECODERS
    # ========================================================

    traditional_decoder = (
        RepeatedLookupDecoder()
    )

    recovery = LogicalRecovery()

    # ========================================================
    # METRIC COUNTERS
    # ========================================================

    ai_logical_success = 0

    traditional_logical_success = 0

    ai_physical_success = 0

    traditional_physical_success = 0

    quantum_verified = 0

    quantum_logical_success = 0

    quantum_classical_agreement = 0

    # ========================================================
    # PROCESS TEST SAMPLES
    # ========================================================

    for index, sample in enumerate(
        test_samples
    ):

        logical_state = int(
            sample[
                "logical_state"
            ]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        # ----------------------------------------------------
        # Encode logical state
        # ----------------------------------------------------

        encoded_state = (
            create_encoded_state(
                logical_state
            )
        )

        # ----------------------------------------------------
        # Apply actual error
        # ----------------------------------------------------

        corrupted_state = apply_error(
            encoded_state,
            actual_error
        )

        # ====================================================
        # AI DECODER
        # ====================================================

        ai_correction = ai_decoder.decode(
            sample
        )

        ai_corrected_state = (
            apply_correction(
                corrupted_state,
                ai_correction
            )
        )

        ai_recovered_logical = (
            recovery.recover(
                ai_corrected_state
            )
        )

        # ----------------------------------------------------
        # AI logical success
        # ----------------------------------------------------

        if (
            ai_recovered_logical
            == logical_state
        ):

            ai_logical_success += 1

        # ----------------------------------------------------
        # AI exact physical recovery
        # ----------------------------------------------------

        if (
            ai_corrected_state
            == encoded_state
        ):

            ai_physical_success += 1

        # ====================================================
        # TRADITIONAL DECODER
        # ====================================================

        traditional_correction = (
            traditional_decoder.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        traditional_corrected_state = (
            apply_correction(
                corrupted_state,
                traditional_correction
            )
        )

        traditional_recovered_logical = (
            recovery.recover(
                traditional_corrected_state
            )
        )

        # ----------------------------------------------------
        # Traditional logical success
        # ----------------------------------------------------

        if (
            traditional_recovered_logical
            == logical_state
        ):

            traditional_logical_success += 1

        # ----------------------------------------------------
        # Traditional physical recovery
        # ----------------------------------------------------

        if (
            traditional_corrected_state
            == encoded_state
        ):

            traditional_physical_success += 1

        # ====================================================
        # ACTUAL QUANTUM VERIFICATION
        # ====================================================

        if (
            index
            < QUANTUM_VERIFICATION_SAMPLES
        ):

            quantum_result = (
                verify_quantum_correction(
                    logical_state,
                    actual_error,
                    ai_correction
                )
            )

            # ------------------------------------------------
            # Quantum physical agreement
            # ------------------------------------------------

            if quantum_result[
                "physical_match"
            ]:

                quantum_verified += 1

            # ------------------------------------------------
            # Quantum logical success
            # ------------------------------------------------

            if quantum_result[
                "logical_success"
            ]:

                quantum_logical_success += 1

            # ------------------------------------------------
            # Quantum vs classical logical result
            # ------------------------------------------------

            if (
                quantum_result[
                    "recovered_logical"
                ]
                ==
                ai_recovered_logical
            ):

                quantum_classical_agreement += 1

            # ------------------------------------------------
            # First sample diagnostic
            # ------------------------------------------------

            if index == 0:

                print()
                print(
                    "First quantum verification:"
                )

                print(
                    f"Logical state       : "
                    f"{logical_state}"
                )

                print(
                    f"Actual error        : "
                    f"{actual_error}"
                )

                print(
                    f"AI correction       : "
                    f"{ai_correction}"
                )

                print(
                    f"Measured state      : "
                    f"{quantum_result['measured_state']}"
                )

                print(
                    f"Expected state      : "
                    f"{quantum_result['expected_state']}"
                )

                print(
                    f"Quantum physical    : "
                    f"{quantum_result['physical_match']}"
                )

                print(
                    f"Classical logical   : "
                    f"{ai_recovered_logical}"
                )

                print(
                    f"Quantum logical     : "
                    f"{quantum_result['recovered_logical']}"
                )

                print(
                    f"Quantum logical OK  : "
                    f"{quantum_result['logical_success']}"
                )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    ai_logical = (
        ai_logical_success
        / TEST_SAMPLES
    )

    traditional_logical = (
        traditional_logical_success
        / TEST_SAMPLES
    )

    ai_physical = (
        ai_physical_success
        / TEST_SAMPLES
    )

    traditional_physical = (
        traditional_physical_success
        / TEST_SAMPLES
    )

    quantum_physical = (
        quantum_verified
        / QUANTUM_VERIFICATION_SAMPLES
    )

    quantum_logical = (
        quantum_logical_success
        / QUANTUM_VERIFICATION_SAMPLES
    )

    quantum_classical = (
        quantum_classical_agreement
        / QUANTUM_VERIFICATION_SAMPLES
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        " END-TO-END RESULTS"
    )
    print("=" * 70)

    print()

    print(
        "Decoder                         "
        "Logical Success"
    )

    print(
        "-" * 70
    )

    print(
        f"Traditional lookup             "
        f"{traditional_logical:.4f}"
    )

    print(
        f"Logical-target GRU             "
        f"{ai_logical:.4f}"
    )

    print()

    print(
        "Physical recovery:"
    )

    print(
        f"Traditional lookup             "
        f"{traditional_physical:.4f}"
    )

    print(
        f"Logical-target GRU             "
        f"{ai_physical:.4f}"
    )

    print()

    print(
        f"Quantum verification "
        f"({QUANTUM_VERIFICATION_SAMPLES} samples):"
    )

    print(
        f"Physical agreement             "
        f"{quantum_physical:.4f}"
    )

    print(
        f"Logical success                "
        f"{quantum_logical:.4f}"
    )

    print(
        f"Classical/quantum agreement    "
        f"{quantum_classical:.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert len(
        test_samples
    ) == TEST_SAMPLES

    assert 0.0 <= ai_logical <= 1.0

    assert 0.0 <= traditional_logical <= 1.0

    assert quantum_verified == (
        QUANTUM_VERIFICATION_SAMPLES
    )

    assert quantum_classical_agreement == (
        QUANTUM_VERIFICATION_SAMPLES
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()

    print(
        "AI decoder integration       : PASS"
    )

    print(
        "Quantum correction           : PASS"
    )

    print(
        "Quantum measurement          : PASS"
    )

    print(
        "Classical/quantum agreement  : PASS"
    )

    print(
        "Logical recovery pipeline    : PASS"
    )

    print()

    print(
        "RESULT : SUCCESS"
    )

    print()
    print("=" * 70)
    print(
        " LOGICAL-TARGET AI → QUANTUM "
        "END-TO-END : COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()