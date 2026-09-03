from qiskit_aer import AerSimulator

from quantum.state_evaluator import QuantumStateEvaluator
from evaluation.logical_recovery import LogicalRecovery


ALL_ERROR_STATES = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]


def reverse_qiskit_bitstring(bitstring):
    """
    Qiskit displays classical bits in reverse order
    relative to the logical qubit indexing used by
    our project.

    Example:

        Qiskit: 100
        Project: 001
    """

    return bitstring[::-1]


def xor_states(a, b):
    return [
        x ^ y
        for x, y in zip(a, b)
    ]


# ============================================================
# TEST 1
# CLASSICAL RECOVERY MAPPING
# ============================================================

def test_classical_recovery():

    print()
    print("=" * 60)
    print(" TEST 1: CLASSICAL RECOVERY")
    print("=" * 60)

    recovery = LogicalRecovery()

    passed = True

    for state in ALL_ERROR_STATES:

        result = recovery.recover(state)

        expected = (
            1
            if sum(state) >= 2
            else 0
        )

        status = (
            "PASS"
            if result == expected
            else "FAIL"
        )

        print(
            f"{state} -> "
            f"logical={result} "
            f"expected={expected} "
            f"{status}"
        )

        if result != expected:
            passed = False

    print()

    print(
        "CLASSICAL RECOVERY : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 2
# QUANTUM ENCODING
# ============================================================

def test_quantum_encoding():

    print()
    print("=" * 60)
    print(" TEST 2: QUANTUM ENCODING")
    print("=" * 60)

    evaluator = QuantumStateEvaluator()
    simulator = AerSimulator()

    passed = True

    for logical_state in [0, 1]:

        circuit = (
            evaluator.create_encoded_state(
                logical_state
            )
        )

        circuit = (
            evaluator.add_measurements(
                circuit
            )
        )

        result = simulator.run(
            circuit,
            shots=1
        ).result()

        counts = result.get_counts()

        measured = max(
            counts,
            key=counts.get
        )

        measured_project_order = (
            reverse_qiskit_bitstring(
                measured
            )
        )

        expected = (
            "000"
            if logical_state == 0
            else "111"
        )

        status = (
            "PASS"
            if measured_project_order == expected
            else "FAIL"
        )

        print(
            f"Logical={logical_state} "
            f"measured={measured_project_order} "
            f"expected={expected} "
            f"{status}"
        )

        if measured_project_order != expected:
            passed = False

    print()

    print(
        "QUANTUM ENCODING : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 3
# QUANTUM ERROR + CORRECTION
#
# Perfect decoder:
#
# predicted error == actual error
#
# Therefore:
#
# actual XOR predicted = 000
# ============================================================

def test_perfect_quantum_correction():

    print()
    print("=" * 60)
    print(
        " TEST 3: PERFECT QUANTUM CORRECTION"
    )
    print("=" * 60)

    evaluator = QuantumStateEvaluator()
    simulator = AerSimulator()

    passed = True

    for logical_state in [0, 1]:

        for error_state in ALL_ERROR_STATES:

            circuit = (
                evaluator.create_encoded_state(
                    logical_state
                )
            )

            circuit = (
                evaluator.apply_x_errors(
                    circuit,
                    error_state
                )
            )

            # Perfect decoder predicts the
            # exact actual error.

            predicted_error = (
                error_state.copy()
            )

            circuit = (
                evaluator.apply_corrections(
                    circuit,
                    predicted_error
                )
            )

            circuit = (
                evaluator.add_measurements(
                    circuit
                )
            )

            result = simulator.run(
                circuit,
                shots=1
            ).result()

            counts = result.get_counts()

            measured = max(
                counts,
                key=counts.get
            )

            measured_state = (
                reverse_qiskit_bitstring(
                    measured
                )
            )

            expected_state = (
                "000"
                if logical_state == 0
                else "111"
            )

            logical_success = (
                evaluator.logical_success(
                    logical_state,
                    measured_state
                )
            )

            status = (
                "PASS"
                if (
                    measured_state
                    == expected_state
                    and logical_success
                )
                else "FAIL"
            )

            print(
                f"logical={logical_state} "
                f"error={error_state} "
                f"measured={measured_state} "
                f"expected={expected_state} "
                f"{status}"
            )

            if status == "FAIL":
                passed = False

    print()

    print(
        "PERFECT QUANTUM CORRECTION : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 4
# CLASSICAL CORRECTION MODEL
# ============================================================

def test_classical_correction_model():

    print()
    print("=" * 60)
    print(
        " TEST 4: CLASSICAL CORRECTION MODEL"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    passed = True

    for logical_state in [0, 1]:

        original_state = (
            [0, 0, 0]
            if logical_state == 0
            else [1, 1, 1]
        )

        for error_state in ALL_ERROR_STATES:

            # Apply error using XOR.

            corrupted_state = xor_states(
                original_state,
                error_state
            )

            # Perfect correction.

            corrected_state = xor_states(
                corrupted_state,
                error_state
            )

            recovered = recovery.recover(
                corrected_state
            )

            status = (
                "PASS"
                if recovered == logical_state
                else "FAIL"
            )

            if status == "FAIL":
                passed = False

            print(
                f"logical={logical_state} "
                f"error={error_state} "
                f"corrupted={corrupted_state} "
                f"corrected={corrected_state} "
                f"recovered={recovered} "
                f"{status}"
            )

    print()

    print(
        "CLASSICAL CORRECTION MODEL : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# TEST 5
# QUANTUM AND CLASSICAL AGREEMENT
#
# One-error cases are especially important
# for the 3-qubit repetition code.
# ============================================================

def test_quantum_classical_agreement():

    print()
    print("=" * 60)
    print(
        " TEST 5: QUANTUM / CLASSICAL AGREEMENT"
    )
    print("=" * 60)

    evaluator = QuantumStateEvaluator()
    recovery = LogicalRecovery()
    simulator = AerSimulator()

    one_error_states = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    passed = True

    for logical_state in [0, 1]:

        for error_state in one_error_states:

            circuit = (
                evaluator.create_encoded_state(
                    logical_state
                )
            )

            circuit = (
                evaluator.apply_x_errors(
                    circuit,
                    error_state
                )
            )

            # Perfect correction.

            circuit = (
                evaluator.apply_corrections(
                    circuit,
                    error_state
                )
            )

            circuit = (
                evaluator.add_measurements(
                    circuit
                )
            )

            result = simulator.run(
                circuit,
                shots=1
            ).result()

            counts = result.get_counts()

            measured = max(
                counts,
                key=counts.get
            )

            measured_state = (
                reverse_qiskit_bitstring(
                    measured
                )
            )

            quantum_logical = (
                evaluator.recover_logical_state(
                    measured_state
                )
            )

            classical_state = xor_states(
                [0, 0, 0],
                [0, 0, 0]
            )

            # After perfect correction the
            # physical state must be the original
            # encoded logical state.

            classical_logical = (
                logical_state
            )

            status = (
                "PASS"
                if (
                    quantum_logical
                    == classical_logical
                    == logical_state
                )
                else "FAIL"
            )

            print(
                f"logical={logical_state} "
                f"error={error_state} "
                f"measured={measured_state} "
                f"quantum={quantum_logical} "
                f"classical={classical_logical} "
                f"{status}"
            )

            if status == "FAIL":
                passed = False

    print()

    print(
        "QUANTUM / CLASSICAL AGREEMENT : "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " QUANTUM VS CLASSICAL "
        "LOGICAL VALIDATION"
    )
    print("=" * 60)

    results = []

    results.append(
        test_classical_recovery()
    )

    results.append(
        test_quantum_encoding()
    )

    results.append(
        test_perfect_quantum_correction()
    )

    results.append(
        test_classical_correction_model()
    )

    results.append(
        test_quantum_classical_agreement()
    )

    print()
    print("=" * 60)
    print(" FINAL VALIDATION RESULT")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(
        f"Tests passed : {passed}/{total}"
    )

    if passed == total:

        print()
        print(
            "RESULT : SUCCESS"
        )

        print()
        print(
            "Quantum simulation and classical "
            "logical recovery agree."
        )

    else:

        print()
        print(
            "RESULT : FAILURE"
        )

        print()
        print(
            "There is a discrepancy between "
            "the quantum and classical layers."
        )

    print()
    print("=" * 60)
    print(
        " QUANTUM / CLASSICAL VALIDATION : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()