from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.state_evaluator import (
    QuantumStateEvaluator
)


def run_quantum_trial(
    evaluator,
    backend,
    logical_state,
    actual_error_state,
    predicted_error_state,
    shots=100
):
    """
    Run one complete quantum QEC trial.

    Flow:

        logical state
            ↓
        encode
            ↓
        physical errors
            ↓
        AI correction
            ↓
        measurement
            ↓
        logical recovery
    """

    circuit = evaluator.create_encoded_state(
        logical_state
    )

    circuit = evaluator.apply_x_errors(
        circuit,
        actual_error_state
    )

    circuit = evaluator.apply_corrections(
        circuit,
        predicted_error_state
    )

    circuit = evaluator.add_measurements(
        circuit
    )

    compiled = transpile(
        circuit,
        backend
    )

    result = backend.run(
        compiled,
        shots=shots
    ).result()

    counts = result.get_counts()

    measured_state = max(
        counts,
        key=counts.get
    )

    # Qiskit classical-bit ordering can appear
    # reversed relative to qubit numbering.
    #
    # Reverse the displayed bitstring so that
    # the string follows q0, q1, q2 ordering.

    measured_state = measured_state[::-1]

    recovered_logical_state = (
        evaluator.recover_logical_state(
            measured_state
        )
    )

    success = (
        recovered_logical_state
        == logical_state
    )

    return {
        "logical_state":
            logical_state,

        "actual_error_state":
            actual_error_state,

        "predicted_error_state":
            predicted_error_state,

        "counts":
            counts,

        "measured_state":
            measured_state,

        "recovered_logical_state":
            recovered_logical_state,

        "logical_success":
            success
    }


def test_quantum_end_to_end():

    print("\n===================================")
    print(" REAL QUANTUM END-TO-END QEC TEST")
    print("===================================")

    evaluator = QuantumStateEvaluator()

    backend = AerSimulator()

    # ===================================
    # TEST 1
    # ===================================

    print(
        "\nTEST 1 — LOGICAL 0"
    )

    result = run_quantum_trial(
        evaluator=evaluator,
        backend=backend,
        logical_state=0,
        actual_error_state=[1, 0, 1],
        predicted_error_state=[1, 0, 1],
        shots=100
    )

    print(
        f"Actual error       : "
        f"{result['actual_error_state']}"
    )

    print(
        f"Predicted correction: "
        f"{result['predicted_error_state']}"
    )

    print(
        f"Measured state     : "
        f"{result['measured_state']}"
    )

    print(
        f"Recovered logical  : "
        f"{result['recovered_logical_state']}"
    )

    print(
        f"Logical success    : "
        f"{result['logical_success']}"
    )

    assert (
        result["measured_state"]
        == "000"
    )

    assert (
        result["recovered_logical_state"]
        == 0
    )

    assert (
        result["logical_success"]
        is True
    )

    print(
        "Logical 0 QEC: PASS"
    )

    # ===================================
    # TEST 2
    # ===================================

    print(
        "\nTEST 2 — LOGICAL 1"
    )

    result = run_quantum_trial(
        evaluator=evaluator,
        backend=backend,
        logical_state=1,
        actual_error_state=[1, 0, 1],
        predicted_error_state=[1, 0, 1],
        shots=100
    )

    print(
        f"Actual error       : "
        f"{result['actual_error_state']}"
    )

    print(
        f"Predicted correction: "
        f"{result['predicted_error_state']}"
    )

    print(
        f"Measured state     : "
        f"{result['measured_state']}"
    )

    print(
        f"Recovered logical  : "
        f"{result['recovered_logical_state']}"
    )

    print(
        f"Logical success    : "
        f"{result['logical_success']}"
    )

    assert (
        result["measured_state"]
        == "111"
    )

    assert (
        result["recovered_logical_state"]
        == 1
    )

    assert (
        result["logical_success"]
        is True
    )

    print(
        "Logical 1 QEC: PASS"
    )

    # ===================================
    # TEST 3
    # ===================================

    print(
        "\nTEST 3 — PARTIAL CORRECTION"
    )

    result = run_quantum_trial(
        evaluator=evaluator,
        backend=backend,
        logical_state=0,
        actual_error_state=[1, 0, 0],
        predicted_error_state=[0, 0, 0],
        shots=100
    )

    print(
        f"Actual error       : "
        f"{result['actual_error_state']}"
    )

    print(
        f"Predicted correction: "
        f"{result['predicted_error_state']}"
    )

    print(
        f"Measured state     : "
        f"{result['measured_state']}"
    )

    print(
        f"Recovered logical  : "
        f"{result['recovered_logical_state']}"
    )

    print(
        f"Logical success    : "
        f"{result['logical_success']}"
    )

    # One remaining physical error is
    # tolerated by the 3-qubit repetition code.

    assert (
        result["measured_state"]
        == "100"
    )

    assert (
        result["recovered_logical_state"]
        == 0
    )

    assert (
        result["logical_success"]
        is True
    )

    print(
        "Logical protection with "
        "one remaining error: PASS"
    )

    # ===================================
    # TEST 4
    # ===================================

    print(
        "\nTEST 4 — TOO MANY ERRORS"
    )

    result = run_quantum_trial(
        evaluator=evaluator,
        backend=backend,
        logical_state=0,
        actual_error_state=[1, 1, 0],
        predicted_error_state=[0, 0, 0],
        shots=100
    )

    print(
        f"Actual error       : "
        f"{result['actual_error_state']}"
    )

    print(
        f"Predicted correction: "
        f"{result['predicted_error_state']}"
    )

    print(
        f"Measured state     : "
        f"{result['measured_state']}"
    )

    print(
        f"Recovered logical  : "
        f"{result['recovered_logical_state']}"
    )

    print(
        f"Logical success    : "
        f"{result['logical_success']}"
    )

    assert (
        result["measured_state"]
        == "110"
    )

    assert (
        result["recovered_logical_state"]
        == 1
    )

    assert (
        result["logical_success"]
        is False
    )

    print(
        "Logical failure detection: PASS"
    )

    # ===================================
    # FINAL
    # ===================================

    print(
        "\n==================================="
    )

    print(
        " QUANTUM END-TO-END RESULT"
    )

    print(
        "==================================="
    )

    print(
        "Quantum encoding       : PASS"
    )

    print(
        "Physical X errors      : PASS"
    )

    print(
        "AI correction interface: PASS"
    )

    print(
        "Quantum measurement    : PASS"
    )

    print(
        "Logical recovery       : PASS"
    )

    print(
        "Logical failure test   : PASS"
    )

    print(
        "RESULT                 : SUCCESS"
    )


if __name__ == "__main__":
    test_quantum_end_to_end()