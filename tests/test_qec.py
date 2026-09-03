from quantum.simulator import QuantumSimulator
from syndrome.extractor import SyndromeExtractor
from decoders.lookup import LookupDecoder
from correction.engine import CorrectionEngine
from qec.bit_flip_3 import BitFlipCode3


def run_qec_case(logical_state, error_qubit):
    """
    Run one complete 3-qubit QEC test case.

    Pipeline:

        Encode
          ↓
        X error
          ↓
        Syndrome
          ↓
        Decode
          ↓
        Correction
          ↓
        Final measurement
    """

    syndrome_extractor = SyndromeExtractor()
    decoder = LookupDecoder()
    correction_engine = CorrectionEngine()
    simulator = QuantumSimulator()

    # ---------------------------------
    # 1. Create syndrome circuit
    # ---------------------------------

    syndrome_circuit = (
        syndrome_extractor.create_syndrome_circuit(
            logical_state=logical_state,
            error_qubit=error_qubit
        )
    )

    # ---------------------------------
    # 2. Measure syndrome
    # ---------------------------------

    result = simulator.run(
        syndrome_circuit,
        shots=1
    )

    syndrome_counts = result.get_counts()

    measured_syndrome = list(
        syndrome_counts.keys()
    )[0]

    # Qiskit displays classical bits
    # from highest index to lowest.

    syndrome = measured_syndrome[::-1]

    # ---------------------------------
    # 3. Decode syndrome
    # ---------------------------------

    predicted_error_qubit = decoder.decode(
        syndrome
    )

    # ---------------------------------
    # 4. Create fresh correction circuit
    # ---------------------------------

    qec_code = BitFlipCode3()

    corrected_circuit = (
        qec_code.create_encoding_circuit(
            logical_state=logical_state
        )
    )

    # ---------------------------------
    # 5. Reproduce physical error
    # ---------------------------------

    if error_qubit is not None:
        corrected_circuit.x(error_qubit)

    # ---------------------------------
    # 6. Apply decoder correction
    # ---------------------------------

    corrected_circuit = correction_engine.apply(
        corrected_circuit,
        predicted_error_qubit
    )

    # ---------------------------------
    # 7. Measure physical qubits
    # ---------------------------------

    corrected_circuit.measure(
        [0, 1, 2],
        [0, 1, 2]
    )

    # ---------------------------------
    # 8. Execute corrected circuit
    # ---------------------------------

    result = simulator.run(
        corrected_circuit,
        shots=1
    )

    final_counts = result.get_counts()

    final_state = list(
        final_counts.keys()
    )[0]

    # ---------------------------------
    # 9. Expected logical state
    # ---------------------------------

    if logical_state == 0:
        expected_state = "000"
    else:
        expected_state = "111"

    # ---------------------------------
    # 10. Determine success
    # ---------------------------------

    success = (
        final_state == expected_state
    )

    return {
        "logical_state": logical_state,
        "error_qubit": error_qubit,
        "syndrome": syndrome,
        "predicted_error_qubit": predicted_error_qubit,
        "final_state": final_state,
        "expected_state": expected_state,
        "success": success,
    }


def test_all_single_qubit_errors():
    """
    Validate all single-qubit bit-flip cases.

    Tests:

        |0>L:
            no error
            X q0
            X q1
            X q2

        |1>L:
            no error
            X q0
            X q1
            X q2

    Total:
        8 test cases
    """

    print("\n===================================")
    print(" AUTOMATED QEC VALIDATION")
    print("===================================")

    logical_states = [0, 1]
    error_qubits = [None, 0, 1, 2]

    total_tests = 0
    passed_tests = 0

    # ---------------------------------
    # Expected syndrome table
    # ---------------------------------

    expected_syndromes = {
        None: "00",
        0: "10",
        1: "11",
        2: "01",
    }

    for logical_state in logical_states:

        for error_qubit in error_qubits:

            total_tests += 1

            result = run_qec_case(
                logical_state=logical_state,
                error_qubit=error_qubit
            )

            expected_syndrome = (
                expected_syndromes[error_qubit]
            )

            # ---------------------------------
            # Validate syndrome
            # ---------------------------------

            syndrome_correct = (
                result["syndrome"]
                == expected_syndrome
            )

            # ---------------------------------
            # Validate decoder
            # ---------------------------------

            decoder_correct = (
                result["predicted_error_qubit"]
                == error_qubit
            )

            # ---------------------------------
            # Validate recovery
            # ---------------------------------

            recovery_correct = (
                result["final_state"]
                == result["expected_state"]
            )

            test_passed = (
                syndrome_correct
                and decoder_correct
                and recovery_correct
            )

            if test_passed:
                passed_tests += 1

            # ---------------------------------
            # Display result
            # ---------------------------------

            if error_qubit is None:
                error_description = "No error"
            else:
                error_description = (
                    f"X on q{error_qubit}"
                )

            print("\n-----------------------------------")

            print(
                f"Logical state : |{logical_state}>L"
            )

            print(
                f"Error         : {error_description}"
            )

            print(
                f"Syndrome      : "
                f"{result['syndrome']}"
            )

            print(
                f"Expected      : "
                f"{expected_syndrome}"
            )

            if result["predicted_error_qubit"] is None:
                prediction = "No error"
            else:
                prediction = (
                    f"q{result['predicted_error_qubit']}"
                )

            print(
                f"Decoder       : {prediction}"
            )

            print(
                f"Final state   : "
                f"{result['final_state']}"
            )

            print(
                f"Expected final: "
                f"{result['expected_state']}"
            )

            if test_passed:
                print("RESULT        : PASS")
            else:
                print("RESULT        : FAIL")

    # ---------------------------------
    # Final summary
    # ---------------------------------

    print("\n===================================")
    print(" QEC VALIDATION SUMMARY")
    print("===================================")

    print(
        f"Tests passed : "
        f"{passed_tests}/{total_tests}"
    )

    print(
        f"Tests failed : "
        f"{total_tests - passed_tests}/{total_tests}"
    )

    if passed_tests == total_tests:
        print("\nALL QEC TESTS PASSED")
    else:
        print("\nQEC VALIDATION FAILED")

        raise AssertionError(
            "One or more QEC validation tests failed."
        )


if __name__ == "__main__":
    test_all_single_qubit_errors()