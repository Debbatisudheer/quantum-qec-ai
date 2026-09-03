from quantum.circuit import create_three_qubit_circuit
from quantum.simulator import QuantumSimulator

from qec.bit_flip_3 import BitFlipCode3
from noise.bit_flip import BitFlipNoise

from syndrome.extractor import SyndromeExtractor
from decoders.lookup import LookupDecoder
from correction.engine import CorrectionEngine


def test_basic_simulator():
    """
    Test the basic quantum simulator.
    """

    print("\n===================================")
    print(" BASIC QUANTUM SIMULATOR TEST")
    print("===================================")

    circuit = create_three_qubit_circuit()

    print("\nQuantum Circuit:")
    print(circuit)

    simulator = QuantumSimulator()

    result = simulator.run(
        circuit,
        shots=10
    )

    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)


def test_bit_flip_encoding(logical_state):
    """
    Test logical-state encoding.

    Logical states:

        |0>L = |000>
        |1>L = |111>
    """

    print("\n===================================")
    print(" 3-QUBIT BIT-FLIP CODE")
    print("===================================")

    qec_code = BitFlipCode3()

    print("\nLogical state:")

    print(
        qec_code.logical_state_description(
            logical_state
        )
    )

    circuit = qec_code.create_encoding_circuit(
        logical_state=logical_state
    )

    print("\nEncoding Circuit:")
    print(circuit)

    circuit = qec_code.add_measurements(
        circuit
    )

    simulator = QuantumSimulator()

    result = simulator.run(
        circuit,
        shots=10
    )

    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)


def test_complete_qec():
    """
    Original prototype QEC test.

    This test is kept for comparison.

    Pipeline:

        Encode
          ↓
        Noise
          ↓
        Manual physical state
          ↓
        Syndrome
          ↓
        Lookup decoder
          ↓
        Correction
          ↓
        Measure
    """

    print("\n===================================")
    print(" COMPLETE QEC TEST")
    print("===================================")

    # ---------------------------------
    # 1. Create QEC code
    # ---------------------------------

    qec_code = BitFlipCode3()

    # ---------------------------------
    # 2. Encode logical |1>
    # ---------------------------------

    circuit = qec_code.create_encoding_circuit(
        logical_state=1
    )

    print("\nOriginal logical state:")
    print("|1>L = |111>")

    # ---------------------------------
    # 3. Apply X error on q1
    # ---------------------------------

    noise = BitFlipNoise(
        probability=1.0
    )

    circuit = noise.apply(
        circuit,
        qubit=1
    )

    print("\nNoise:")
    print("X error on q1")

    # ---------------------------------
    # 4. Temporary prototype bridge
    # ---------------------------------

    corrupted_state = "101"

    print("\nCorrupted physical state:")
    print(corrupted_state)

    # ---------------------------------
    # 5. Extract syndrome
    # ---------------------------------

    syndrome_extractor = SyndromeExtractor()

    syndrome = syndrome_extractor.extract_from_string(
        corrupted_state
    )

    print("\nSyndrome:")
    print(syndrome)

    # ---------------------------------
    # 6. Decode syndrome
    # ---------------------------------

    decoder = LookupDecoder()

    error_qubit = decoder.decode(
        syndrome
    )

    print("\nDecoder prediction:")

    if error_qubit is None:
        print("No error detected")
    else:
        print(
            f"X error detected on q{error_qubit}"
        )

    # ---------------------------------
    # 7. Apply correction
    # ---------------------------------

    correction_engine = CorrectionEngine()

    circuit = correction_engine.apply(
        circuit,
        error_qubit
    )

    print("\nCorrection applied:")

    if error_qubit is None:
        print("No correction required")
    else:
        print(
            f"X correction applied to q{error_qubit}"
        )

    # ---------------------------------
    # 8. Measure corrected state
    # ---------------------------------

    circuit = qec_code.add_measurements(
        circuit
    )

    simulator = QuantumSimulator()

    result = simulator.run(
        circuit,
        shots=10
    )

    counts = result.get_counts()

    print("\nFinal measurement:")
    print(counts)

    print("\nExpected logical state:")
    print("|111>")


def test_quantum_syndrome_extraction():
    """
    Test REAL quantum ancilla-based syndrome extraction.

    Pipeline:

        Logical state
              ↓
           Encode
              ↓
         X error on q1
              ↓
      Quantum syndrome extraction
              ↓
        Ancilla measurement
              ↓
           Syndrome
              ↓
        Lookup decoder
    """

    print("\n===================================")
    print(" QUANTUM SYNDROME EXTRACTION TEST")
    print("===================================")

    syndrome_extractor = SyndromeExtractor()

    circuit = syndrome_extractor.create_syndrome_circuit(
        logical_state=1,
        error_qubit=1
    )

    print("\nQuantum Syndrome Circuit:")
    print(circuit)

    simulator = QuantumSimulator()

    result = simulator.run(
        circuit,
        shots=10
    )

    counts = result.get_counts()

    print("\nRaw Syndrome Measurement:")
    print(counts)

    decoder = LookupDecoder()

    print("\nDecoder Results:")

    for measured_syndrome in counts:

        print(
            f"Measured syndrome: "
            f"{measured_syndrome}"
        )

        actual_syndrome = measured_syndrome[::-1]

        print(
            f"Interpreted syndrome: "
            f"{actual_syndrome}"
        )

        error_qubit = decoder.decode(
            actual_syndrome
        )

        if error_qubit is None:

            print(
                "Decoder prediction: "
                "No error"
            )

        else:

            print(
                "Decoder prediction: "
                f"X error on q{error_qubit}"
            )

    print("\nExpected syndrome:")
    print("11")

    print("\nExpected decoder prediction:")
    print("X error on q1")


def test_full_quantum_qec():
    """
    REAL end-to-end quantum QEC test.

    Pipeline:

        Encode
          ↓
        Physical X error
          ↓
        Quantum syndrome extraction
          ↓
        Syndrome measurement
          ↓
        Lookup decoder
          ↓
        Correction
          ↓
        Final physical measurement
          ↓
        Logical recovery
    """

    print("\n===================================")
    print(" FULL QUANTUM QEC TEST")
    print("===================================")

    # ---------------------------------
    # 1. Configuration
    # ---------------------------------

    logical_state = 1
    error_qubit = 1

    print("\nConfiguration:")
    print("Logical state: |1>L")
    print("Physical qubits: 3")
    print("Error: X on q1")

    # ---------------------------------
    # 2. Create quantum circuit
    # ---------------------------------

    circuit = SyndromeExtractor().create_syndrome_circuit(
        logical_state=logical_state,
        error_qubit=error_qubit
    )

    print("\nInitial QEC Circuit:")
    print(circuit)

    # ---------------------------------
    # 3. Run syndrome measurement
    # ---------------------------------

    simulator = QuantumSimulator()

    result = simulator.run(
        circuit,
        shots=1
    )

    syndrome_counts = result.get_counts()

    print("\nRaw syndrome:")
    print(syndrome_counts)

    # ---------------------------------
    # 4. Get measured syndrome
    # ---------------------------------

    measured_syndrome = list(
        syndrome_counts.keys()
    )[0]

    # Qiskit displays classical bits
    # from highest index to lowest.
    #
    # c0 = S1
    # c1 = S2
    #
    # Therefore reverse the displayed string.

    syndrome = measured_syndrome[::-1]

    print("\nObserved syndrome:")
    print(syndrome)

    # ---------------------------------
    # 5. Decode
    # ---------------------------------

    decoder = LookupDecoder()

    predicted_error_qubit = decoder.decode(
        syndrome
    )

    print("\nDecoder prediction:")

    if predicted_error_qubit is None:

        print("No error detected")

    else:

        print(
            f"X error detected on "
            f"q{predicted_error_qubit}"
        )

    # ---------------------------------
    # 6. Build correction circuit
    # ---------------------------------
    #
    # We cannot modify the already executed
    # circuit and expect the previous measurement
    # to change.
    #
    # Therefore we create a fresh circuit:
    #
    # Encode
    #   ↓
    # Error
    #   ↓
    # Correction
    #   ↓
    # Measure
    #

    corrected_circuit = BitFlipCode3().create_encoding_circuit(
        logical_state=logical_state
    )

    # Reproduce the same physical error.
    corrected_circuit.x(error_qubit)

    print("\nError reproduced:")
    print(
        f"X error applied to q{error_qubit}"
    )

    # Apply decoder's correction.
    correction_engine = CorrectionEngine()

    corrected_circuit = correction_engine.apply(
        corrected_circuit,
        predicted_error_qubit
    )

    print("\nCorrection:")

    if predicted_error_qubit is None:

        print("No correction applied")

    else:

        print(
            f"X correction applied to "
            f"q{predicted_error_qubit}"
        )

    # ---------------------------------
    # 7. Measure physical qubits
    # ---------------------------------

    corrected_circuit.measure(
        [0, 1, 2],
        [0, 1, 2]
    )

    print("\nCorrection Circuit:")
    print(corrected_circuit)

    # ---------------------------------
    # 8. Execute corrected circuit
    # ---------------------------------

    result = simulator.run(
        corrected_circuit,
        shots=10
    )

    final_counts = result.get_counts()

    print("\nFinal physical measurement:")
    print(final_counts)

    # ---------------------------------
    # 9. Logical recovery
    # ---------------------------------

    expected_state = "111"

    if final_counts == {expected_state: 10}:

        print("\nLogical recovery:")
        print("SUCCESS")

        print("\nRecovered logical state:")
        print("|1>L = |111>")

    else:

        print("\nLogical recovery:")
        print("FAILED")

        print(
            "\nExpected:"
            f" {expected_state}"
        )

        print(
            "Actual:"
            f" {final_counts}"
        )

    # ---------------------------------
    # 10. Final result
    # ---------------------------------

    print("\n===================================")
    print(" QEC RESULT")
    print("===================================")

    print("Original logical state : |111>")
    print("Injected error         : X on q1")
    print(f"Observed syndrome      : {syndrome}")
    print(
        "Decoder prediction     : "
        f"q{predicted_error_qubit}"
    )
    print("Correction             : X on q1")
    print(f"Final state            : {final_counts}")

    if final_counts == {expected_state: 10}:

        print("\nRESULT: QEC SUCCESS")


def main():

    print("===================================")
    print(" AI QUANTUM ERROR CORRECTION SYSTEM")
    print("===================================")

    # ---------------------------------
    # Test 1
    # ---------------------------------

    test_basic_simulator()

    # ---------------------------------
    # Test 2
    # ---------------------------------

    test_bit_flip_encoding(
        logical_state=0
    )

    # ---------------------------------
    # Test 3
    # ---------------------------------

    test_bit_flip_encoding(
        logical_state=1
    )

    # ---------------------------------
    # Test 4
    # ---------------------------------

    test_complete_qec()

    # ---------------------------------
    # Test 5
    # ---------------------------------

    test_quantum_syndrome_extraction()

    # ---------------------------------
    # Test 6
    # ---------------------------------
    #
    # REAL END-TO-END QEC
    #

    test_full_quantum_qec()


if __name__ == "__main__":
    main()