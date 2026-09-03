from qiskit import QuantumCircuit


class SyndromeExtractor:
    """
    Quantum syndrome extraction for the 3-qubit bit-flip code.

    Physical qubits:

        q0
        q1
        q2

    Ancilla qubits:

        q3 -> measures S1 = Z0 Z1
        q4 -> measures S2 = Z1 Z2

    Syndrome mapping:

        00 -> No error
        10 -> X on q0
        11 -> X on q1
        01 -> X on q2
    """

    def __init__(self):
        self.stabilizers = [
            "ZZI",
            "IZZ",
        ]

    def create_syndrome_circuit(
        self,
        logical_state=1,
        error_qubit=None
    ):
        """
        Create a complete quantum circuit containing:

            Logical state
                  ↓
              Encoding
                  ↓
             Physical error
                  ↓
        Quantum syndrome extraction
                  ↓
          Syndrome measurement

        Args:
            logical_state: 0 or 1
            error_qubit: None, 0, 1, or 2

        Returns:
            QuantumCircuit
        """

        # ---------------------------------
        # Validate logical state
        # ---------------------------------

        if logical_state not in (0, 1):
            raise ValueError(
                "logical_state must be 0 or 1"
            )

        # ---------------------------------
        # Validate error qubit
        # ---------------------------------

        if error_qubit not in (None, 0, 1, 2):
            raise ValueError(
                "error_qubit must be None, 0, 1, or 2"
            )

        # ---------------------------------
        # Create circuit
        # ---------------------------------
        #
        # q0, q1, q2 = physical qubits
        # q3         = syndrome ancilla S1
        # q4         = syndrome ancilla S2
        #
        # c0         = S1
        # c1         = S2
        #

        circuit = QuantumCircuit(5, 2)

        # ---------------------------------
        # 1. Prepare logical state
        # ---------------------------------

        if logical_state == 1:
            circuit.x(0)

        # ---------------------------------
        # 2. Encode
        # ---------------------------------
        #
        # q0 controls q1
        # q0 controls q2
        #
        # |000> -> |000>
        # |100> -> |111>
        #

        circuit.cx(0, 1)
        circuit.cx(0, 2)

        # ---------------------------------
        # 3. Apply physical X error
        # ---------------------------------

        if error_qubit is not None:
            circuit.x(error_qubit)

        # ---------------------------------
        # 4. Measure S1 = Z0 Z1
        # ---------------------------------
        #
        # q0 ─────●────────
        #          │
        # q3 ─────X────────
        #
        # q1 ─────●────────
        #          │
        # q3 ─────X────────
        #
        # q3 contains:
        #
        # q0 XOR q1
        #
        # Therefore q3 represents S1.
        #

        circuit.cx(0, 3)
        circuit.cx(1, 3)

        # ---------------------------------
        # 5. Measure S2 = Z1 Z2
        # ---------------------------------
        #
        # q1 ─────●────────
        #          │
        # q4 ─────X────────
        #
        # q2 ─────●────────
        #          │
        # q4 ─────X────────
        #
        # q4 contains:
        #
        # q1 XOR q2
        #
        # Therefore q4 represents S2.
        #

        circuit.cx(1, 4)
        circuit.cx(2, 4)

        # ---------------------------------
        # 6. Measure syndrome ancillas
        # ---------------------------------

        circuit.measure(3, 0)
        circuit.measure(4, 1)

        return circuit

    def extract_from_bits(self, bits):
        """
        Calculate syndrome from three physical-qubit bits.

        This helper is retained for testing and validation.

        Examples:

            111 -> 00
            011 -> 10
            101 -> 11
            110 -> 01
        """

        if len(bits) != 3:
            raise ValueError(
                "bits must contain exactly 3 values"
            )

        q0, q1, q2 = bits

        # S1 = Z0 Z1
        s1 = q0 ^ q1

        # S2 = Z1 Z2
        s2 = q1 ^ q2

        return f"{s1}{s2}"

    def extract_from_string(self, bitstring):
        """
        Calculate syndrome from a three-bit physical state.

        Example:

            '101' -> '11'
        """

        if len(bitstring) != 3:
            raise ValueError(
                "bitstring must contain exactly 3 bits"
            )

        bits = [
            int(bitstring[0]),
            int(bitstring[1]),
            int(bitstring[2]),
        ]

        return self.extract_from_bits(bits)