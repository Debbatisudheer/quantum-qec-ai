from qiskit import QuantumCircuit


class BitFlipCode3:
    """
    3-qubit bit-flip quantum error-correcting code.

    Logical states:

        |0>L = |000>
        |1>L = |111>

    Stabilizers:

        S1 = Z0 Z1
        S2 = Z1 Z2

    Single-qubit bit-flip syndrome:

        No error -> 00
        X0       -> 10
        X1       -> 11
        X2       -> 01
    """

    def __init__(self):
        self.num_physical_qubits = 3
        self.num_logical_qubits = 1

        self.stabilizers = [
            "ZZI",
            "IZZ",
        ]

    def create_encoding_circuit(self, logical_state=0):
        """
        Create a circuit that encodes one logical qubit
        into three physical qubits.

        Args:
            logical_state: 0 or 1

        Returns:
            QuantumCircuit
        """

        if logical_state not in (0, 1):
            raise ValueError("logical_state must be 0 or 1")

        circuit = QuantumCircuit(3, 3)

        # Prepare logical |1> if requested.
        #
        # Before encoding:
        #
        # |0> -> |000>
        # |1> -> |100>
        #
        if logical_state == 1:
            circuit.x(0)

        # Encode:
        #
        # q0 controls q1
        # q0 controls q2
        #
        # |000> -> |000>
        # |100> -> |111>
        circuit.cx(0, 1)
        circuit.cx(0, 2)

        return circuit

    def add_measurements(self, circuit):
        """
        Add measurements to all three physical qubits.
        """

        circuit.measure(
            [0, 1, 2],
            [0, 1, 2]
        )

        return circuit

    def logical_state_description(self, logical_state):
        """
        Return a human-readable description of the
        encoded logical state.
        """

        if logical_state == 0:
            return "|0>L = |000>"

        if logical_state == 1:
            return "|1>L = |111>"

        raise ValueError("logical_state must be 0 or 1")