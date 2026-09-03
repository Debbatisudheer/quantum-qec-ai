from qiskit import QuantumCircuit


def create_three_qubit_circuit():
    """
    Create a basic 3-qubit quantum circuit.

    The qubits start in |000⟩.
    We then measure all three qubits.
    """

    circuit = QuantumCircuit(3, 3)

    # Measure all three qubits
    circuit.measure(
        [0, 1, 2],
        [0, 1, 2]
    )

    return circuit