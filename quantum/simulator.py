from qiskit import transpile
from qiskit_aer import AerSimulator


class QuantumSimulator:
    """
    Wrapper around the Qiskit Aer simulator.

    This class will later become the main execution
    layer for our quantum experiments.
    """

    def __init__(self):
        self.backend = AerSimulator()

    def run(self, circuit, shots=1024):
        """
        Execute a quantum circuit.

        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of repeated executions

        Returns:
            Qiskit Result object
        """

        # Transpile the circuit for the selected simulator
        compiled_circuit = transpile(
            circuit,
            self.backend
        )

        # Execute the transpiled circuit
        result = self.backend.run(
            compiled_circuit,
            shots=shots
        ).result()

        return result