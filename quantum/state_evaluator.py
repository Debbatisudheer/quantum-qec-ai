from qiskit import QuantumCircuit


class QuantumStateEvaluator:
    """
    Evaluate the final physical state of the
    3-qubit repetition code.

    Logical states:

        |0>L = |000>
        |1>L = |111>

    This evaluator measures the three physical
    qubits and determines the recovered logical
    state using majority voting.
    """

    def __init__(self):
        self.num_qubits = 3

    def validate_logical_state(self, logical_state):
        if logical_state not in (0, 1):
            raise ValueError(
                "logical_state must be 0 or 1"
            )

    def create_encoded_state(
        self,
        logical_state
    ):
        """
        Create the encoded logical state.

            logical 0 -> |000>
            logical 1 -> |111>
        """

        self.validate_logical_state(
            logical_state
        )

        circuit = QuantumCircuit(
            self.num_qubits,
            self.num_qubits
        )

        if logical_state == 1:
            circuit.x(0)

        circuit.cx(0, 1)
        circuit.cx(0, 2)

        return circuit

    def apply_x_errors(
        self,
        circuit,
        error_state
    ):
        """
        Apply an X error to every qubit whose
        error-state bit is 1.
        """

        if len(error_state) != 3:
            raise ValueError(
                "error_state must contain 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in error_state
        ):
            raise ValueError(
                "error_state must contain only 0 and 1"
            )

        for qubit, error in enumerate(
            error_state
        ):
            if error == 1:
                circuit.x(qubit)

        return circuit

    def apply_corrections(
        self,
        circuit,
        predicted_error_state
    ):
        """
        Apply the AI-predicted correction.

        X correction is applied wherever the
        predicted error pattern contains 1.
        """

        if len(predicted_error_state) != 3:
            raise ValueError(
                "predicted_error_state must contain 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in predicted_error_state
        ):
            raise ValueError(
                "predicted_error_state must contain "
                "only 0 and 1"
            )

        for qubit, correction in enumerate(
            predicted_error_state
        ):
            if correction == 1:
                circuit.x(qubit)

        return circuit

    def add_measurements(
        self,
        circuit
    ):
        """
        Measure all three physical qubits.
        """

        circuit.measure(
            [0, 1, 2],
            [0, 1, 2]
        )

        return circuit

    def recover_logical_state(
        self,
        measured_state
    ):
        """
        Recover logical state using majority voting.

        000 -> 0
        001 -> 0
        010 -> 0
        100 -> 0

        111 -> 1
        110 -> 1
        101 -> 1
        011 -> 1
        """

        if not isinstance(
            measured_state,
            str
        ):
            raise ValueError(
                "measured_state must be a string"
            )

        if len(measured_state) != 3:
            raise ValueError(
                "measured_state must contain 3 bits"
            )

        if any(
            bit not in "01"
            for bit in measured_state
        ):
            raise ValueError(
                "measured_state must contain only 0 and 1"
            )

        number_of_ones = measured_state.count(
            "1"
        )

        if number_of_ones >= 2:
            return 1

        return 0

    def logical_success(
        self,
        original_logical_state,
        measured_state
    ):
        """
        Determine whether the final measured
        physical state preserves the original
        logical information.
        """

        self.validate_logical_state(
            original_logical_state
        )

        recovered_state = (
            self.recover_logical_state(
                measured_state
            )
        )

        return (
            recovered_state
            == original_logical_state
        )