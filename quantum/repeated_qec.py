from qiskit import QuantumCircuit


class RepeatedQuantumQEC:
    """
    Round-by-round quantum QEC circuit for the
    3-qubit bit-flip repetition code.

    Data qubits:

        q0
        q1
        q2

    Each round gets two fresh syndrome ancillas:

        S1 ancilla
        S2 ancilla

    Stabilizers:

        S1 = Z0 Z1
        S2 = Z1 Z2

    Syndrome extraction:

        ancilla_S1:
            CNOT(q0 -> ancilla)
            CNOT(q1 -> ancilla)

        ancilla_S2:
            CNOT(q1 -> ancilla)
            CNOT(q2 -> ancilla)

    This implementation keeps separate ancillas for
    each round. This avoids having to reset ancillas
    between rounds and makes the circuit easier to
    inspect and debug.
    """

    def __init__(self, rounds=5):

        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        self.rounds = rounds

        self.data_qubits = 3

        self.ancilla_qubits_per_round = 2

        self.total_qubits = (
            self.data_qubits
            + (
                self.rounds
                * self.ancilla_qubits_per_round
            )
        )

        # First 3 classical bits are reserved
        # for final physical-state measurement.
        self.final_classical_bits = 3

        self.syndrome_classical_bits = (
            self.rounds * 2
        )

        self.total_classical_bits = (
            self.final_classical_bits
            + self.syndrome_classical_bits
        )

    def validate_logical_state(
        self,
        logical_state
    ):
        if logical_state not in (0, 1):
            raise ValueError(
                "logical_state must be 0 or 1"
            )

    def validate_error_state(
        self,
        error_state
    ):
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

    def validate_error_history(
        self,
        error_history
    ):
        if len(error_history) != self.rounds:
            raise ValueError(
                "error_history length must equal "
                "configured rounds"
            )

        for error_state in error_history:
            self.validate_error_state(
                error_state
            )

    def create_encoded_state(
        self,
        logical_state
    ):
        """
        Encode:

            |0>L = |000>
            |1>L = |111>
        """

        self.validate_logical_state(
            logical_state
        )

        circuit = QuantumCircuit(
            self.total_qubits,
            self.total_classical_bits
        )

        if logical_state == 1:
            circuit.x(0)

        circuit.cx(0, 1)
        circuit.cx(0, 2)

        return circuit

    def apply_error_transition(
        self,
        circuit,
        previous_error_state,
        current_error_state
    ):
        """
        Apply only the X errors that changed between
        two consecutive physical error states.

        Example:

            previous = [0, 0, 0]
            current  = [1, 0, 1]

        Apply:

            X(q0)
            X(q2)

        Because X-error parity is accumulated.
        """

        self.validate_error_state(
            previous_error_state
        )

        self.validate_error_state(
            current_error_state
        )

        for qubit in range(3):

            changed = (
                previous_error_state[qubit]
                ^ current_error_state[qubit]
            )

            if changed == 1:
                circuit.x(qubit)

        return circuit

    def syndrome_ancilla_indices(
        self,
        round_index
    ):
        """
        Return the two ancilla qubits for a round.

        Round 0:

            S1 -> q3
            S2 -> q4

        Round 1:

            S1 -> q5
            S2 -> q6

        etc.
        """

        if not 0 <= round_index < self.rounds:
            raise ValueError(
                "round_index out of range"
            )

        first_ancilla = (
            self.data_qubits
            + (
                round_index
                * self.ancilla_qubits_per_round
            )
        )

        second_ancilla = (
            first_ancilla + 1
        )

        return (
            first_ancilla,
            second_ancilla
        )

    def syndrome_classical_indices(
        self,
        round_index
    ):
        """
        Return the two classical bits used
        for syndrome measurement in a round.

        Classical bits 0,1,2 are reserved for
        final physical-state measurement.

        Therefore:

            Round 0 -> c3,c4
            Round 1 -> c5,c6
            ...
        """

        if not 0 <= round_index < self.rounds:
            raise ValueError(
                "round_index out of range"
            )

        first_bit = (
            self.final_classical_bits
            + (
                round_index * 2
            )
        )

        second_bit = (
            first_bit + 1
        )

        return (
            first_bit,
            second_bit
        )

    def extract_syndrome(
        self,
        circuit,
        round_index
    ):
        """
        Extract:

            S1 = Z0 Z1
            S2 = Z1 Z2
        """

        (
            s1_ancilla,
            s2_ancilla
        ) = self.syndrome_ancilla_indices(
            round_index
        )

        (
            s1_classical,
            s2_classical
        ) = self.syndrome_classical_indices(
            round_index
        )

        # S1 = Z0 Z1
        circuit.cx(
            0,
            s1_ancilla
        )

        circuit.cx(
            1,
            s1_ancilla
        )

        # S2 = Z1 Z2
        circuit.cx(
            1,
            s2_ancilla
        )

        circuit.cx(
            2,
            s2_ancilla
        )

        circuit.measure(
            s1_ancilla,
            s1_classical
        )

        circuit.measure(
            s2_ancilla,
            s2_classical
        )

        return circuit

    def add_final_measurements(
        self,
        circuit
    ):
        """
        Measure the three physical data qubits
        into classical bits 0,1,2.
        """

        circuit.measure(
            0,
            0
        )

        circuit.measure(
            1,
            1
        )

        circuit.measure(
            2,
            2
        )

        return circuit

    def create_round_by_round_circuit(
        self,
        logical_state,
        physical_error_history
    ):
        """
        Create the complete round-by-round
        quantum QEC circuit.

        physical_error_history contains the
        accumulated X-error state after each round.

        Example:

            [
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [1,1,0],
                [0,1,0]
            ]
        """

        self.validate_logical_state(
            logical_state
        )

        self.validate_error_history(
            physical_error_history
        )

        circuit = (
            self.create_encoded_state(
                logical_state
            )
        )

        previous_error_state = [
            0,
            0,
            0
        ]

        for round_index in range(
            self.rounds
        ):

            current_error_state = (
                physical_error_history[
                    round_index
                ]
            )

            # --------------------------------
            # Physical noise for this round
            # --------------------------------

            self.apply_error_transition(
                circuit,
                previous_error_state,
                current_error_state
            )

            # --------------------------------
            # Syndrome extraction
            # --------------------------------

            self.extract_syndrome(
                circuit,
                round_index
            )

            previous_error_state = (
                current_error_state.copy()
            )

        # --------------------------------
        # Final physical measurement
        # --------------------------------

        self.add_final_measurements(
            circuit
        )

        return circuit