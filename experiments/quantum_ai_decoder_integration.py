from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from quantum.repeated_qec import RepeatedQuantumQEC
from quantum.repeated_measurement import RepeatedQuantumMeasurementParser
from evaluation.logical_recovery import LogicalRecovery
from syndrome.measurement_noise import SyndromeMeasurementNoise


class QuantumAIDecoderIntegration:
    """
    Connects the repeated quantum QEC circuit
    to the time-varying AI decoder.

    Complete flow:

        Quantum QEC circuit
                ↓
        Perfect quantum syndrome
                ↓
        Measurement / readout noise
                ↓
        Observed syndrome
                ↓
        Detection events
                ↓
        Feature encoding
                ↓
        AI decoder
                ↓
        Predicted final error
                ↓
        Quantum correction
                ↓
        Final quantum measurement
                ↓
        Logical recovery
    """

    def __init__(
        self,
        rounds=5,
        shots=1,
        measurement_noise_probability=0.0,
        random_seed=42
    ):

        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if shots <= 0:
            raise ValueError(
                "shots must be greater than 0"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability must be between 0 and 1"
            )

        self.rounds = rounds
        self.shots = shots

        self.measurement_noise_probability = (
            measurement_noise_probability
        )

        self.measurement_noise = SyndromeMeasurementNoise(
            probability=measurement_noise_probability,
            seed=random_seed
        )

        self.backend = AerSimulator()

        self.quantum_qec = RepeatedQuantumQEC(
            rounds=rounds
        )

        self.measurement_parser = (
            RepeatedQuantumMeasurementParser(
                rounds=rounds
            )
        )

        self.logical_recovery = LogicalRecovery()

    # =========================================================
    # DETECTION EVENTS
    # =========================================================

    def calculate_detection_events(
        self,
        syndrome_history
    ):

        if len(syndrome_history) != self.rounds:
            raise ValueError(
                "syndrome_history length must match rounds"
            )

        detection_events = []

        previous_syndrome = "00"

        for syndrome in syndrome_history:

            if len(syndrome) != 2:
                raise ValueError(
                    "Each syndrome must contain exactly 2 bits"
                )

            if any(
                bit not in "01"
                for bit in syndrome
            ):
                raise ValueError(
                    "Syndrome must contain only 0 and 1"
                )

            event_s1 = (
                int(previous_syndrome[0])
                ^ int(syndrome[0])
            )

            event_s2 = (
                int(previous_syndrome[1])
                ^ int(syndrome[1])
            )

            detection_events.append(
                f"{event_s1}{event_s2}"
            )

            previous_syndrome = syndrome

        return detection_events

    # =========================================================
    # SYNDROME MEASUREMENT NOISE
    # =========================================================

    def apply_syndrome_measurement_noise(
        self,
        syndrome_history
    ):
        """
        Apply measurement/readout noise to the
        perfect syndrome history.

        The quantum circuit produces the perfect
        syndrome measurement.

        This method models the classical observation
        becoming noisy after measurement.
        """

        if len(syndrome_history) != self.rounds:
            raise ValueError(
                "syndrome_history length must match rounds"
            )

        observed_syndrome_history = []

        for syndrome in syndrome_history:

            observed_syndrome = (
                self.measurement_noise.apply(
                    syndrome
                )
            )

            observed_syndrome_history.append(
                observed_syndrome
            )

        return observed_syndrome_history

    # =========================================================
    # FEATURE ENCODING
    # =========================================================

    def encode_features(
        self,
        syndrome_history,
        detection_events
    ):

        if len(syndrome_history) != self.rounds:
            raise ValueError(
                "syndrome_history length must match rounds"
            )

        if len(detection_events) != self.rounds:
            raise ValueError(
                "detection_events length must match rounds"
            )

        features = []

        # -----------------------------------------------------
        # Syndrome history
        # -----------------------------------------------------

        for syndrome in syndrome_history:

            if len(syndrome) != 2:
                raise ValueError(
                    "Each syndrome must contain 2 bits"
                )

            if any(
                bit not in "01"
                for bit in syndrome
            ):
                raise ValueError(
                    "Syndrome must contain only 0 and 1"
                )

            features.extend(
                int(bit)
                for bit in syndrome
            )

        # -----------------------------------------------------
        # Detection events
        # -----------------------------------------------------

        for event in detection_events:

            if len(event) != 2:
                raise ValueError(
                    "Each detection event must contain 2 bits"
                )

            if any(
                bit not in "01"
                for bit in event
            ):
                raise ValueError(
                    "Detection event must contain only 0 and 1"
                )

            features.extend(
                int(bit)
                for bit in event
            )

        return features

    # =========================================================
    # QUANTUM SYNDROME GENERATION
    # =========================================================

    def run_quantum_trial(
        self,
        logical_state,
        physical_error_history
    ):

        circuit = (
            self.quantum_qec
            .create_round_by_round_circuit(
                logical_state=logical_state,
                physical_error_history=
                    physical_error_history
            )
        )

        result = self.backend.run(
            circuit,
            shots=self.shots
        ).result()

        counts = result.get_counts()

        if not counts:
            raise ValueError(
                "Quantum simulation returned no measurement counts"
            )

        raw_measurement = max(
            counts,
            key=counts.get
        )

        parsed = (
            self.measurement_parser.parse(
                raw_measurement
            )
        )

        return {
            "counts":
                counts,

            "raw_measurement":
                raw_measurement,

            "final_state":
                parsed["final_state"],

            "syndrome_history":
                parsed["syndrome_history"]
        }

    # =========================================================
    # CREATE CORRECTION CIRCUIT
    # =========================================================

    def create_correction_circuit(
        self,
        logical_state
    ):
        """
        Create a fresh quantum circuit for
        the correction stage.

        Only the three data qubits need to be
        measured during the correction stage.
        """

        if logical_state not in (0, 1):
            raise ValueError(
                "logical_state must be 0 or 1"
            )

        circuit = QuantumCircuit(
            self.quantum_qec.total_qubits,
            3
        )

        # -----------------------------------------------------
        # Encode logical state
        # -----------------------------------------------------

        if logical_state == 1:
            circuit.x(0)

        circuit.cx(0, 1)
        circuit.cx(0, 2)

        return circuit

    # =========================================================
    # QUANTUM CORRECTION
    # =========================================================

    def apply_quantum_correction(
        self,
        logical_state,
        physical_error_history,
        predicted_error_state
    ):

        # -----------------------------------------------------
        # Validate predicted error
        # -----------------------------------------------------

        if len(predicted_error_state) != 3:
            raise ValueError(
                "predicted_error_state must contain 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in predicted_error_state
        ):
            raise ValueError(
                "predicted_error_state must contain only 0 and 1"
            )

        # -----------------------------------------------------
        # Create correction circuit
        # -----------------------------------------------------

        circuit = (
            self.create_correction_circuit(
                logical_state=logical_state
            )
        )

        # -----------------------------------------------------
        # Replay physical error evolution
        # -----------------------------------------------------

        previous_error_state = [
            0,
            0,
            0
        ]

        for current_error_state in physical_error_history:

            self.quantum_qec.apply_error_transition(
                circuit,
                previous_error_state,
                current_error_state
            )

            previous_error_state = (
                current_error_state.copy()
            )

        # -----------------------------------------------------
        # Apply AI-predicted correction
        # -----------------------------------------------------

        for qubit in range(3):

            if predicted_error_state[qubit] == 1:

                circuit.x(
                    qubit
                )

        # -----------------------------------------------------
        # Measure data qubits
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Run correction circuit
        # -----------------------------------------------------

        result = self.backend.run(
            circuit,
            shots=self.shots
        ).result()

        counts = result.get_counts()

        if not counts:
            raise ValueError(
                "Quantum correction circuit returned no counts"
            )

        raw_measurement = max(
            counts,
            key=counts.get
        )

        # -----------------------------------------------------
        # Clean measurement
        # -----------------------------------------------------

        clean_measurement = (
            raw_measurement
            .replace(
                " ",
                ""
            )
        )

        if len(clean_measurement) != 3:
            raise ValueError(
                "Correction measurement must contain exactly 3 bits"
            )

        if any(
            bit not in "01"
            for bit in clean_measurement
        ):
            raise ValueError(
                "Correction measurement must contain only 0 and 1"
            )

        # -----------------------------------------------------
        # Qiskit displays:
        #
        #     c2 c1 c0
        #
        # Reverse to:
        #
        #     q0 q1 q2
        # -----------------------------------------------------

        final_state_string = (
            clean_measurement[::-1]
        )

        # -----------------------------------------------------
        # Convert:
        #
        #     "101"
        #
        # into:
        #
        #     [1, 0, 1]
        #
        # LogicalRecovery expects integer bits.
        # -----------------------------------------------------

        final_state = [
            int(bit)
            for bit in final_state_string
        ]

        # -----------------------------------------------------
        # Validate final state
        # -----------------------------------------------------

        if len(final_state) != 3:
            raise ValueError(
                "Final physical state must contain 3 bits"
            )

        if any(
            bit not in (0, 1)
            for bit in final_state
        ):
            raise ValueError(
                "Final physical state must contain only 0 and 1"
            )

        # -----------------------------------------------------
        # Logical recovery
        # -----------------------------------------------------

        recovered_logical = (
            self.logical_recovery.recover(
                final_state
            )
        )

        logical_success = (
            recovered_logical == logical_state
        )

        return {
            "counts":
                counts,

            "raw_measurement":
                raw_measurement,

            "final_state":
                final_state,

            "final_state_string":
                final_state_string,

            "recovered_logical":
                recovered_logical,

            "logical_success":
                logical_success
        }

    # =========================================================
    # COMPLETE QUANTUM + AI TRIAL
    # =========================================================

    def run_trial(
        self,
        sample,
        decoder
    ):

        # -----------------------------------------------------
        # Ground truth
        # -----------------------------------------------------

        logical_state = sample[
            "logical_state"
        ]

        physical_error_history = sample[
            "physical_error_history"
        ]

        actual_error = sample[
            "final_error_state"
        ]

        # -----------------------------------------------------
        # STEP 1
        #
        # Real quantum repeated-QEC circuit
        # -----------------------------------------------------

        quantum_result = (
            self.run_quantum_trial(
                logical_state=
                    logical_state,

                physical_error_history=
                    physical_error_history
            )
        )

        # -----------------------------------------------------
        # Perfect syndrome produced by
        # quantum syndrome measurements
        # -----------------------------------------------------

        perfect_syndrome_history = (
            quantum_result[
                "syndrome_history"
            ]
        )

        # -----------------------------------------------------
        # STEP 2
        #
        # Apply measurement/readout noise
        #
        # IMPORTANT:
        #
        # The AI does NOT receive the perfect
        # syndrome history.
        # -----------------------------------------------------

        observed_syndrome_history = (
            self.apply_syndrome_measurement_noise(
                perfect_syndrome_history
            )
        )

        # -----------------------------------------------------
        # STEP 3
        #
        # Detection events are calculated from
        # the OBSERVED syndrome.
        # -----------------------------------------------------

        detection_events = (
            self.calculate_detection_events(
                observed_syndrome_history
            )
        )

        # -----------------------------------------------------
        # STEP 4
        #
        # Observed quantum data → AI features
        # -----------------------------------------------------

        features = (
            self.encode_features(
                observed_syndrome_history,
                detection_events
            )
        )

        # -----------------------------------------------------
        # STEP 5
        #
        # AI decoder
        # -----------------------------------------------------

        predicted_error = (
            decoder.decode(
                features
            )
        )

        # -----------------------------------------------------
        # STEP 6
        #
        # Quantum correction
        # -----------------------------------------------------

        correction_result = (
            self.apply_quantum_correction(
                logical_state=
                    logical_state,

                physical_error_history=
                    physical_error_history,

                predicted_error_state=
                    predicted_error
            )
        )

        # -----------------------------------------------------
        # STEP 7
        #
        # Physical decoder correctness
        #
        # Ground truth is used ONLY for evaluation.
        # -----------------------------------------------------

        physical_success = (
            actual_error == predicted_error
        )

        # -----------------------------------------------------
        # Complete result
        # -----------------------------------------------------

        return {
            "logical_state":
                logical_state,

            "actual_error":
                actual_error,

            "perfect_syndrome_history":
                perfect_syndrome_history,

            "quantum_syndrome_history":
                observed_syndrome_history,

            "observed_syndrome_history":
                observed_syndrome_history,

            "detection_events":
                detection_events,

            "features":
                features,

            "predicted_error":
                predicted_error,

            "measured_state":
                correction_result[
                    "final_state"
                ],

            "measured_state_string":
                correction_result[
                    "final_state_string"
                ],

            "recovered_logical":
                correction_result[
                    "recovered_logical"
                ],

            "physical_success":
                physical_success,

            "logical_success":
                correction_result[
                    "logical_success"
                ],

            "counts":
                correction_result[
                    "counts"
                ]
        }

    # =========================================================
    # RUN MULTIPLE TRIALS
    # =========================================================

    def run_experiment(
        self,
        samples,
        decoder
    ):

        if not samples:
            raise ValueError(
                "samples cannot be empty"
            )

        results = []

        for sample in samples:

            result = (
                self.run_trial(
                    sample=sample,
                    decoder=decoder
                )
            )

            results.append(
                result
            )

        return results

    # =========================================================
    # METRICS
    # =========================================================

    def calculate_metrics(
        self,
        results
    ):

        if not results:
            raise ValueError(
                "results cannot be empty"
            )

        total = len(results)

        physical_successes = sum(
            result["physical_success"]
            for result in results
        )

        logical_successes = sum(
            result["logical_success"]
            for result in results
        )

        physical_success_rate = (
            physical_successes
            / total
        )

        logical_success_rate = (
            logical_successes
            / total
        )

        logical_error_rate = (
            1.0
            - logical_success_rate
        )

        return {
            "total_trials":
                total,

            "physical_success_rate":
                physical_success_rate,

            "logical_success_rate":
                logical_success_rate,

            "logical_error_rate":
                logical_error_rate
        }