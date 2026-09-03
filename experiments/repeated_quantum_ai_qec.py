from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.repeated_qec import (
    RepeatedQuantumQEC
)

from quantum.repeated_measurement import (
    RepeatedQuantumMeasurementParser
)

from correction.time_varying_correction import (
    TimeVaryingCorrectionEngine
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


class RepeatedQuantumAIQECExperiment:
    """
    Full round-by-round quantum AI-QEC experiment.

    Flow:

        Logical state
             ↓
        Quantum encoding
             ↓
        Round 1
             ↓
        Physical X errors
             ↓
        Quantum syndrome
             ↓
        Syndrome measurement
             ↓
        Round 2
             ↓
            ...
             ↓
        Observed syndrome history
             ↓
        Detection events
             ↓
        AI decoder
             ↓
        Predicted final error state
             ↓
        Quantum correction
             ↓
        Final measurement
             ↓
        Logical recovery
    """

    def __init__(
        self,
        rounds=5,
        shots=1
    ):

        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if shots <= 0:
            raise ValueError(
                "shots must be greater than 0"
            )

        self.rounds = rounds
        self.shots = shots

        self.backend = AerSimulator()

        self.qec = (
            RepeatedQuantumQEC(
                rounds=rounds
            )
        )

        self.parser = (
            RepeatedQuantumMeasurementParser(
                rounds=rounds
            )
        )

        self.correction = (
            TimeVaryingCorrectionEngine()
        )

        self.logical_recovery = (
            LogicalRecovery()
        )

    def calculate_detection_events(
        self,
        syndrome_history
    ):
        """
        Calculate temporal syndrome changes.
        """

        if len(syndrome_history) != (
            self.rounds
        ):
            raise ValueError(
                "syndrome history length "
                "does not match rounds"
            )

        events = []

        previous = "00"

        for syndrome in syndrome_history:

            if len(syndrome) != 2:
                raise ValueError(
                    "Each syndrome must contain 2 bits"
                )

            event_s1 = (
                int(previous[0])
                ^ int(syndrome[0])
            )

            event_s2 = (
                int(previous[1])
                ^ int(syndrome[1])
            )

            events.append(
                f"{event_s1}{event_s2}"
            )

            previous = syndrome

        return events

    def encode_features(
        self,
        syndrome_history,
        detection_events
    ):
        """
        Convert observed quantum syndrome data
        into the exact feature representation
        used by the existing AI decoder.
        """

        features = []

        for syndrome in syndrome_history:

            features.extend([
                int(syndrome[0]),
                int(syndrome[1])
            ])

        for event in detection_events:

            features.extend([
                int(event[0]),
                int(event[1])
            ])

        return features

    def apply_quantum_correction(
        self,
        logical_state,
        actual_error_history,
        predicted_error_state
    ):
        """
        Build a second quantum circuit containing
        the same accumulated physical errors and
        then apply the AI-predicted correction.

        This gives us a clean final quantum
        correction stage after decoding.
        """

        circuit = (
            self.qec.create_encoded_state(
                logical_state
            )
        )

        previous = [
            0,
            0,
            0
        ]

        for error_state in (
            actual_error_history
        ):

            self.qec.apply_error_transition(
                circuit,
                previous,
                error_state
            )

            previous = (
                error_state.copy()
            )

        # AI correction.
        for qubit, correction in enumerate(
            predicted_error_state
        ):

            if correction == 1:
                circuit.x(qubit)

        self.qec.add_final_measurements(
            circuit
        )

        compiled = transpile(
            circuit,
            self.backend
        )

        result = self.backend.run(
            compiled,
            shots=self.shots
        ).result()

        counts = result.get_counts()

        measured_state = (
            self.parser.extract_most_likely_result(
                counts
            )
        )

        parsed = self.parser.parse(
            measured_state
        )

        final_state = (
            parsed["final_state"]
        )

        recovered_logical_state = (
            self.logical_recovery.recover(
                [
                    int(bit)
                    for bit in final_state
                ]
            )
        )

        logical_success = (
            recovered_logical_state
            == logical_state
        )

        return {
            "counts":
                counts,

            "measured_state":
                final_state,

            "recovered_logical_state":
                recovered_logical_state,

            "logical_success":
                logical_success
        }

    def run_trial(
        self,
        sample,
        decoder
    ):
        """
        Run one complete quantum trial.

        IMPORTANT:

        The decoder receives only:

            observed quantum syndrome history
            detection events

        It never receives:

            actual error history
            final error state
            perfect syndrome
        """

        logical_state = (
            sample["logical_state"]
        )

        physical_error_history = (
            sample["physical_error_history"]
        )

        # --------------------------------
        # Build round-by-round quantum circuit
        # --------------------------------

        circuit = (
            self.qec.create_round_by_round_circuit(
                logical_state,
                physical_error_history
            )
        )

        compiled = transpile(
            circuit,
            self.backend
        )

        result = self.backend.run(
            compiled,
            shots=self.shots
        ).result()

        counts = result.get_counts()

        raw_result = (
            self.parser.extract_most_likely_result(
                counts
            )
        )

        parsed = self.parser.parse(
            raw_result
        )

        quantum_syndrome_history = (
            parsed["syndrome_history"]
        )

        # --------------------------------
        # Detection events
        # --------------------------------

        detection_events = (
            self.calculate_detection_events(
                quantum_syndrome_history
            )
        )

        # --------------------------------
        # AI features
        # --------------------------------

        features = self.encode_features(
            quantum_syndrome_history,
            detection_events
        )

        # --------------------------------
        # AI prediction
        # --------------------------------

        predicted_error_state = (
            decoder.decode(
                features
            )
        )

        # --------------------------------
        # Final quantum correction
        # --------------------------------

        correction_result = (
            self.apply_quantum_correction(
                logical_state,
                physical_error_history,
                predicted_error_state
            )
        )

        return {
            "logical_state":
                logical_state,

            "actual_error_state":
                list(
                    sample[
                        "final_error_state"
                    ]
                ),

            "observed_syndrome_history":
                quantum_syndrome_history,

            "detection_events":
                detection_events,

            "features":
                features,

            "predicted_error_state":
                predicted_error_state,

            "measured_state":
                correction_result[
                    "measured_state"
                ],

            "recovered_logical_state":
                correction_result[
                    "recovered_logical_state"
                ],

            "logical_success":
                correction_result[
                    "logical_success"
                ],

            "counts":
                correction_result[
                    "counts"
                ]
        }

    def run_experiment(
        self,
        samples,
        decoder
    ):
        """
        Run multiple real quantum-simulation trials.
        """

        if len(samples) == 0:
            raise ValueError(
                "samples cannot be empty"
            )

        results = []

        for sample in samples:

            result = self.run_trial(
                sample,
                decoder
            )

            results.append(
                result
            )

        return results

    def calculate_metrics(
        self,
        results
    ):
        if len(results) == 0:
            raise ValueError(
                "results cannot be empty"
            )

        total = len(results)

        logical_successes = sum(
            1
            for result in results
            if result[
                "logical_success"
            ]
        )

        logical_failures = (
            total
            - logical_successes
        )

        physical_successes = 0

        for result in results:

            if (
                result[
                    "actual_error_state"
                ]
                ==
                result[
                    "predicted_error_state"
                ]
            ):
                physical_successes += 1

        physical_failures = (
            total
            - physical_successes
        )

        return {
            "total_trials":
                total,

            "physical_success_rate":
                physical_successes / total,

            "physical_error_rate":
                physical_failures / total,

            "logical_success_rate":
                logical_successes / total,

            "logical_error_rate":
                logical_failures / total
        }