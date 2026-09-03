from qiskit import transpile
from qiskit_aer import AerSimulator

from quantum.state_evaluator import (
    QuantumStateEvaluator
)

from correction.time_varying_correction import (
    TimeVaryingCorrectionEngine
)


class TimeVaryingQuantumQECExperiment:
    """
    Multi-trial quantum QEC experiment.

    Complete flow:

        Logical state
             ↓
        Encoded quantum state
             ↓
        Time-varying physical errors
             ↓
        AI predicted correction
             ↓
        Quantum correction
             ↓
        Quantum measurement
             ↓
        Logical recovery
             ↓
        Logical success / failure

    This experiment receives an already-trained AI decoder.
    """

    def __init__(
        self,
        shots=1
    ):
        if shots <= 0:
            raise ValueError(
                "shots must be greater than 0"
            )

        self.shots = shots

        self.backend = AerSimulator()

        self.evaluator = (
            QuantumStateEvaluator()
        )

        self.correction = (
            TimeVaryingCorrectionEngine()
        )

    def run_trial(
        self,
        logical_state,
        actual_error_state,
        predicted_error_state
    ):
        """
        Run one quantum QEC trial.
        """

        circuit = (
            self.evaluator.create_encoded_state(
                logical_state
            )
        )

        # Apply actual accumulated
        # physical X errors.
        circuit = (
            self.evaluator.apply_x_errors(
                circuit,
                actual_error_state
            )
        )

        # Apply AI-predicted correction.
        circuit = (
            self.evaluator.apply_corrections(
                circuit,
                predicted_error_state
            )
        )

        # Measure final physical state.
        circuit = (
            self.evaluator.add_measurements(
                circuit
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

        measured_state = max(
            counts,
            key=counts.get
        )

        # Convert Qiskit classical-bit
        # ordering to q0, q1, q2 ordering.
        measured_state = (
            measured_state[::-1]
        )

        recovered_logical_state = (
            self.evaluator.recover_logical_state(
                measured_state
            )
        )

        logical_success = (
            recovered_logical_state
            == logical_state
        )

        correction_result = (
            self.correction.correct_sample(
                actual_error_state,
                predicted_error_state
            )
        )

        return {
            "logical_state":
                logical_state,

            "actual_error_state":
                list(actual_error_state),

            "predicted_error_state":
                list(predicted_error_state),

            "corrected_error_state":
                correction_result[
                    "corrected_state"
                ],

            "measured_state":
                measured_state,

            "recovered_logical_state":
                recovered_logical_state,

            "logical_success":
                logical_success,

            "counts":
                counts
        }

    def run_experiment(
        self,
        samples,
        ml_features,
        decoder
    ):
        """
        Run the trained decoder across all
        supplied samples.

        The decoder sees only ML features.

        Ground truth is used only for:
            - quantum simulation
            - evaluation

        It is NOT provided to the decoder.
        """

        if len(samples) == 0:
            raise ValueError(
                "samples cannot be empty"
            )

        if len(samples) != len(
            ml_features
        ):
            raise ValueError(
                "samples and ml_features "
                "must have the same length"
            )

        results = []

        for sample, features in zip(
            samples,
            ml_features
        ):

            # AI prediction.
            predicted_error = (
                decoder.decode(
                    features
                )
            )

            logical_state = (
                sample.get(
                    "logical_state",
                    0
                )
            )

            actual_error_state = (
                sample[
                    "final_error_state"
                ]
            )

            trial_result = (
                self.run_trial(
                    logical_state,
                    actual_error_state,
                    predicted_error
                )
            )

            results.append(
                trial_result
            )

        return results

    def calculate_metrics(
        self,
        results
    ):
        """
        Calculate experiment-level metrics.
        """

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

        physical_successes = sum(
            1
            for result in results
            if result[
                "corrected_error_state"
            ] == [0, 0, 0]
        )

        physical_failures = (
            total
            - physical_successes
        )

        return {
            "total_trials":
                total,

            "logical_successes":
                logical_successes,

            "logical_failures":
                logical_failures,

            "logical_success_rate":
                logical_successes / total,

            "logical_error_rate":
                logical_failures / total,

            "physical_successes":
                physical_successes,

            "physical_failures":
                physical_failures,

            "physical_success_rate":
                physical_successes / total,

            "physical_error_rate":
                physical_failures / total
        }