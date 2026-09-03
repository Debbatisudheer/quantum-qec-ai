from typing import Any

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator,
)
from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder,
)
from evaluation.decoder_evaluator import DecoderEvaluator
from evaluation.logical_recovery import LogicalRecovery


class SimulationTraceService:
    """
    Generate a complete single-sample QEC trace.

    This service deliberately reuses the existing:
        - dataset generator
        - logical-target Random Forest decoder
        - decoder evaluator
        - logical recovery

    The service does not implement a second QEC algorithm.
    """

    def __init__(
        self,
        rounds: int = 5,
        physical_noise_probability: float = 0.10,
        measurement_noise_probability: float = 0.10,
        training_samples: int = 5000,
        random_forest_estimators: int = 100,
        seed: int = 42,
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= physical_noise_probability <= 1.0:
            raise ValueError(
                "physical_noise_probability must be "
                "between 0 and 1"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability must be "
                "between 0 and 1"
            )

        if training_samples <= 0:
            raise ValueError(
                "training_samples must be greater than 0"
            )

        if random_forest_estimators <= 0:
            raise ValueError(
                "random_forest_estimators must be "
                "greater than 0"
            )

        self.rounds = rounds
        self.physical_noise_probability = (
            physical_noise_probability
        )
        self.measurement_noise_probability = (
            measurement_noise_probability
        )
        self.training_samples = training_samples
        self.random_forest_estimators = (
            random_forest_estimators
        )
        self.seed = seed

        self.generator = TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=(
                physical_noise_probability
            ),
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            seed=seed,
        )

        self.decoder = LogicalTargetRandomForestDecoder(
            rounds=rounds,
            n_estimators=random_forest_estimators,
            random_seed=seed,
        )

        self.evaluator = DecoderEvaluator()
        self.recovery = LogicalRecovery()

    # =========================================================
    # STATE HELPERS
    # =========================================================

    @staticmethod
    def _bits_to_string(bits):
        return "".join(str(int(bit)) for bit in bits)

    @staticmethod
    def _string_to_bits(value):
        return [
            int(bit)
            for bit in str(value)
        ]

    @staticmethod
    def _xor_states(left, right):
        if len(left) != len(right):
            raise ValueError(
                "States must have the same length"
            )

        return [
            int(a) ^ int(b)
            for a, b in zip(left, right)
        ]

    # =========================================================
    # TRAIN
    # =========================================================

    def _train_decoder(self):
        training_generator = (
            TimeVaryingQECDatasetGenerator(
                rounds=self.rounds,
                physical_error_probability=(
                    self.physical_noise_probability
                ),
                measurement_noise_probability=(
                    self.measurement_noise_probability
                ),
                seed=self.seed,
            )
        )

        training_samples = (
            training_generator.generate_dataset(
                self.training_samples
            )
        )

        self.decoder.train(training_samples)

        return training_samples

    # =========================================================
    # TRACE
    # =========================================================

    def generate_trace(self):
        """
        Generate one complete QEC trace.
        """

        # -----------------------------------------------------
        # 1. Train decoder using independent training data
        # -----------------------------------------------------

        training_samples = self._train_decoder()

        # -----------------------------------------------------
        # 2. Generate ONE trace sample
        # -----------------------------------------------------

        trace_sample_id = self.training_samples

        sample = self.generator.generate_sample(
            trace_sample_id
        )

        # -----------------------------------------------------
        # 3. Decode
        # -----------------------------------------------------

        predicted_correction = self.decoder.decode(
            sample
        )

        predicted_correction = [
            int(bit)
            for bit in predicted_correction
        ]

        # -----------------------------------------------------
        # 4. Physical states
        # -----------------------------------------------------

        encoded_state = self._string_to_bits(
            sample["encoded_state"]
        )

        actual_error = [
            int(bit)
            for bit in sample["final_error_state"]
        ]

        corrupted_state = (
            self._xor_states(
                encoded_state,
                actual_error,
            )
        )

        corrected_state = (
            self._xor_states(
                corrupted_state,
                predicted_correction,
            )
        )

        # -----------------------------------------------------
        # 5. Logical recovery
        # -----------------------------------------------------

        recovered_logical_state = (
            self.recovery.recover(
                corrected_state
            )
        )

        original_logical_state = int(
            sample["logical_state"]
        )

        logical_success = (
            recovered_logical_state
            == original_logical_state
        )

        physical_recovery = (
            corrected_state
            == encoded_state
        )

        exact_error_match = (
            actual_error
            == predicted_correction
        )

        # -----------------------------------------------------
        # 6. Decoder confidence
        # -----------------------------------------------------

        confidence = None

        try:
            feature_vector = (
                self.decoder.encode_features(
                    sample
                )
            )

            probabilities = (
                self.decoder.predict_proba(
                    [feature_vector]
                )
            )

            if probabilities:
                per_qubit_max = []

                for probability in probabilities:
                    values = [
                        float(value)
                        for value in probability[0]
                    ]

                    if values:
                        per_qubit_max.append(
                            max(values)
                        )

                if per_qubit_max:
                    confidence = (
                        sum(per_qubit_max)
                        / len(per_qubit_max)
                    )

        except Exception:
            # Confidence is supplementary.
            # The trace itself must remain valid.
            confidence = None

        # -----------------------------------------------------
        # 7. Round-by-round state representation
        # -----------------------------------------------------

        physical_error_history = []

        for state in sample[
            "physical_error_history"
        ]:
            physical_error_history.append(
                self._bits_to_string(state)
            )

        # -----------------------------------------------------
        # 8. Return complete trace
        # -----------------------------------------------------

        return {
            "sample_id": sample["sample_id"],

            "qec_code": sample["qec_code"],
            "num_qubits": sample["num_qubits"],
            "rounds": sample["rounds"],

            "logical_state": original_logical_state,

            "encoded_state": (
                sample["encoded_state"]
            ),

            "noise": {
                "physical_error_probability": (
                    sample[
                        "physical_error_probability"
                    ]
                ),
                "measurement_noise_probability": (
                    sample[
                        "measurement_noise_probability"
                    ]
                ),

                "physical_error_history": (
                    physical_error_history
                ),

                "final_error_state": (
                    self._bits_to_string(
                        actual_error
                    )
                ),

                "final_error_description": (
                    sample[
                        "final_error_description"
                    ]
                ),
            },

            "quantum_state": {
                "encoded": (
                    sample["encoded_state"]
                ),

                "corrupted": (
                    self._bits_to_string(
                        corrupted_state
                    )
                ),
            },

            "syndrome": {
                "perfect_history": (
                    sample["syndrome_history"]
                ),

                "observed_history": (
                    sample[
                        "observed_syndrome_history"
                    ]
                ),

                "detection_events": (
                    sample["detection_events"]
                ),

                "final_perfect": (
                    sample["final_syndrome"]
                ),

                "final_observed": (
                    sample[
                        "final_observed_syndrome"
                    ]
                ),
            },

            "decoder": {
                "type": (
                    "logical_target_random_forest"
                ),

                "training_samples": (
                    self.training_samples
                ),

                "random_forest_estimators": (
                    self.random_forest_estimators
                ),

                "predicted_correction": (
                    self._bits_to_string(
                        predicted_correction
                    )
                ),

                "predicted_correction_bits": (
                    predicted_correction
                ),

                "confidence": confidence,
            },

            "correction": {
                "actual_error": (
                    self._bits_to_string(
                        actual_error
                    )
                ),

                "predicted_correction": (
                    self._bits_to_string(
                        predicted_correction
                    )
                ),

                "corrected_state": (
                    self._bits_to_string(
                        corrected_state
                    )
                ),
            },

            "recovery": {
                "original_logical_state": (
                    original_logical_state
                ),

                "recovered_logical_state": (
                    recovered_logical_state
                ),

                "logical_success": (
                    logical_success
                ),

                "logical_failure": (
                    not logical_success
                ),

                "physical_recovery": (
                    physical_recovery
                ),

                "exact_error_match": (
                    exact_error_match
                ),
            },
        }