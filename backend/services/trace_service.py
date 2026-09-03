from typing import Any

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator,
)

from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder,
)

from experiments.quantum_ai_decoder_integration import (
    QuantumAIDecoderIntegration,
)


class _LogicalTargetDecoderAdapter:
    """
    Adapter between QuantumAIDecoderIntegration and the
    existing LogicalTargetRandomForestDecoder.

    QuantumAIDecoderIntegration produces features as:

        all syndrome bits
        +
        all detection-event bits

    The existing Random Forest decoder expects:

        syndrome_1
        syndrome_2
        detection_1
        detection_2

    for every round.

    This adapter converts the feature ordering without
    changing either existing component.
    """

    def __init__(
        self,
        decoder: LogicalTargetRandomForestDecoder,
        rounds: int,
    ):
        self.decoder = decoder
        self.rounds = rounds

    def _convert_features(self, features):
        expected_length = self.rounds * 4

        if len(features) != expected_length:
            raise ValueError(
                "Feature vector length must be "
                f"{expected_length} for {self.rounds} rounds"
            )

        syndrome_end = self.rounds * 2

        syndrome_features = features[:syndrome_end]
        detection_features = features[syndrome_end:]

        decoder_features = []

        for index in range(self.rounds):
            syndrome_start = index * 2
            detection_start = index * 2

            decoder_features.extend(
                [
                    int(syndrome_features[syndrome_start]),
                    int(syndrome_features[syndrome_start + 1]),
                    int(detection_features[detection_start]),
                    int(detection_features[detection_start + 1]),
                ]
            )

        return decoder_features

    def decode(self, features):
        """
        Decode the feature vector using the existing
        trained Random Forest decoder.
        """

        decoder_features = self._convert_features(
            features
        )

        prediction = self.decoder.predict(
            [decoder_features]
        )

        return [
            int(bit)
            for bit in prediction[0]
        ]

    def predict_proba(self, features):
        """
        Return Random Forest class probabilities using
        the same feature conversion as decode().
        """

        decoder_features = self._convert_features(
            features
        )

        return self.decoder.predict_proba(
            [decoder_features]
        )


class SimulationTraceService:
    """
    Generate a complete single-sample quantum + AI QEC trace.

    Complete flow:

        Training dataset
              ↓
        Logical-target Random Forest
              ↓
        Adapter
              ↓
        QuantumAIDecoderIntegration
              ↓
        Real Qiskit repeated-QEC circuit
              ↓
        Perfect syndrome
              ↓
        Measurement/readout noise
              ↓
        Observed syndrome
              ↓
        Detection events
              ↓
        AI decoder
              ↓
        Predicted correction
              ↓
        Quantum correction
              ↓
        Final quantum measurement
              ↓
        Logical recovery
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
                "physical_noise_probability must be between 0 and 1"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability must be between 0 and 1"
            )

        if training_samples <= 0:
            raise ValueError(
                "training_samples must be greater than 0"
            )

        if random_forest_estimators <= 0:
            raise ValueError(
                "random_forest_estimators must be greater than 0"
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

        # ---------------------------------------------------------
        # Dataset generator
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Existing AI decoder
        # ---------------------------------------------------------

        self.decoder = LogicalTargetRandomForestDecoder(
            rounds=rounds,
            n_estimators=random_forest_estimators,
            random_seed=seed,
        )

        # ---------------------------------------------------------
        # Adapter
        # ---------------------------------------------------------

        self.decoder_adapter = _LogicalTargetDecoderAdapter(
            decoder=self.decoder,
            rounds=rounds,
        )

        # ---------------------------------------------------------
        # Real quantum + AI integration
        # ---------------------------------------------------------

        self.quantum_ai = QuantumAIDecoderIntegration(
            rounds=rounds,
            shots=1,
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            random_seed=seed,
        )

    # =========================================================
    # STATE HELPERS
    # =========================================================

    @staticmethod
    def _bits_to_string(bits):
        return "".join(
            str(int(bit))
            for bit in bits
        )

    @staticmethod
    def _xor_state(state_a: str, state_b: str) -> str:
        """
        Apply a bitwise XOR between two 3-qubit state strings.

        Example:

            000 XOR 001 = 001
            111 XOR 100 = 011
        """

        if len(state_a) != len(state_b):
            raise ValueError(
                "Quantum state strings must have the same length"
            )

        return "".join(
            "1" if a != b else "0"
            for a, b in zip(state_a, state_b)
        )

    # =========================================================
    # TRAIN AI DECODER
    # =========================================================

    def _train_decoder(self):
        """
        Train the existing logical-target Random Forest
        using independently generated training data.
        """

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

        self.decoder.train(
            training_samples
        )

        return training_samples

    # =========================================================
    # DECODER CONFIDENCE
    # =========================================================

    def _decoder_confidence(
        self,
        features,
    ):
        """
        Estimate Random Forest confidence from the exact
        feature vector produced by QuantumAIDecoderIntegration.

        Confidence is supplementary. If probability
        information is unavailable, return None.
        """

        try:
            probabilities = (
                self.decoder_adapter.predict_proba(
                    features
                )
            )

            if not probabilities:
                return None

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

            if not per_qubit_max:
                return None

            return (
                sum(per_qubit_max)
                / len(per_qubit_max)
            )

        except Exception:
            return None

    # =========================================================
    # COMPLETE TRACE
    # =========================================================

    def generate_trace(self) -> dict[str, Any]:
        """
        Generate one complete quantum + AI QEC trace.
        """

        # -----------------------------------------------------
        # 1. Train AI decoder
        # -----------------------------------------------------

        training_samples = (
            self._train_decoder()
        )

        # -----------------------------------------------------
        # 2. Generate one independent sample
        # -----------------------------------------------------

        trace_sample_id = self.training_samples

        sample = self.generator.generate_sample(
            trace_sample_id
        )

        # -----------------------------------------------------
        # 3. Run real quantum + AI trial
        # -----------------------------------------------------

        integration_result = (
            self.quantum_ai.run_trial(
                sample=sample,
                decoder=self.decoder_adapter,
            )
        )

        # -----------------------------------------------------
        # 4. Ground truth
        # -----------------------------------------------------

        logical_state = int(
            sample["logical_state"]
        )

        encoded_state = (
            sample["encoded_state"]
        )

        actual_error_bits = (
            sample["final_error_state"]
        )

        actual_error = (
            self._bits_to_string(
                actual_error_bits
            )
        )

        predicted_error = (
            integration_result[
                "predicted_error"
            ]
        )

        measured_state = (
            integration_result[
                "measured_state_string"
            ]
        )

        # -----------------------------------------------------
        # 5. Calculate the TRUE corrupted state
        #
        # encoded state XOR actual physical error
        #
        # IMPORTANT:
        # measured_state_string is the state after the
        # correction process, so it must NOT be used as
        # the corrupted state.
        # -----------------------------------------------------

        corrupted_state = self._xor_state(
            encoded_state,
            actual_error,
        )

        # -----------------------------------------------------
        # 6. Decoder confidence
        # -----------------------------------------------------

        confidence = (
            self._decoder_confidence(
                integration_result["features"]
            )
        )

        # -----------------------------------------------------
        # 7. Physical error history
        # -----------------------------------------------------

        physical_error_history = []

        for state in sample[
            "physical_error_history"
        ]:
            physical_error_history.append(
                self._bits_to_string(state)
            )

        # -----------------------------------------------------
        # 8. Syndrome history
        # -----------------------------------------------------

        perfect_history = (
            integration_result[
                "perfect_syndrome_history"
            ]
        )

        observed_history = (
            integration_result[
                "observed_syndrome_history"
            ]
        )

        detection_events = (
            integration_result[
                "detection_events"
            ]
        )

        # -----------------------------------------------------
        # 9. Round-by-round trace
        # -----------------------------------------------------

        rounds = []

        for index in range(self.rounds):
            rounds.append(
                {
                    "round": index + 1,

                    "physical_error_state": (
                        physical_error_history[index]
                    ),

                    "perfect_syndrome": (
                        perfect_history[index]
                    ),

                    "observed_syndrome": (
                        observed_history[index]
                    ),

                    "detection_event": (
                        detection_events[index]
                    ),
                }
            )

        # -----------------------------------------------------
        # 10. Physical recovery
        # -----------------------------------------------------

        physical_recovery = (
            measured_state
            == encoded_state
        )

        # -----------------------------------------------------
        # 11. Exact error match
        # -----------------------------------------------------

        predicted_error_string = (
            self._bits_to_string(
                predicted_error
            )
        )

        exact_error_match = (
            actual_error
            == predicted_error_string
        )

        # -----------------------------------------------------
        # 12. Return complete trace
        # -----------------------------------------------------

        return {
            "sample_id": sample["sample_id"],

            "qec_code": sample["qec_code"],

            "num_qubits": sample["num_qubits"],

            "rounds": sample["rounds"],

            "logical_state": logical_state,

            "encoded_state": encoded_state,

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
                    actual_error
                ),

                "final_error_description": (
                    sample[
                        "final_error_description"
                    ]
                ),
            },

            # -------------------------------------------------
            # TRUE quantum state progression
            # -------------------------------------------------

            "quantum_state": {
                "encoded": encoded_state,

                "corrupted": corrupted_state,
            },

            "syndrome": {
                "perfect_history": (
                    perfect_history
                ),

                "observed_history": (
                    observed_history
                ),

                "detection_events": (
                    detection_events
                ),

                "final_perfect": (
                    sample["final_syndrome"]
                ),

                "final_observed": (
                    sample[
                        "final_observed_syndrome"
                    ]
                ),

                "rounds": rounds,
            },

            "decoder": {
                "type": (
                    "logical_target_random_forest"
                ),

                "training_samples": (
                    len(training_samples)
                ),

                "random_forest_estimators": (
                    self.random_forest_estimators
                ),

                "predicted_correction": (
                    predicted_error_string
                ),

                "predicted_correction_bits": [
                    int(bit)
                    for bit in predicted_error
                ],

                "confidence": confidence,
            },

            "correction": {
                "actual_error": (
                    actual_error
                ),

                "predicted_correction": (
                    predicted_error_string
                ),

                "corrected_state": (
                    measured_state
                ),
            },

            "recovery": {
                "original_logical_state": (
                    logical_state
                ),

                "recovered_logical_state": (
                    integration_result[
                        "recovered_logical"
                    ]
                ),

                "logical_success": (
                    integration_result[
                        "logical_success"
                    ]
                ),

                "logical_failure": (
                    not integration_result[
                        "logical_success"
                    ]
                ),

                "physical_recovery": (
                    physical_recovery
                ),

                "exact_error_match": (
                    exact_error_match
                ),
            },
        }