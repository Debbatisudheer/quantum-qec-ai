from noise.stochastic_repeated_noise import (
    StochasticRepeatedBitFlipNoise
)

from quantum.repeated_qec import (
    RepeatedQuantumQEC
)

from quantum.repeated_measurement import (
    RepeatedQuantumMeasurementParser
)

from syndrome.measurement_noise import (
    SyndromeMeasurementNoise
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


class StochasticQuantumAIExperiment:
    """
    End-to-end stochastic quantum QEC experiment.

    Pipeline:

        stochastic physical noise
                ↓
        repeated quantum QEC
                ↓
        quantum syndrome history
                ↓
        measurement noise
                ↓
        AI decoder
                ↓
        predicted physical error
                ↓
        correction
                ↓
        logical recovery
    """

    def __init__(
        self,
        rounds=5,
        physical_error_probability=0.10,
        measurement_noise_probability=0.10,
        seed=42
    ):
        if rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= physical_error_probability <= 1.0:
            raise ValueError(
                "physical_error_probability must be between 0 and 1"
            )

        if not 0.0 <= measurement_noise_probability <= 1.0:
            raise ValueError(
                "measurement_noise_probability must be between 0 and 1"
            )

        self.rounds = rounds

        self.physical_error_probability = (
            physical_error_probability
        )

        self.measurement_noise_probability = (
            measurement_noise_probability
        )

        self.seed = seed

        self.noise_generator = (
            StochasticRepeatedBitFlipNoise(
                rounds=rounds,
                probability=physical_error_probability,
                seed=seed
            )
        )

        self.measurement_noise = (
            SyndromeMeasurementNoise(
                probability=measurement_noise_probability,
                seed=seed + 1000
            )
        )

        self.quantum_qec = (
            RepeatedQuantumQEC(
                rounds=rounds
            )
        )

        self.measurement_parser = (
            RepeatedQuantumMeasurementParser(
                rounds=rounds
            )
        )

        self.logical_recovery = (
            LogicalRecovery()
        )

    def calculate_detection_events(
        self,
        syndrome_history
    ):
        """
        Calculate detection events from
        observed syndrome history.

        Each event is:

            previous syndrome XOR
            current syndrome
        """

        if not syndrome_history:
            raise ValueError(
                "syndrome_history cannot be empty"
            )

        events = []

        previous = "00"

        for syndrome in syndrome_history:

            if not isinstance(
                syndrome,
                str
            ):
                raise ValueError(
                    "syndrome must be a string"
                )

            if len(syndrome) != 2:
                raise ValueError(
                    "Each syndrome must contain 2 bits"
                )

            if any(
                bit not in "01"
                for bit in syndrome
            ):
                raise ValueError(
                    "syndrome must contain only 0 and 1"
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

    def apply_measurement_noise(
        self,
        syndrome_history
    ):
        """
        Apply software-level syndrome
        measurement noise.

        This represents an observed noisy
        syndrome, not physical ancilla noise.
        """

        observed = []

        for syndrome in syndrome_history:

            observed_syndrome = (
                self.measurement_noise.apply(
                    syndrome
                )
            )

            observed.append(
                observed_syndrome
            )

        return observed

    def encode_features(
        self,
        observed_syndrome_history,
        detection_events
    ):
        """
        Convert observable syndrome information
        into the AI feature vector.

        For 5 rounds:

            syndrome history = 10 bits
            detection events  = 10 bits

            total = 20 features
        """

        features = []

        for syndrome in (
            observed_syndrome_history
        ):

            for bit in syndrome:

                features.append(
                    int(bit)
                )

        for event in detection_events:

            for bit in event:

                features.append(
                    int(bit)
                )

        return features

    def run_quantum_trial(
        self,
        logical_state,
        error_history
    ):
        """
        Build and execute one repeated quantum
        QEC experiment.
        """

        circuit = (
            self.quantum_qec
            .create_round_by_round_circuit(
                logical_state=logical_state,
                physical_error_history=error_history
            )
        )

        from qiskit_aer import (
            AerSimulator
        )

        backend = AerSimulator()

        result = backend.run(
            circuit,
            shots=1
        ).result()

        counts = result.get_counts()

        raw_measurement = (
            self.measurement_parser
            .extract_most_likely_result(
                counts
            )
        )

        parsed = (
            self.measurement_parser.parse(
                raw_measurement
            )
        )

        return {
            "raw_measurement":
                raw_measurement,

            "final_state":
                parsed["final_state"],

            "syndrome_history":
                parsed[
                    "syndrome_history"
                ]
        }

    def correct_state(
        self,
        actual_error_state,
        predicted_error_state,
        logical_state
    ):
        """
        Apply the predicted X correction
        to the actual accumulated physical
        error state.

        Because Pauli-X satisfies:

            X * X = I

        correction is represented by XOR.

        Example:

            actual    = [1,0,1]
            predicted = [1,0,0]

            corrected = [0,0,1]
        """

        if len(actual_error_state) != 3:
            raise ValueError(
                "actual_error_state must contain 3 bits"
            )

        if len(predicted_error_state) != 3:
            raise ValueError(
                "predicted_error_state must contain 3 bits"
            )

        corrected_state = []

        for actual, predicted in zip(
            actual_error_state,
            predicted_error_state
        ):

            actual_bit = int(actual)
            predicted_bit = int(predicted)

            if actual_bit not in (0, 1):
                raise ValueError(
                    "actual_error_state must contain only 0 and 1"
                )

            if predicted_bit not in (0, 1):
                raise ValueError(
                    "predicted_error_state must contain only 0 and 1"
                )

            corrected_state.append(
                actual_bit ^ predicted_bit
            )

        # IMPORTANT:
        #
        # LogicalRecovery expects a list of
        # integer bits, not a string.
        final_state = corrected_state.copy()

        recovered_logical = (
            self.logical_recovery.recover(
                final_state
            )
        )

        logical_success = (
            recovered_logical == logical_state
        )

        physical_success = (
            corrected_state
            == [0, 0, 0]
        )

        return {
            "corrected_state":
                corrected_state,

            "recovered_logical":
                recovered_logical,

            "logical_success":
                logical_success,

            "physical_success":
                physical_success
        }

    def run_trial(
        self,
        decoder,
        logical_state=None
    ):
        """
        Run one complete stochastic
        quantum + AI trial.
        """

        if logical_state is None:

            logical_state = (
                self.noise_generator
                .noise
                .random
                .choice(
                    [0, 1]
                )
            )

        # ------------------------------------------------
        # 1. Generate stochastic physical errors.
        # ------------------------------------------------

        error_history = (
            self.noise_generator
            .generate_error_history()
        )

        # ------------------------------------------------
        # 2. Run the repeated quantum QEC circuit.
        # ------------------------------------------------

        quantum_result = (
            self.run_quantum_trial(
                logical_state=logical_state,
                error_history=error_history
            )
        )

        # ------------------------------------------------
        # 3. Get perfect quantum syndrome history.
        # ------------------------------------------------

        perfect_syndrome_history = (
            quantum_result[
                "syndrome_history"
            ]
        )

        # ------------------------------------------------
        # 4. Apply measurement/readout noise.
        # ------------------------------------------------

        observed_syndrome_history = (
            self.apply_measurement_noise(
                perfect_syndrome_history
            )
        )

        # ------------------------------------------------
        # 5. Calculate detection events.
        # ------------------------------------------------

        detection_events = (
            self.calculate_detection_events(
                observed_syndrome_history
            )
        )

        # ------------------------------------------------
        # 6. Encode observable information.
        # ------------------------------------------------

        features = (
            self.encode_features(
                observed_syndrome_history,
                detection_events
            )
        )

        # ------------------------------------------------
        # 7. AI predicts the physical error pattern.
        # ------------------------------------------------

        predicted_error = (
            decoder.decode(
                features
            )
        )

        # ------------------------------------------------
        # 8. Ground truth.
        #
        # This is NOT supplied to the decoder.
        # It is used only for evaluation.
        # ------------------------------------------------

        actual_error = (
            error_history[-1]
        )

        # ------------------------------------------------
        # 9. Apply predicted correction.
        # ------------------------------------------------

        correction_result = (
            self.correct_state(
                actual_error_state=
                    actual_error,

                predicted_error_state=
                    predicted_error,

                logical_state=
                    logical_state
            )
        )

        # ------------------------------------------------
        # 10. Return complete trial result.
        # ------------------------------------------------

        return {
            "logical_state":
                logical_state,

            "error_history":
                error_history,

            "actual_error":
                actual_error,

            "perfect_syndrome_history":
                perfect_syndrome_history,

            "observed_syndrome_history":
                observed_syndrome_history,

            "detection_events":
                detection_events,

            "features":
                features,

            "predicted_error":
                predicted_error,

            **correction_result
        }

    def run_experiment(
        self,
        decoder,
        num_trials=100
    ):
        """
        Run multiple independent quantum + AI
        trials and calculate aggregate metrics.
        """

        if num_trials <= 0:
            raise ValueError(
                "num_trials must be greater than 0"
            )

        results = []

        for _ in range(num_trials):

            trial_result = (
                self.run_trial(
                    decoder
                )
            )

            results.append(
                trial_result
            )

        physical_success_count = sum(
            1
            for result in results
            if result["physical_success"]
        )

        logical_success_count = sum(
            1
            for result in results
            if result["logical_success"]
        )

        physical_success = (
            physical_success_count
            / num_trials
        )

        logical_success = (
            logical_success_count
            / num_trials
        )

        logical_error_rate = (
            1.0
            - logical_success
        )

        return {
            "physical_success":
                physical_success,

            "logical_success":
                logical_success,

            "logical_error_rate":
                logical_error_rate,

            "results":
                results
        }