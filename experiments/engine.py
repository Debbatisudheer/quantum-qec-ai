import time
import uuid

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder
)

from evaluation.decoder_evaluator import (
    DecoderEvaluator
)

from experiments.config import (
    ExperimentConfig
)

from experiments.result import (
    ExperimentResult
)

from experiments.result_storage import (
    ExperimentResultStorage
)


class ExperimentEngine:
    """
    Unified end-to-end experiment engine.

    Responsibilities:

        1. Validate configuration
        2. Generate training dataset
        3. Generate test dataset
        4. Train decoder
        5. Evaluate decoder
        6. Produce ExperimentResult
        7. Optionally persist the result

    The engine orchestrates components.

    It does NOT implement:

        - quantum physics
        - noise mathematics
        - QEC mathematics
        - decoder algorithms
        - evaluation mathematics
        - JSON storage logic
    """

    def __init__(
        self,
        config: ExperimentConfig,
        storage=None
    ):
        if not isinstance(
            config,
            ExperimentConfig
        ):
            raise TypeError(
                "config must be an ExperimentConfig"
            )

        config.validate()

        if storage is not None:
            if not isinstance(
                storage,
                ExperimentResultStorage
            ):
                raise TypeError(
                    "storage must be an "
                    "ExperimentResultStorage"
                )

        self.config = config

        self.storage = storage

        self.evaluator = (
            DecoderEvaluator()
        )

    # ========================================================
    # DATA GENERATION
    # ========================================================

    def generate_samples(
        self,
        count,
        seed
    ):
        """
        Generate time-varying QEC samples.
        """

        generator = (
            TimeVaryingQECDatasetGenerator(
                rounds=self.config.rounds,

                physical_error_probability=(
                    self.config
                    .physical_noise_probability
                ),

                measurement_noise_probability=(
                    self.config
                    .measurement_noise_probability
                ),

                seed=seed
            )
        )

        return [
            generator.generate_sample(
                sample_id=i
            )
            for i in range(count)
        ]

    # ========================================================
    # DECODER CREATION
    # ========================================================

    def create_decoder(self):
        """
        Create the decoder selected by the
        experiment configuration.
        """

        if (
            self.config.decoder_type
            == "logical_target_random_forest"
        ):
            return (
                LogicalTargetRandomForestDecoder(
                    rounds=self.config.rounds,

                    n_estimators=(
                        self.config
                        .random_forest_estimators
                    ),

                    random_seed=self.config.seed
                )
            )

        raise ValueError(
            f"Unsupported decoder type: "
            f"{self.config.decoder_type}"
        )

    # ========================================================
    # CONFIGURATION SERIALIZATION
    # ========================================================

    def config_to_dict(self):
        """
        Convert configuration into a
        serializable dictionary.
        """

        return {
            "qec_code": (
                self.config.qec_code
            ),

            "num_qubits": (
                self.config.num_qubits
            ),

            "logical_state": (
                self.config.logical_state
            ),

            "rounds": (
                self.config.rounds
            ),

            "physical_noise_probability": (
                self.config
                .physical_noise_probability
            ),

            "measurement_noise_probability": (
                self.config
                .measurement_noise_probability
            ),

            "training_samples": (
                self.config.training_samples
            ),

            "test_samples": (
                self.config.test_samples
            ),

            "decoder_type": (
                self.config.decoder_type
            ),

            "random_forest_estimators": (
                self.config
                .random_forest_estimators
            ),

            "seed": (
                self.config.seed
            ),
        }

    # ========================================================
    # SAVE RESULT
    # ========================================================

    def save_result(
        self,
        result
    ):
        """
        Save an ExperimentResult when storage
        has been configured.
        """

        if self.storage is None:
            return None

        return self.storage.save(
            result
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self):
        """
        Execute one complete experiment.
        """

        experiment_id = (
            uuid.uuid4().hex[:12]
        )

        print()
        print("=" * 70)
        print(
            " UNIFIED QEC EXPERIMENT"
        )
        print("=" * 70)

        print()
        print(
            f"Experiment ID       : "
            f"{experiment_id}"
        )

        print(
            f"QEC code            : "
            f"{self.config.qec_code}"
        )

        print(
            f"Rounds              : "
            f"{self.config.rounds}"
        )

        print(
            f"Physical noise      : "
            f"{self.config.physical_noise_probability:.2f}"
        )

        print(
            f"Measurement noise   : "
            f"{self.config.measurement_noise_probability:.2f}"
        )

        print(
            f"Training samples    : "
            f"{self.config.training_samples}"
        )

        print(
            f"Test samples        : "
            f"{self.config.test_samples}"
        )

        print(
            f"Decoder             : "
            f"{self.config.decoder_type}"
        )

        print()

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        print(
            "Generating training dataset..."
        )

        training_samples = (
            self.generate_samples(
                self.config.training_samples,
                self.config.seed
            )
        )

        print(
            f"Training dataset    : "
            f"{len(training_samples)} samples"
        )

        # ----------------------------------------------------
        # TEST DATA
        # ----------------------------------------------------

        print(
            "Generating test dataset..."
        )

        test_samples = (
            self.generate_samples(
                self.config.test_samples,
                self.config.seed + 10000
            )
        )

        print(
            f"Test dataset        : "
            f"{len(test_samples)} samples"
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        decoder = (
            self.create_decoder()
        )

        print()
        print(
            "Decoder created     : PASS"
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        print()
        print(
            "Training decoder..."
        )

        training_start = (
            time.perf_counter()
        )

        decoder.train(
            training_samples
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        print(
            f"Training complete   : "
            f"{training_seconds:.4f} sec"
        )

        # ----------------------------------------------------
        # TARGET INFORMATION
        # ----------------------------------------------------

        logical_targets_learned = (
            len(decoder.targets)
        )

        if decoder.target_scores:

            average_target_score = (
                sum(
                    decoder.target_scores.values()
                )
                / len(
                    decoder.target_scores
                )
            )

        else:

            average_target_score = 0.0

        print(
            f"Logical targets     : "
            f"{logical_targets_learned}"
        )

        print(
            f"Average target score: "
            f"{average_target_score:.4f}"
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        print()
        print(
            "Evaluating decoder..."
        )

        metrics = (
            self.evaluator.evaluate(
                decoder,
                test_samples
            )
        )

        print(
            "Evaluation complete : PASS"
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = ExperimentResult(

            experiment_id=(
                experiment_id
            ),

            config=(
                self.config_to_dict()
            ),

            training_samples=(
                len(training_samples)
            ),

            test_samples=(
                len(test_samples)
            ),

            logical_targets_learned=(
                logical_targets_learned
            ),

            average_target_score=(
                average_target_score
            ),

            exact_accuracy=(
                metrics["exact"]
            ),

            physical_accuracy=(
                metrics["physical"]
            ),

            bit_accuracy=(
                metrics["bit"]
            ),

            logical_accuracy=(
                metrics["logical"]
            ),

            training_seconds=(
                training_seconds
            ),

            inference_seconds=(
                metrics[
                    "inference_seconds"
                ]
            ),

            samples_per_second=(
                metrics[
                    "samples_per_second"
                ]
            ),

            decoder_type=(
                self.config.decoder_type
            ),
        )

        # ----------------------------------------------------
        # PERSIST RESULT
        # ----------------------------------------------------

        saved_path = (
            self.save_result(
                result
            )
        )

        if saved_path is not None:

            print()
            print(
                "Result storage       : PASS"
            )

            print(
                f"Saved result        : "
                f"{saved_path}"
            )

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            " EXPERIMENT RESULT"
        )
        print("=" * 70)

        print()

        print(
            f"Experiment ID       : "
            f"{result.experiment_id}"
        )

        print(
            f"Decoder             : "
            f"{result.decoder_type}"
        )

        print(
            f"Exact accuracy      : "
            f"{result.exact_accuracy:.4f}"
        )

        print(
            f"Physical recovery   : "
            f"{result.physical_accuracy:.4f}"
        )

        print(
            f"Bit accuracy        : "
            f"{result.bit_accuracy:.4f}"
        )

        print(
            f"Logical success     : "
            f"{result.logical_accuracy:.4f}"
        )

        print(
            f"Training time       : "
            f"{result.training_seconds:.4f} sec"
        )

        print(
            f"Inference time      : "
            f"{result.inference_seconds:.4f} sec"
        )

        print(
            f"Throughput          : "
            f"{result.samples_per_second:.2f} "
            f"samples/sec"
        )

        print()

        return result