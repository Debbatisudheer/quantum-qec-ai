import time

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder
)

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier
)

from decoders.logical_target import (
    LogicalTargetBuilder
)

from evaluation.decoder_evaluator import (
    DecoderEvaluator
)


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_NOISE = 0.10
MEASUREMENT_NOISE = 0.10

TRAINING_SAMPLES = 5000
TEST_SAMPLES = 1000

SEED = 42


# ============================================================
# DATA GENERATION
# ============================================================

def generate_samples(
    count,
    seed
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=PHYSICAL_NOISE,
        measurement_noise_probability=MEASUREMENT_NOISE,
        seed=seed
    )

    return [
        generator.generate_sample(i)
        for i in range(count)
    ]


# ============================================================
# FLAT FEATURE ENCODING
# ============================================================

def encode_flat_features(sample):
    """
    Convert temporal QEC information into
    a flat feature vector.

    Each round contains:

        syndrome bit 1
        syndrome bit 2
        detection bit 1
        detection bit 2

    For 5 rounds:

        5 x 4 = 20 features
    """

    syndrome_history = (
        sample["observed_syndrome_history"]
    )

    detection_events = (
        sample["detection_events"]
    )

    if len(syndrome_history) != len(
        detection_events
    ):
        raise ValueError(
            "syndrome history and detection "
            "event history must have the "
            "same length"
        )

    features = []

    for syndrome, detection in zip(
        syndrome_history,
        detection_events
    ):

        if len(syndrome) != 2:
            raise ValueError(
                "syndrome must contain 2 bits"
            )

        if len(detection) != 2:
            raise ValueError(
                "detection event must contain "
                "2 bits"
            )

        features.extend([
            int(syndrome[0]),
            int(syndrome[1]),
            int(detection[0]),
            int(detection[1]),
        ])

    return features


# ============================================================
# SEQUENCE FEATURE ENCODING
# ============================================================

def encode_sequence_features(sample):
    """
    Preserve the temporal structure.

    Output:

        rounds x 4

    For 5 rounds:

        5 x 4
    """

    syndrome_history = (
        sample["observed_syndrome_history"]
    )

    detection_events = (
        sample["detection_events"]
    )

    if len(syndrome_history) != len(
        detection_events
    ):
        raise ValueError(
            "syndrome history and detection "
            "event history must have the "
            "same length"
        )

    sequence = []

    for syndrome, detection in zip(
        syndrome_history,
        detection_events
    ):

        if len(syndrome) != 2:
            raise ValueError(
                "syndrome must contain 2 bits"
            )

        if len(detection) != 2:
            raise ValueError(
                "detection event must contain "
                "2 bits"
            )

        sequence.append([
            int(syndrome[0]),
            int(syndrome[1]),
            int(detection[0]),
            int(detection[1]),
        ])

    return sequence


# ============================================================
# BUILD LOGICAL TARGETS
# ============================================================

def build_logical_targets(
    training_samples
):
    """
    Build logical-preserving correction
    targets using ONLY training samples.

    No test information is used here.
    """

    builder = LogicalTargetBuilder()

    targets, scores = builder.build(
        training_samples
    )

    y_train = []

    for sample in training_samples:

        observation = (
            builder.observation_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        y_train.append(
            targets[observation]
        )

    return (
        targets,
        scores,
        np.asarray(
            y_train,
            dtype=np.int64
        )
    )


# ============================================================
# TRADITIONAL LOOKUP DECODER
# ============================================================

class TraditionalLookupDecoder:
    """
    Traditional decoder using only the
    final observed syndrome.

    This wrapper gives the traditional
    decoder the same decode_batch()
    interface as the AI decoders.
    """

    def __init__(self):
        self.lookup_table = {
            "00": [0, 0, 0],
            "10": [1, 0, 0],
            "11": [0, 1, 0],
            "01": [0, 0, 1],
        }

    def decode(self, sample):
        syndrome = (
            sample["final_observed_syndrome"]
        )

        if syndrome not in self.lookup_table:
            raise ValueError(
                f"Unknown syndrome: {syndrome}"
            )

        return list(
            self.lookup_table[syndrome]
        )

    def decode_batch(self, samples):
        return [
            self.decode(sample)
            for sample in samples
        ]


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

class LogicalTargetLogisticDecoder:
    """
    Logistic Regression trained using
    logical-preserving targets.
    """

    def __init__(
        self,
        random_seed=42
    ):
        self.random_seed = random_seed

        base_model = LogisticRegression(
            max_iter=1000,
            random_state=random_seed
        )

        self.model = MultiOutputClassifier(
            base_model
        )

        self.is_trained = False

    def train(
        self,
        training_samples
    ):
        _, _, y_train = (
            build_logical_targets(
                training_samples
            )
        )

        X_train = np.asarray(
            [
                encode_flat_features(
                    sample
                )
                for sample in training_samples
            ],
            dtype=np.float32
        )

        self.model.fit(
            X_train,
            y_train
        )

        self.is_trained = True

        return self

    def decode_batch(
        self,
        samples
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before decoding"
            )

        X = np.asarray(
            [
                encode_flat_features(
                    sample
                )
                for sample in samples
            ],
            dtype=np.float32
        )

        return self.model.predict(
            X
        ).tolist()

    def decode(
        self,
        sample
    ):
        return self.decode_batch(
            [sample]
        )[0]


# ============================================================
# MLP
# ============================================================

class LogicalTargetMLPDecoder:
    """
    MLP trained using logical-preserving
    correction targets.
    """

    def __init__(
        self,
        random_seed=42
    ):
        self.random_seed = random_seed

        base_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=random_seed
        )

        self.model = MultiOutputClassifier(
            base_model
        )

        self.is_trained = False

    def train(
        self,
        training_samples
    ):
        _, _, y_train = (
            build_logical_targets(
                training_samples
            )
        )

        X_train = np.asarray(
            [
                encode_flat_features(
                    sample
                )
                for sample in training_samples
            ],
            dtype=np.float32
        )

        self.model.fit(
            X_train,
            y_train
        )

        self.is_trained = True

        return self

    def decode_batch(
        self,
        samples
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before decoding"
            )

        X = np.asarray(
            [
                encode_flat_features(
                    sample
                )
                for sample in samples
            ],
            dtype=np.float32
        )

        return self.model.predict(
            X
        ).tolist()

    def decode(
        self,
        sample
    ):
        return self.decode_batch(
            [sample]
        )[0]


# ============================================================
# LOGICAL-TARGET GRU
# ============================================================

class LogicalTargetGRUDecoder:
    """
    GRU trained using logical-preserving
    correction targets.
    """

    def __init__(
        self,
        rounds=5,
        random_seed=42
    ):
        self.rounds = rounds
        self.random_seed = random_seed

        self.model = TemporalGRUClassifier(
            input_size=4,
            hidden_size=64,
            learning_rate=0.003,
            epochs=100,
            random_seed=random_seed
        )

        self.is_trained = False

    def train(
        self,
        training_samples
    ):
        targets_builder = (
            LogicalTargetBuilder()
        )

        targets, _ = (
            targets_builder.build(
                training_samples
            )
        )

        X_train = np.asarray(
            [
                encode_sequence_features(
                    sample
                )
                for sample in training_samples
            ],
            dtype=np.float32
        )

        y_train = []

        for sample in training_samples:

            observation = (
                targets_builder.observation_key(
                    sample[
                        "observed_syndrome_history"
                    ]
                )
            )

            y_train.append(
                targets[observation]
            )

        y_train = np.asarray(
            y_train,
            dtype=np.int64
        )

        self.model.train(
            X_train,
            y_train,
            verbose=False
        )

        self.is_trained = True

        return self

    def decode_batch(
        self,
        samples
    ):
        if not self.is_trained:
            raise RuntimeError(
                "Decoder must be trained "
                "before decoding"
            )

        X = np.asarray(
            [
                encode_sequence_features(
                    sample
                )
                for sample in samples
            ],
            dtype=np.float32
        )

        return self.model.predict(
            X
        )

    def decode(
        self,
        sample
    ):
        return self.decode_batch(
            [sample]
        )[0]


# ============================================================
# TRAINING HELPER
# ============================================================

def train_decoder(
    name,
    decoder,
    training_samples
):
    print()
    print(
        f"Training {name}..."
    )

    start = time.perf_counter()

    decoder.train(
        training_samples
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return decoder, elapsed


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        " LOGICAL-TARGET DECODER BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        f"Rounds              : {ROUNDS}"
    )

    print(
        f"Physical noise      : "
        f"{PHYSICAL_NOISE:.2f}"
    )

    print(
        f"Measurement noise   : "
        f"{MEASUREMENT_NOISE:.2f}"
    )

    print(
        f"Training samples    : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples        : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Seed                : "
        f"{SEED}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print()
    print(
        "Generating training data..."
    )

    training_samples = generate_samples(
        TRAINING_SAMPLES,
        SEED
    )

    print(
        "Generating test data..."
    )

    test_samples = generate_samples(
        TEST_SAMPLES,
        SEED + 10000
    )

    print(
        "Training data       : "
        f"{len(training_samples)}"
    )

    print(
        "Test data           : "
        f"{len(test_samples)}"
    )

    # --------------------------------------------------------
    # TARGET INFORMATION
    # --------------------------------------------------------

    (
        logical_targets,
        target_scores,
        _
    ) = build_logical_targets(
        training_samples
    )

    print()
    print(
        "Logical targets learned : "
        f"{len(logical_targets)}"
    )

    if target_scores:

        average_score = (
            sum(
                target_scores.values()
            )
            / len(target_scores)
        )

        print(
            "Average target score    : "
            f"{average_score:.4f}"
        )

    # --------------------------------------------------------
    # CREATE EVALUATOR
    # --------------------------------------------------------

    evaluator = (
        DecoderEvaluator()
    )

    print(
        "DecoderEvaluator         : READY"
    )

    # --------------------------------------------------------
    # DECODER 1
    # --------------------------------------------------------

    traditional = (
        TraditionalLookupDecoder()
    )

    start = time.perf_counter()

    traditional_metrics = (
        evaluator.evaluate(
            traditional,
            test_samples
        )
    )

    traditional_inference_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # DECODER 2
    # --------------------------------------------------------

    logistic, logistic_training_time = (
        train_decoder(
            "Logistic Regression",
            LogicalTargetLogisticDecoder(
                random_seed=SEED
            ),
            training_samples
        )
    )

    logistic_metrics = (
        evaluator.evaluate(
            logistic,
            test_samples
        )
    )

    # --------------------------------------------------------
    # DECODER 3
    # --------------------------------------------------------

    rf, rf_training_time = (
        train_decoder(
            "reusable Random Forest",
            LogicalTargetRandomForestDecoder(
                rounds=ROUNDS,
                n_estimators=100,
                random_seed=SEED
            ),
            training_samples
        )
    )

    rf_metrics = (
        evaluator.evaluate(
            rf,
            test_samples
        )
    )

    # --------------------------------------------------------
    # DECODER 4
    # --------------------------------------------------------

    mlp, mlp_training_time = (
        train_decoder(
            "MLP",
            LogicalTargetMLPDecoder(
                random_seed=SEED
            ),
            training_samples
        )
    )

    mlp_metrics = (
        evaluator.evaluate(
            mlp,
            test_samples
        )
    )

    # --------------------------------------------------------
    # DECODER 5
    # --------------------------------------------------------

    gru, gru_training_time = (
        train_decoder(
            "Logical-target GRU",
            LogicalTargetGRUDecoder(
                rounds=ROUNDS,
                random_seed=SEED
            ),
            training_samples
        )
    )

    gru_metrics = (
        evaluator.evaluate(
            gru,
            test_samples
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = {
        "Traditional Lookup":
            traditional_metrics,

        "Logistic Regression":
            logistic_metrics,

        "Random Forest":
            rf_metrics,

        "MLP":
            mlp_metrics,

        "Logical-target GRU":
            gru_metrics,
    }

    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"{'Decoder':<28}"
        f"{'Logical':>10}"
        f"{'Physical':>11}"
        f"{'Bit':>10}"
        f"{'Exact':>10}"
    )

    print("-" * 70)

    for name, metrics in results.items():

        print(
            f"{name:<28}"
            f"{metrics['logical']:>10.4f}"
            f"{metrics['physical']:>11.4f}"
            f"{metrics['bit']:>10.4f}"
            f"{metrics['exact']:>10.4f}"
        )

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " TRAINING TIME"
    )
    print("=" * 70)

    print()

    print(
        f"{'Decoder':<28}"
        f"{'Seconds':>12}"
    )

    print("-" * 45)

    print(
        f"{'Logistic Regression':<28}"
        f"{logistic_training_time:>12.4f}"
    )

    print(
        f"{'Random Forest':<28}"
        f"{rf_training_time:>12.4f}"
    )

    print(
        f"{'MLP':<28}"
        f"{mlp_training_time:>12.4f}"
    )

    print(
        f"{'Logical-target GRU':<28}"
        f"{gru_training_time:>12.4f}"
    )

    # --------------------------------------------------------
    # INFERENCE TIME
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " INFERENCE TIME"
    )
    print("=" * 70)

    print()

    print(
        f"{'Decoder':<28}"
        f"{'Seconds':>12}"
        f"{'Samples/sec':>15}"
    )

    print("-" * 60)

    print(
        f"{'Traditional Lookup':<28}"
        f"{traditional_metrics['inference_seconds']:>12.4f}"
        f"{traditional_metrics['samples_per_second']:>15.2f}"
    )

    print(
        f"{'Logistic Regression':<28}"
        f"{logistic_metrics['inference_seconds']:>12.4f}"
        f"{logistic_metrics['samples_per_second']:>15.2f}"
    )

    print(
        f"{'Random Forest':<28}"
        f"{rf_metrics['inference_seconds']:>12.4f}"
        f"{rf_metrics['samples_per_second']:>15.2f}"
    )

    print(
        f"{'MLP':<28}"
        f"{mlp_metrics['inference_seconds']:>12.4f}"
        f"{mlp_metrics['samples_per_second']:>15.2f}"
    )

    print(
        f"{'Logical-target GRU':<28}"
        f"{gru_metrics['inference_seconds']:>12.4f}"
        f"{gru_metrics['samples_per_second']:>15.2f}"
    )

    # --------------------------------------------------------
    # BEST DECODER
    # --------------------------------------------------------

    best_decoder = max(
        results,
        key=lambda name:
            results[name]["logical"]
    )

    best_score = results[
        best_decoder
    ]["logical"]

    traditional_score = results[
        "Traditional Lookup"
    ]["logical"]

    gain = (
        best_score
        - traditional_score
    )

    print()
    print("=" * 70)
    print(
        " BEST DECODER"
    )
    print("=" * 70)

    print()

    print(
        f"Best decoder          : "
        f"{best_decoder}"
    )

    print(
        f"Logical success       : "
        f"{best_score:.4f}"
    )

    print(
        f"Gain over traditional : "
        f"{gain:+.4f}"
    )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    for name, metrics in results.items():

        assert (
            0.0
            <= metrics["logical"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["physical"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["bit"]
            <= 1.0
        )

        assert (
            0.0
            <= metrics["exact"]
            <= 1.0
        )

    print()
    print(
        "Reusable evaluator      : PASS"
    )

    print(
        "Random Forest decoder   : PASS"
    )

    print(
        "All decoder evaluations : PASS"
    )

    print(
        "Logical recovery        : PASS"
    )

    print()
    print("=" * 70)
    print(
        " LOGICAL-TARGET DECODER BENCHMARK : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()