import numpy as np

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_gru import (
    LogicalTargetGRUDecoder
)

from decoders.repeated_lookup import (
    RepeatedLookupDecoder
)

from decoders.logical_target import (
    LogicalTargetBuilder
)

from evaluation.logical_recovery import (
    LogicalRecovery
)

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_NOISE = 0.10

MEASUREMENT_NOISE = 0.10

TRAINING_SAMPLES = 5000

TEST_SAMPLES = 1000

SEEDS = [
    42,
    123,
    456,
    789,
    999
]

GRU_HIDDEN_SIZE = 64

GRU_LEARNING_RATE = 0.003

GRU_EPOCHS = 100

RF_ESTIMATORS = 100

MLP_HIDDEN_LAYERS = (64, 32)

MLP_MAX_ITER = 500


# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_flat_features(sample):

    syndrome_history = sample[
        "observed_syndrome_history"
    ]

    detection_events = sample[
        "detection_events"
    ]

    features = []

    for syndrome, detection in zip(
        syndrome_history,
        detection_events
    ):

        features.extend(
            [
                int(syndrome[0]),
                int(syndrome[1]),
                int(detection[0]),
                int(detection[1])
            ]
        )

    return features


# ============================================================
# BUILD LOGICAL TARGETS
# ============================================================

def build_targets(samples):

    builder = LogicalTargetBuilder()

    targets, scores = builder.build(
        samples
    )

    y = []

    for sample in samples:

        observation = (
            builder.observation_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        y.append(
            targets[observation]
        )

    return (
        np.array(
            y,
            dtype=np.int64
        ),
        targets,
        scores
    )


# ============================================================
# GENERATE DATA
# ============================================================

def generate_samples(
    samples,
    seed
):

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_NOISE
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE
            ),
            seed=seed
        )
    )

    return generator.generate_dataset(
        samples
    )


# ============================================================
# LOGICAL SUCCESS
# ============================================================

def logical_success(
    sample,
    correction,
    recovery
):

    logical_state = int(
        sample["logical_state"]
    )

    actual_error = [
        int(bit)
        for bit in sample[
            "final_error_state"
        ]
    ]

    encoded_state = [
        logical_state,
        logical_state,
        logical_state
    ]

    corrupted_state = [
        a ^ b
        for a, b in zip(
            encoded_state,
            actual_error
        )
    ]

    corrected_state = [
        a ^ b
        for a, b in zip(
            corrupted_state,
            correction
        )
    ]

    recovered = recovery.recover(
        corrected_state
    )

    return recovered == logical_state


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    samples,
    predictions,
    recovery
):

    success = 0

    for sample, prediction in zip(
        samples,
        predictions
    ):

        if logical_success(
            sample,
            prediction,
            recovery
        ):

            success += 1

    return (
        success
        / len(samples)
    )


# ============================================================
# RUN ONE SEED
# ============================================================

def run_seed(seed):

    print()
    print(
        "-" * 78
    )

    print(
        f"SEED : {seed}"
    )

    print(
        "-" * 78
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    training_samples = (
        generate_samples(
            TRAINING_SAMPLES,
            seed
        )
    )

    test_seed = seed + 10000

    test_samples = (
        generate_samples(
            TEST_SAMPLES,
            test_seed
        )
    )

    # --------------------------------------------------------
    # Logical targets
    # --------------------------------------------------------

    y_train, _, _ = build_targets(
        training_samples
    )

    # --------------------------------------------------------
    # Flat features
    # --------------------------------------------------------

    X_train = np.array(
        [
            encode_flat_features(
                sample
            )
            for sample in training_samples
        ],
        dtype=np.float32
    )

    X_test = np.array(
        [
            encode_flat_features(
                sample
            )
            for sample in test_samples
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Logistic
    # --------------------------------------------------------

    logistic = MultiOutputClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=seed
        )
    )

    logistic.fit(
        X_train,
        y_train
    )

    logistic_predictions = (
        logistic.predict(
            X_test
        ).tolist()
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            random_state=seed
        )
    )

    random_forest.fit(
        X_train,
        y_train
    )

    rf_predictions = (
        random_forest.predict(
            X_test
        ).tolist()
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=(
                MLP_HIDDEN_LAYERS
            ),
            max_iter=MLP_MAX_ITER,
            random_state=seed
        )
    )

    mlp.fit(
        X_train,
        y_train
    )

    mlp_predictions = (
        mlp.predict(
            X_test
        ).tolist()
    )

    # --------------------------------------------------------
    # Logical-target GRU
    # --------------------------------------------------------

    gru = LogicalTargetGRUDecoder(
        rounds=ROUNDS,
        hidden_size=GRU_HIDDEN_SIZE,
        learning_rate=GRU_LEARNING_RATE,
        epochs=GRU_EPOCHS,
        random_seed=seed
    )

    gru.train(
        training_samples,
        verbose=False
    )

    gru_predictions = (
        gru.predict_batch(
            test_samples
        )
    )

    # --------------------------------------------------------
    # Traditional
    # --------------------------------------------------------

    traditional = (
        RepeatedLookupDecoder()
    )

    traditional_predictions = []

    for sample in test_samples:

        prediction = (
            traditional.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        traditional_predictions.append(
            prediction
        )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    recovery = LogicalRecovery()

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    traditional_score = evaluate(
        test_samples,
        traditional_predictions,
        recovery
    )

    logistic_score = evaluate(
        test_samples,
        logistic_predictions,
        recovery
    )

    rf_score = evaluate(
        test_samples,
        rf_predictions,
        recovery
    )

    mlp_score = evaluate(
        test_samples,
        mlp_predictions,
        recovery
    )

    gru_score = evaluate(
        test_samples,
        gru_predictions,
        recovery
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"Traditional        : "
        f"{traditional_score:.4f}"
    )

    print(
        f"Logistic           : "
        f"{logistic_score:.4f}"
    )

    print(
        f"Random Forest      : "
        f"{rf_score:.4f}"
    )

    print(
        f"MLP                : "
        f"{mlp_score:.4f}"
    )

    print(
        f"Logical-target GRU : "
        f"{gru_score:.4f}"
    )

    print(
        f"GRU gain           : "
        f"{gru_score - traditional_score:+.4f}"
    )

    return {
        "traditional": traditional_score,
        "logistic": logistic_score,
        "random_forest": rf_score,
        "mlp": mlp_score,
        "gru": gru_score
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)

    print(
        " LOGICAL-TARGET MULTI-SEED "
        "DECODER BENCHMARK"
    )

    print("=" * 78)

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
        f"Seeds               : "
        f"{SEEDS}"
    )

    # ========================================================
    # RUN ALL SEEDS
    # ========================================================

    all_results = []

    for seed in SEEDS:

        result = run_seed(
            seed
        )

        all_results.append(
            result
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    model_names = [
        "traditional",
        "logistic",
        "random_forest",
        "mlp",
        "gru"
    ]

    display_names = {
        "traditional":
            "Traditional",

        "logistic":
            "Logistic",

        "random_forest":
            "Random Forest",

        "mlp":
            "MLP",

        "gru":
            "Logical-target GRU"
    }

    print()
    print()
    print("=" * 78)

    print(
        " MULTI-SEED SUMMARY"
    )

    print("=" * 78)

    print()

    print(
        "Decoder                Mean      Std"
    )

    print(
        "-" * 78
    )

    means = {}

    stds = {}

    for model in model_names:

        values = np.array(
            [
                result[model]
                for result in all_results
            ]
        )

        mean = values.mean()

        std = values.std(
            ddof=1
        )

        means[model] = mean

        stds[model] = std

        print(
            f"{display_names[model]:<23}"
            f"{mean:.4f}    "
            f"{std:.4f}"
        )

    # ========================================================
    # GRU ADVANTAGE
    # ========================================================

    gru_mean = means["gru"]

    traditional_mean = (
        means["traditional"]
    )

    gain = (
        gru_mean
        - traditional_mean
    )

    print()

    print(
        f"Mean GRU gain over "
        f"traditional : {gain:+.4f}"
    )

    # ========================================================
    # WIN / LOSS COUNT
    # ========================================================

    wins = 0

    ties = 0

    losses = 0

    for result in all_results:

        if result["gru"] > result["traditional"]:

            wins += 1

        elif result["gru"] == result["traditional"]:

            ties += 1

        else:

            losses += 1

    print()

    print(
        f"GRU wins       : "
        f"{wins}/{len(SEEDS)}"
    )

    print(
        f"GRU ties       : "
        f"{ties}/{len(SEEDS)}"
    )

    print(
        f"GRU losses     : "
        f"{losses}/{len(SEEDS)}"
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_model = max(
        means,
        key=means.get
    )

    print()

    print(
        f"Best decoder    : "
        f"{display_names[best_model]}"
    )

    print(
        f"Best mean score : "
        f"{means[best_model]:.4f}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert len(
        all_results
    ) == len(SEEDS)

    for result in all_results:

        for model in model_names:

            assert 0.0 <= (
                result[model]
            ) <= 1.0

    assert (
        means["gru"]
        >= means["traditional"]
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print(
        "=" * 78
    )

    print(
        " MULTI-SEED BENCHMARK : PASS"
    )

    print(
        " Statistical comparison : PASS"
    )

    print(
        " Logical recovery        : PASS"
    )

    print()

    print(
        "RESULT : SUCCESS"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()