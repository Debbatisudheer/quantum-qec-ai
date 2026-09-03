import numpy as np

from dataset.time_varying_generator import TimeVaryingQECDatasetGenerator

from decoders.logical_target import LogicalTargetBuilder
from decoders.logical_target_gru import LogicalTargetGRUDecoder
from decoders.repeated_lookup import RepeatedLookupDecoder

from evaluation.logical_recovery import LogicalRecovery

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS_LIST = [3, 5, 7]

NOISE_CONFIGS = [
    (0.05, 0.05),
    (0.10, 0.05),
    (0.10, 0.10),
    (0.10, 0.20),
    (0.20, 0.10),
]

SEEDS = [42, 123, 456]

TRAINING_SAMPLES = 5000
TEST_SAMPLES = 1000

GRU_HIDDEN_SIZE = 64
GRU_LEARNING_RATE = 0.003
GRU_EPOCHS = 100

RF_ESTIMATORS = 100

MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER = 500


# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_features(sample):
    """
    Convert:

        observed syndrome history
        +
        detection event history

    into a flat feature vector.

    Each round contributes:

        syndrome bit 1
        syndrome bit 2
        detection bit 1
        detection bit 2
    """

    syndrome_history = sample["observed_syndrome_history"]
    detection_events = sample["detection_events"]

    if len(syndrome_history) != len(detection_events):
        raise ValueError(
            "syndrome history and detection event history "
            "must have the same length"
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
                "detection event must contain 2 bits"
            )

        features.extend([
            int(syndrome[0]),
            int(syndrome[1]),
            int(detection[0]),
            int(detection[1]),
        ])

    return features


def encode_sequence(sample):
    """
    GRU input:

        rounds x 4

    Each round:

        [syndrome1, syndrome2,
         detection1, detection2]
    """

    syndrome_history = sample["observed_syndrome_history"]
    detection_events = sample["detection_events"]

    sequence = []

    for syndrome, detection in zip(
        syndrome_history,
        detection_events
    ):
        sequence.append([
            int(syndrome[0]),
            int(syndrome[1]),
            int(detection[0]),
            int(detection[1]),
        ])

    return sequence


# ============================================================
# LOGICAL RECOVERY
# ============================================================

def logical_success(
    sample,
    predicted_correction,
    recovery
):
    """
    Evaluate whether the predicted correction
    preserves the original logical state.

    Flow:

        encoded state
              ↓
        actual physical error
              ↓
        corrupted state
              ↓
        predicted correction
              ↓
        corrected state
              ↓
        logical recovery
    """

    encoded_state = [
        int(bit)
        for bit in sample["encoded_state"]
    ]

    actual_error = [
        int(bit)
        for bit in sample["final_error_state"]
    ]

    correction = [
        int(bit)
        for bit in predicted_correction
    ]

    corrupted_state = [
        encoded_state[i] ^ actual_error[i]
        for i in range(3)
    ]

    corrected_state = [
        corrupted_state[i] ^ correction[i]
        for i in range(3)
    ]

    recovered_logical = recovery.recover(
        corrected_state
    )

    return (
        recovered_logical
        == sample["logical_state"]
    )


# ============================================================
# DATA GENERATION
# ============================================================

def generate_samples(
    rounds,
    physical_noise,
    measurement_noise,
    count,
    seed
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=rounds,
        physical_error_probability=physical_noise,
        measurement_noise_probability=measurement_noise,
        seed=seed
    )

    return [
        generator.generate_sample(i)
        for i in range(count)
    ]


# ============================================================
# BUILD LOGICAL TARGETS
# ============================================================

def build_training_targets(samples):
    builder = LogicalTargetBuilder()

    targets, scores = builder.build(
        samples
    )

    X = np.array(
        [
            encode_features(sample)
            for sample in samples
        ],
        dtype=np.float32
    )

    y = np.array(
        [
            targets[
                builder.observation_key(
                    sample["observed_syndrome_history"]
                )
            ]
            for sample in samples
        ],
        dtype=np.int64
    )

    return builder, targets, scores, X, y


# ============================================================
# TRAIN CLASSICAL MODELS
# ============================================================

def train_logistic(X, y):
    model = MultiOutputClassifier(
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

    model.fit(X, y)

    return model


def train_random_forest(X, y, seed):
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            random_state=seed
        )
    )

    model.fit(X, y)

    return model


def train_mlp(X, y, seed):
    model = MultiOutputClassifier(
        MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            random_state=seed
        )
    )

    model.fit(X, y)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    samples,
    predictions,
    recovery
):
    successes = 0

    for sample, prediction in zip(
        samples,
        predictions
    ):
        if logical_success(
            sample,
            prediction,
            recovery
        ):
            successes += 1

    return successes / len(samples)


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    rounds,
    physical_noise,
    measurement_noise,
    seed
):
    print()
    print("=" * 78)
    print(
        f"ROUNDS={rounds} | "
        f"PHYSICAL={physical_noise:.2f} | "
        f"MEASUREMENT={measurement_noise:.2f} | "
        f"SEED={seed}"
    )
    print("=" * 78)

    train_samples = generate_samples(
        rounds=rounds,
        physical_noise=physical_noise,
        measurement_noise=measurement_noise,
        count=TRAINING_SAMPLES,
        seed=seed
    )

    test_samples = generate_samples(
        rounds=rounds,
        physical_noise=physical_noise,
        measurement_noise=measurement_noise,
        count=TEST_SAMPLES,
        seed=seed + 10000
    )

    print(
        f"Training samples : {len(train_samples)}"
    )

    print(
        f"Test samples     : {len(test_samples)}"
    )

    builder, targets, scores, X_train, y_train = (
        build_training_targets(
            train_samples
        )
    )

    X_test = np.array(
        [
            encode_features(sample)
            for sample in test_samples
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # TRADITIONAL
    # --------------------------------------------------------

    traditional = RepeatedLookupDecoder()

    traditional_predictions = []

    for sample in test_samples:
        prediction = traditional.decode_history(
            sample["observed_syndrome_history"]
        )

        traditional_predictions.append(
            prediction
        )

    # --------------------------------------------------------
    # LOGISTIC
    # --------------------------------------------------------

    logistic = train_logistic(
        X_train,
        y_train
    )

    logistic_predictions = (
        logistic.predict(X_test).tolist()
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    random_forest = train_random_forest(
        X_train,
        y_train,
        seed
    )

    rf_predictions = (
        random_forest.predict(X_test).tolist()
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    mlp = train_mlp(
        X_train,
        y_train,
        seed
    )

    mlp_predictions = (
        mlp.predict(X_test).tolist()
    )

    # --------------------------------------------------------
    # LOGICAL-TARGET GRU
    # --------------------------------------------------------

    gru = LogicalTargetGRUDecoder(
        rounds=rounds,
        hidden_size=GRU_HIDDEN_SIZE,
        learning_rate=GRU_LEARNING_RATE,
        epochs=GRU_EPOCHS,
        random_seed=seed
    )

    gru.train(
        train_samples,
        verbose=False
    )

    gru_predictions = gru.predict_batch(
        test_samples
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    recovery = LogicalRecovery()

    results = {
        "traditional": evaluate_predictions(
            test_samples,
            traditional_predictions,
            recovery
        ),

        "logistic": evaluate_predictions(
            test_samples,
            logistic_predictions,
            recovery
        ),

        "random_forest": evaluate_predictions(
            test_samples,
            rf_predictions,
            recovery
        ),

        "mlp": evaluate_predictions(
            test_samples,
            mlp_predictions,
            recovery
        ),

        "gru": evaluate_predictions(
            test_samples,
            gru_predictions,
            recovery
        ),
    }

    print()
    print(
        f"Traditional        : "
        f"{results['traditional']:.4f}"
    )

    print(
        f"Logistic           : "
        f"{results['logistic']:.4f}"
    )

    print(
        f"Random Forest      : "
        f"{results['random_forest']:.4f}"
    )

    print(
        f"MLP                : "
        f"{results['mlp']:.4f}"
    )

    print(
        f"Logical-target GRU : "
        f"{results['gru']:.4f}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(" FINAL DECODER SELECTION EXPERIMENT")
    print("=" * 78)

    print()
    print("Rounds       :", ROUNDS_LIST)
    print("Noise cases  :", len(NOISE_CONFIGS))
    print("Seeds        :", SEEDS)
    print("Training     :", TRAINING_SAMPLES)
    print("Testing      :", TEST_SAMPLES)

    all_results = {
        "traditional": [],
        "logistic": [],
        "random_forest": [],
        "mlp": [],
        "gru": [],
    }

    total_experiments = (
        len(ROUNDS_LIST)
        * len(NOISE_CONFIGS)
        * len(SEEDS)
    )

    experiment_number = 0

    # --------------------------------------------------------
    # ALL EXPERIMENTS
    # --------------------------------------------------------

    for rounds in ROUNDS_LIST:

        for physical_noise, measurement_noise in (
            NOISE_CONFIGS
        ):

            for seed in SEEDS:

                experiment_number += 1

                print()
                print(
                    f"[{experiment_number}/"
                    f"{total_experiments}]"
                )

                results = run_experiment(
                    rounds=rounds,
                    physical_noise=physical_noise,
                    measurement_noise=measurement_noise,
                    seed=seed
                )

                for decoder, score in results.items():
                    all_results[decoder].append(
                        score
                    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(" FINAL DECODER SUMMARY")
    print("=" * 78)

    print()
    print(
        f"{'Decoder':<22}"
        f"{'Mean':>10}"
        f"{'Std':>10}"
        f"{'Wins':>10}"
    )

    print("-" * 78)

    means = {}
    stds = {}

    for decoder, scores in all_results.items():

        mean = float(
            np.mean(scores)
        )

        std = float(
            np.std(scores)
        )

        means[decoder] = mean
        stds[decoder] = std

    # Determine winner for every experiment
    experiment_count = len(
        all_results["traditional"]
    )

    wins = {
        decoder: 0
        for decoder in all_results
    }

    for index in range(experiment_count):

        experiment_scores = {
            decoder: all_results[decoder][index]
            for decoder in all_results
        }

        winner = max(
            experiment_scores,
            key=experiment_scores.get
        )

        wins[winner] += 1

    display_names = {
        "traditional": "Traditional",
        "logistic": "Logistic",
        "random_forest": "Random Forest",
        "mlp": "MLP",
        "gru": "Logical-target GRU",
    }

    for decoder in [
        "traditional",
        "logistic",
        "random_forest",
        "mlp",
        "gru",
    ]:

        print(
            f"{display_names[decoder]:<22}"
            f"{means[decoder]:>10.4f}"
            f"{stds[decoder]:>10.4f}"
            f"{wins[decoder]:>10}"
        )

    # --------------------------------------------------------
    # FINAL WINNER
    # --------------------------------------------------------

    winner = max(
        means,
        key=means.get
    )

    print()
    print("-" * 78)

    print(
        "BEST DECODER        : "
        f"{display_names[winner]}"
    )

    print(
        "BEST MEAN SCORE     : "
        f"{means[winner]:.4f}"
    )

    print(
        "WIN RATE            : "
        f"{wins[winner]}/{experiment_count}"
    )

    # --------------------------------------------------------
    # GAINS OVER TRADITIONAL
    # --------------------------------------------------------

    print()
    print("GAINS OVER TRADITIONAL")
    print("-" * 78)

    baseline = means["traditional"]

    for decoder in [
        "logistic",
        "random_forest",
        "mlp",
        "gru",
    ]:

        gain = (
            means[decoder]
            - baseline
        )

        print(
            f"{display_names[decoder]:<22}"
            f"+{gain:.4f}"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    print()
    print("=" * 78)

    if winner == "random_forest":

        print(
            "RECOMMENDATION : "
            "RANDOM FOREST"
        )

    elif winner == "mlp":

        print(
            "RECOMMENDATION : "
            "MLP"
        )

    elif winner == "gru":

        print(
            "RECOMMENDATION : "
            "LOGICAL-TARGET GRU"
        )

    else:

        print(
            "RECOMMENDATION : "
            f"{display_names[winner].upper()}"
        )

    print("=" * 78)

    print()
    print("FINAL DECODER SELECTION : COMPLETE")


if __name__ == "__main__":
    main()