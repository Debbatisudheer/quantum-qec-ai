from sklearn.model_selection import train_test_split

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)

from experiments.quantum_ai_decoder_integration import (
    QuantumAIDecoderIntegration
)


ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30
]

TRAINING_SAMPLES = 3000
TEST_SAMPLES = 100

RANDOM_SEED = 42


def encode_features(
    syndrome_history,
    detection_events
):
    features = []

    for syndrome in syndrome_history:
        features.extend(
            int(bit)
            for bit in syndrome
        )

    for event in detection_events:
        features.extend(
            int(bit)
            for bit in event
        )

    return features


def create_training_data(samples):
    X = []
    y = []

    for sample in samples:

        observed_syndrome_history = sample[
            "observed_syndrome_history"
        ]

        detection_events = sample[
            "detection_events"
        ]

        features = encode_features(
            observed_syndrome_history,
            detection_events
        )

        target = sample[
            "final_error_state"
        ]

        X.append(features)
        y.append(target)

    return X, y


def create_training_dataset(
    measurement_noise_probability
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=(
            PHYSICAL_ERROR_PROBABILITY
        ),
        measurement_noise_probability=(
            measurement_noise_probability
        ),
        seed=RANDOM_SEED
    )

    samples = generator.generate_dataset(
        TRAINING_SAMPLES
    )

    X, y = create_training_data(
        samples
    )

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=RANDOM_SEED
    )

    return X_train, y_train


def create_test_dataset(
    measurement_noise_probability
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=(
            PHYSICAL_ERROR_PROBABILITY
        ),
        measurement_noise_probability=(
            measurement_noise_probability
        ),
        seed=123
    )

    return generator.generate_dataset(
        TEST_SAMPLES
    )


def run_decoder(
    decoder,
    X_train,
    y_train,
    test_samples,
    measurement_noise_probability
):

    decoder.train(
        X_train,
        y_train
    )

    integration = QuantumAIDecoderIntegration(
        rounds=ROUNDS,
        shots=1,
        measurement_noise_probability=(
            measurement_noise_probability
        ),
        random_seed=RANDOM_SEED
    )

    results = integration.run_experiment(
        samples=test_samples,
        decoder=decoder
    )

    metrics = integration.calculate_metrics(
        results
    )

    return metrics


def print_result(
    noise,
    logistic_metrics,
    rf_metrics,
    mlp_metrics
):

    print()

    print(
        f"{noise * 100:5.0f}%"
        f"       "
        f"{logistic_metrics['physical_success_rate']:.4f}"
        f"          "
        f"{logistic_metrics['logical_success_rate']:.4f}"
        f"          "
        f"{rf_metrics['physical_success_rate']:.4f}"
        f"          "
        f"{rf_metrics['logical_success_rate']:.4f}"
        f"          "
        f"{mlp_metrics['physical_success_rate']:.4f}"
        f"          "
        f"{mlp_metrics['logical_success_rate']:.4f}"
    )


def main():

    print()

    print("==============================================")
    print(" QUANTUM + AI QEC")
    print(" MEASUREMENT NOISE SWEEP")
    print("==============================================")

    print()

    print(
        f"Rounds                     : {ROUNDS}"
    )

    print(
        f"Physical error probability : "
        f"{PHYSICAL_ERROR_PROBABILITY}"
    )

    print(
        f"Training samples           : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples               : "
        f"{TEST_SAMPLES}"
    )

    print()

    print(
        "Measurement noise levels:"
    )

    for noise in MEASUREMENT_NOISE_LEVELS:
        print(
            f"  {noise * 100:.0f}%"
        )

    all_results = []

    for noise in MEASUREMENT_NOISE_LEVELS:

        print()
        print("----------------------------------------------")

        print(
            f" Measurement Noise = "
            f"{noise * 100:.0f}%"
        )

        print("----------------------------------------------")

        print()
        print(
            "Generating training dataset..."
        )

        X_train, y_train = create_training_dataset(
            noise
        )

        print(
            f"Training samples : "
            f"{len(X_train)}"
        )

        print(
            f"Input features   : "
            f"{len(X_train[0])}"
        )

        print(
            f"Target size      : "
            f"{len(y_train[0])}"
        )

        print()

        print(
            "Generating independent test dataset..."
        )

        test_samples = create_test_dataset(
            noise
        )

        print(
            f"Test samples     : "
            f"{len(test_samples)}"
        )

        print()

        print(
            "Running Logistic Regression..."
        )

        logistic_metrics = run_decoder(
            TimeVaryingLogisticDecoder(
                random_state=RANDOM_SEED
            ),
            X_train,
            y_train,
            test_samples,
            noise
        )

        print(
            "Running Random Forest..."
        )

        rf_metrics = run_decoder(
            TimeVaryingRandomForestDecoder(
                n_estimators=100,
                random_state=RANDOM_SEED
            ),
            X_train,
            y_train,
            test_samples,
            noise
        )

        print(
            "Running MLP..."
        )

        mlp_metrics = run_decoder(
            TimeVaryingMLPDecoder(
                hidden_layer_sizes=(32, 16),
                max_iter=1000,
                random_state=RANDOM_SEED
            ),
            X_train,
            y_train,
            test_samples,
            noise
        )

        all_results.append(
            {
                "noise": noise,
                "logistic": logistic_metrics,
                "random_forest": rf_metrics,
                "mlp": mlp_metrics
            }
        )

        print()

        print(
            "Results:"
        )

        print(
            f"Logistic physical success : "
            f"{logistic_metrics['physical_success_rate']:.4f}"
        )

        print(
            f"Logistic logical success  : "
            f"{logistic_metrics['logical_success_rate']:.4f}"
        )

        print(
            f"Random Forest physical    : "
            f"{rf_metrics['physical_success_rate']:.4f}"
        )

        print(
            f"Random Forest logical     : "
            f"{rf_metrics['logical_success_rate']:.4f}"
        )

        print(
            f"MLP physical success      : "
            f"{mlp_metrics['physical_success_rate']:.4f}"
        )

        print(
            f"MLP logical success       : "
            f"{mlp_metrics['logical_success_rate']:.4f}"
        )

    print()

    print("==============================================")
    print(" FINAL MEASUREMENT NOISE SWEEP")
    print("==============================================")

    print()

    print(
        "Noise   Logistic         RF               MLP"
    )

    print(
        "        Physical Logical Physical Logical "
        "Physical Logical"
    )

    print(
        "----------------------------------------------"
    )

    for result in all_results:

        noise = result["noise"]

        logistic_metrics = result[
            "logistic"
        ]

        rf_metrics = result[
            "random_forest"
        ]

        mlp_metrics = result[
            "mlp"
        ]

        print_result(
            noise,
            logistic_metrics,
            rf_metrics,
            mlp_metrics
        )

    print()

    print("==============================================")
    print(" EXPERIMENT COMPLETE")
    print("==============================================")

    print()

    print(
        "Quantum syndrome"
    )

    print(
        "      ↓"
    )

    print(
        "Measurement noise sweep"
    )

    print(
        "      ↓"
    )

    print(
        "AI decoding"
    )

    print(
        "      ↓"
    )

    print(
        "Quantum correction"
    )

    print(
        "      ↓"
    )

    print(
        "Logical recovery"
    )

    print(
        "      ↓"
    )

    print(
        "Measurement noise performance curve"
    )

    print()

    print(
        "MEASUREMENT NOISE SWEEP : SUCCESS"
    )


if __name__ == "__main__":
    main()