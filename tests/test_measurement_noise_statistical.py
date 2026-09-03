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

TRAINING_SAMPLES = 10000
TEST_SAMPLES = 1000

SEEDS = [
    42,
    123,
    456
]


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
    measurement_noise_probability,
    seed
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=(
            PHYSICAL_ERROR_PROBABILITY
        ),
        measurement_noise_probability=(
            measurement_noise_probability
        ),
        seed=seed
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
        random_state=seed
    )

    return X_train, y_train


def create_test_dataset(
    measurement_noise_probability,
    seed
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=(
            PHYSICAL_ERROR_PROBABILITY
        ),
        measurement_noise_probability=(
            measurement_noise_probability
        ),
        seed=seed
    )

    return generator.generate_dataset(
        TEST_SAMPLES
    )


def run_decoder(
    decoder,
    X_train,
    y_train,
    test_samples,
    measurement_noise_probability,
    seed
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
        random_seed=seed
    )

    results = integration.run_experiment(
        samples=test_samples,
        decoder=decoder
    )

    metrics = integration.calculate_metrics(
        results
    )

    return metrics


def create_decoders(seed):

    return {
        "Logistic": TimeVaryingLogisticDecoder(
            random_state=seed
        ),

        "Random Forest": TimeVaryingRandomForestDecoder(
            n_estimators=100,
            random_state=seed
        ),

        "MLP": TimeVaryingMLPDecoder(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=seed
        )
    }


def average(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def print_seed_results(
    seed,
    noise,
    results
):

    print()

    print(
        f"Seed {seed} | "
        f"Measurement noise = "
        f"{noise * 100:.0f}%"
    )

    for decoder_name, metrics in results.items():

        print(
            f"{decoder_name:<15} "
            f"Physical = "
            f"{metrics['physical_success_rate']:.4f}   "
            f"Logical = "
            f"{metrics['logical_success_rate']:.4f}   "
            f"Logical Error = "
            f"{metrics['logical_error_rate']:.4f}"
        )


def main():

    print()

    print("======================================================")
    print(" QUANTUM + AI QEC")
    print(" STATISTICAL MEASUREMENT NOISE EXPERIMENT")
    print("======================================================")

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

    print(
        f"Random seeds               : "
        f"{SEEDS}"
    )

    print()

    all_results = {}

    for noise in MEASUREMENT_NOISE_LEVELS:

        print()
        print("======================================================")
        print(
            f" MEASUREMENT NOISE = "
            f"{noise * 100:.0f}%"
        )
        print("======================================================")

        all_results[noise] = {
            "Logistic": [],
            "Random Forest": [],
            "MLP": []
        }

        for seed in SEEDS:

            print()
            print(
                f"Preparing seed {seed}..."
            )

            X_train, y_train = create_training_dataset(
                noise,
                seed
            )

            test_samples = create_test_dataset(
                noise,
                seed + 1000
            )

            print(
                f"Training samples : "
                f"{len(X_train)}"
            )

            print(
                f"Test samples     : "
                f"{len(test_samples)}"
            )

            print(
                f"Features         : "
                f"{len(X_train[0])}"
            )

            decoders = create_decoders(
                seed
            )

            seed_results = {}

            for decoder_name, decoder in decoders.items():

                print(
                    f"Running {decoder_name}..."
                )

                metrics = run_decoder(
                    decoder,
                    X_train,
                    y_train,
                    test_samples,
                    noise,
                    seed
                )

                seed_results[
                    decoder_name
                ] = metrics

                all_results[noise][
                    decoder_name
                ].append(
                    metrics
                )

            print_seed_results(
                seed,
                noise,
                seed_results
            )

    print()

    print("======================================================")
    print(" AVERAGE RESULTS ACROSS RANDOM SEEDS")
    print("======================================================")

    print()

    print(
        "Noise   Decoder          "
        "Physical    Logical     Logical Error"
    )

    print(
        "------------------------------------------------------"
    )

    for noise in MEASUREMENT_NOISE_LEVELS:

        for decoder_name in [
            "Logistic",
            "Random Forest",
            "MLP"
        ]:

            metrics_list = all_results[
                noise
            ][
                decoder_name
            ]

            physical_values = [
                metrics[
                    "physical_success_rate"
                ]
                for metrics in metrics_list
            ]

            logical_values = [
                metrics[
                    "logical_success_rate"
                ]
                for metrics in metrics_list
            ]

            logical_error_values = [
                metrics[
                    "logical_error_rate"
                ]
                for metrics in metrics_list
            ]

            print(
                f"{noise * 100:4.0f}%   "
                f"{decoder_name:<15} "
                f"{average(physical_values):.4f}      "
                f"{average(logical_values):.4f}      "
                f"{average(logical_error_values):.4f}"
            )

        print()

    print("======================================================")
    print(" DECODER COMPARISON")
    print("======================================================")

    print()

    for noise in MEASUREMENT_NOISE_LEVELS:

        print(
            f"Measurement noise = "
            f"{noise * 100:.0f}%"
        )

        logical_scores = {}

        for decoder_name in [
            "Logistic",
            "Random Forest",
            "MLP"
        ]:

            metrics_list = all_results[
                noise
            ][
                decoder_name
            ]

            logical_values = [
                metrics[
                    "logical_success_rate"
                ]
                for metrics in metrics_list
            ]

            logical_scores[
                decoder_name
            ] = average(
                logical_values
            )

        ranked = sorted(
            logical_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for position, (
            decoder_name,
            score
        ) in enumerate(
            ranked,
            start=1
        ):

            print(
                f"{position}. "
                f"{decoder_name:<15} "
                f"Logical success = "
                f"{score:.4f}"
            )

        print()

    print("======================================================")
    print(" STATISTICAL EXPERIMENT COMPLETE")
    print("======================================================")

    print()

    print(
        "Quantum syndrome"
    )

    print(
        "      ↓"
    )

    print(
        "Measurement noise"
    )

    print(
        "      ↓"
    )

    print(
        "Multiple random seeds"
    )

    print(
        "      ↓"
    )

    print(
        "Large independent test sets"
    )

    print(
        "      ↓"
    )

    print(
        "Logistic / Random Forest / MLP"
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
        "Statistical comparison"
    )

    print()

    print(
        "STATISTICAL MEASUREMENT NOISE TEST : SUCCESS"
    )


if __name__ == "__main__":
    main()