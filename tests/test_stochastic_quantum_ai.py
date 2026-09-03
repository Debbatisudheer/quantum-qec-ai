from sklearn.model_selection import train_test_split

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)

from experiments.stochastic_quantum_ai_experiment import (
    StochasticQuantumAIExperiment
)


ROUNDS = 5
TRAINING_SAMPLES = 5000
TEST_TRIALS = 200

PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10


def encode_sample_features(sample):
    """
    Convert one time-varying QEC sample into
    the feature vector used by the AI decoder.

    Features:

        observed_syndrome_history
                    +
        detection_events

    For 5 rounds:

        5 syndrome measurements × 2 bits = 10
        5 detection events × 2 bits       = 10

        Total = 20 features
    """

    features = []

    observed_syndrome_history = sample[
        "observed_syndrome_history"
    ]

    detection_events = sample[
        "detection_events"
    ]

    # Add observed syndrome history.
    for syndrome in observed_syndrome_history:

        for bit in syndrome:

            features.append(
                int(bit)
            )

    # Add detection events.
    for event in detection_events:

        for bit in event:

            features.append(
                int(bit)
            )

    return features


def encode_sample_target(sample):
    """
    Ground-truth target.

    The AI predicts the final accumulated
    physical X-error state:

        [q0, q1, q2]
    """

    return list(
        sample["final_error_state"]
    )


def create_training_arrays():
    """
    Generate training data and convert it into
    X and y arrays.

    The AI receives only observable information.

    Ground truth is used only as the target.
    """

    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=
            PHYSICAL_ERROR_PROBABILITY,
        measurement_noise_probability=
            MEASUREMENT_NOISE_PROBABILITY,
        seed=42
    )

    samples = generator.generate_dataset(
        num_samples=TRAINING_SAMPLES
    )

    X = [
        encode_sample_features(sample)
        for sample in samples
    ]

    y = [
        encode_sample_target(sample)
        for sample in samples
    ]

    return X, y


def create_train_validation_test_split():
    """
    Create:

        80% training
        10% validation
        10% test

    The validation/test sets are independent from
    the quantum trials used later.

    Stratification is based on the complete
    3-bit error pattern.
    """

    X, y = create_training_arrays()

    # Convert each 3-bit target into a pattern label.
    labels = [
        "".join(
            str(bit)
            for bit in target
        )
        for target in y
    ]

    # First split:
    #
    # 80% training
    # 20% temporary
    #
    X_train, X_temp, y_train, y_temp, labels_train, labels_temp = (
        train_test_split(
            X,
            y,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels
        )
    )

    # Second split:
    #
    # temporary 20%
    #
    # -> 10% validation
    # -> 10% test
    #
    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=labels_temp
        )
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


def train_decoder(decoder_class):
    """
    Train one AI decoder.
    """

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = create_train_validation_test_split()

    decoder = decoder_class()

    decoder.train(
        X_train,
        y_train
    )

    return decoder


def run_decoder_experiment(
    decoder_name,
    decoder_class
):
    print()
    print("-----------------------------------")
    print(
        f"Decoder: {decoder_name}"
    )
    print("-----------------------------------")

    decoder = train_decoder(
        decoder_class
    )

    experiment = (
        StochasticQuantumAIExperiment(
            rounds=ROUNDS,
            physical_error_probability=
                PHYSICAL_ERROR_PROBABILITY,
            measurement_noise_probability=
                MEASUREMENT_NOISE_PROBABILITY,
            seed=100
        )
    )

    metrics = (
        experiment.run_experiment(
            decoder=decoder,
            num_trials=TEST_TRIALS
        )
    )

    print()

    print(
        f"Physical success : "
        f"{metrics['physical_success']:.4f}"
    )

    print(
        f"Logical success  : "
        f"{metrics['logical_success']:.4f}"
    )

    print(
        f"Logical error    : "
        f"{metrics['logical_error_rate']:.4f}"
    )

    return metrics


def main():

    print()

    print("===================================")
    print(" STOCHASTIC QUANTUM + AI TEST")
    print("===================================")

    print()

    print(
        f"Rounds                    : {ROUNDS}"
    )

    print(
        f"Physical X noise          : "
        f"{PHYSICAL_ERROR_PROBABILITY}"
    )

    print(
        f"Measurement noise         : "
        f"{MEASUREMENT_NOISE_PROBABILITY}"
    )

    print(
        f"Training samples          : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Quantum test trials       : "
        f"{TEST_TRIALS}"
    )

    print()

    # Verify the feature representation
    # before training.
    X, y = create_training_arrays()

    print(
        f"Feature count             : "
        f"{len(X[0])}"
    )

    print(
        f"Target size               : "
        f"{len(y[0])}"
    )

    print()

    if len(X[0]) != 20:

        print(
            "FEATURE TEST : FAIL"
        )

        return

    if len(y[0]) != 3:

        print(
            "TARGET TEST : FAIL"
        )

        return

    print(
        "Feature representation    : PASS"
    )

    print(
        "Target representation     : PASS"
    )

    print()

    decoders = [
        (
            "Logistic Regression",
            TimeVaryingLogisticDecoder
        ),
        (
            "Random Forest",
            TimeVaryingRandomForestDecoder
        ),
        (
            "MLP",
            TimeVaryingMLPDecoder
        )
    ]

    results = {}

    for decoder_name, decoder_class in decoders:

        results[decoder_name] = (
            run_decoder_experiment(
                decoder_name,
                decoder_class
            )
        )

    print()

    print("===================================")
    print(" STOCHASTIC QUANTUM + AI SUMMARY")
    print("===================================")

    print()

    for decoder_name, metrics in results.items():

        print(
            f"{decoder_name:<22}"
            f"Physical: "
            f"{metrics['physical_success']:.4f}    "
            f"Logical: "
            f"{metrics['logical_success']:.4f}    "
            f"Logical Error: "
            f"{metrics['logical_error_rate']:.4f}"
        )

    print()

    print("===================================")
    print(
        "STOCHASTIC QUANTUM + AI TEST : SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()