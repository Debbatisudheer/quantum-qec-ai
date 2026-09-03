from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.time_varying_ml import (
    TimeVaryingRandomForestDecoder
)


ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10

TRAINING_SAMPLES = 20000
TEST_SAMPLES = 5000

TOTAL_SAMPLES = (
    TRAINING_SAMPLES
    +
    TEST_SAMPLES
)

SEED = 42

RF_ESTIMATORS = 100


def encode_syndrome_history(sample):
    """
    Representation A:

        observed syndrome history only.

    For 5 rounds:

        5 × 2 = 10 features.
    """

    features = []

    for syndrome in sample[
        "observed_syndrome_history"
    ]:

        features.extend(
            int(bit)
            for bit in syndrome
        )

    return features


def encode_detection_events(sample):
    """
    Representation B:

        detection events only.

    For 5 rounds:

        5 × 2 = 10 features.
    """

    features = []

    for event in sample[
        "detection_events"
    ]:

        features.extend(
            int(bit)
            for bit in event
        )

    return features


def encode_final_syndrome(sample):
    """
    Representation C:

        final observed syndrome only.

    Exactly 2 features.
    """

    final_syndrome = sample[
        "final_observed_syndrome"
    ]

    return [
        int(bit)
        for bit in final_syndrome
    ]


def encode_combined_features(sample):
    """
    Representation D:

        observed syndrome history
        +
        detection events

    For 5 rounds:

        10 + 10 = 20 features.
    """

    return (
        encode_syndrome_history(sample)
        +
        encode_detection_events(sample)
    )


def encode_target(sample):
    """
    Target:

        final accumulated physical
        X-error state.
    """

    return list(
        sample["final_error_state"]
    )


def exact_pattern_accuracy(
    predictions,
    targets
):
    """
    Exact physical-error-pattern accuracy.

    All three qubits must match.
    """

    correct = 0

    for prediction, target in zip(
        predictions,
        targets
    ):

        if list(prediction) == list(target):

            correct += 1

    return correct / len(targets)


def bit_accuracy(
    predictions,
    targets
):
    """
    Bit-level accuracy across
    all three physical qubits.
    """

    correct = 0
    total = 0

    for prediction, target in zip(
        predictions,
        targets
    ):

        for predicted_bit, target_bit in zip(
            prediction,
            target
        ):

            if predicted_bit == target_bit:

                correct += 1

            total += 1

    return correct / total


def describe_target_distribution(
    targets
):
    """
    Print target distribution so that
    every representation is evaluated
    against the same target.
    """

    counter = Counter(
        tuple(target)
        for target in targets
    )

    for target, count in (
        counter.most_common()
    ):

        percentage = (
            count / len(targets)
        ) * 100

        print(
            f"{target} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


def train_and_evaluate(
    representation_name,
    encoder,
    samples,
    train_indices,
    test_indices
):
    """
    Train exactly the same Random Forest
    architecture on one feature representation.
    """

    X_train = [
        encoder(
            samples[index]
        )
        for index in train_indices
    ]

    y_train = [
        encode_target(
            samples[index]
        )
        for index in train_indices
    ]

    X_test = [
        encoder(
            samples[index]
        )
        for index in test_indices
    ]

    y_test = [
        encode_target(
            samples[index]
        )
        for index in test_indices
    ]

    feature_lengths = sorted(
        set(
            len(features)
            for features in X_train
        )
    )

    print()
    print(
        f"Representation : "
        f"{representation_name}"
    )

    print(
        f"Feature count  : "
        f"{feature_lengths}"
    )

    decoder = (
        TimeVaryingRandomForestDecoder(
            n_estimators=RF_ESTIMATORS,
            random_state=SEED
        )
    )

    decoder.train(
        X_train,
        y_train
    )

    predictions = decoder.predict(
        X_test
    )

    exact_accuracy = (
        exact_pattern_accuracy(
            predictions,
            y_test
        )
    )

    bits = bit_accuracy(
        predictions,
        y_test
    )

    print(
        f"Exact pattern accuracy : "
        f"{exact_accuracy:.4f}"
    )

    print(
        f"Bit accuracy            : "
        f"{bits:.4f}"
    )

    return {
        "name": representation_name,
        "feature_count": feature_lengths[0],
        "exact_accuracy": exact_accuracy,
        "bit_accuracy": bits
    }


def main():

    print()
    print("===================================")
    print(" FEATURE ABLATION EXPERIMENT")
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
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Random Forest estimators  : "
        f"{RF_ESTIMATORS}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # ------------------------------------------------
    # Generate one common dataset
    # ------------------------------------------------

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=SEED
        )
    )

    samples = generator.generate_dataset(
        TOTAL_SAMPLES
    )

    print()
    print(
        f"Samples generated         : "
        f"{len(samples)}"
    )

    # ------------------------------------------------
    # Fixed train/test split
    # ------------------------------------------------

    train_indices = list(
        range(TRAINING_SAMPLES)
    )

    test_indices = list(
        range(
            TRAINING_SAMPLES,
            TOTAL_SAMPLES
        )
    )

    print()
    print("-----------------------------------")
    print("DATASET SPLIT")
    print("-----------------------------------")

    print(
        f"Training samples          : "
        f"{len(train_indices)}"
    )

    print(
        f"Test samples              : "
        f"{len(test_indices)}"
    )

    # ------------------------------------------------
    # Verify target distribution
    # ------------------------------------------------

    test_targets = [
        encode_target(
            samples[index]
        )
        for index in test_indices
    ]

    print()
    print("-----------------------------------")
    print("FIXED TEST TARGET DISTRIBUTION")
    print("-----------------------------------")

    describe_target_distribution(
        test_targets
    )

    # ------------------------------------------------
    # Feature representations
    # ------------------------------------------------

    representations = [
        (
            "Final syndrome only",
            encode_final_syndrome
        ),
        (
            "Syndrome history only",
            encode_syndrome_history
        ),
        (
            "Detection events only",
            encode_detection_events
        ),
        (
            "Syndrome + detection events",
            encode_combined_features
        )
    ]

    results = []

    # ------------------------------------------------
    # Run ablation
    # ------------------------------------------------

    for representation_name, encoder in (
        representations
    ):

        result = train_and_evaluate(
            representation_name,
            encoder,
            samples,
            train_indices,
            test_indices
        )

        results.append(
            result
        )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print()
    print("===================================")
    print(" FEATURE ABLATION SUMMARY")
    print("===================================")

    print()
    print(
        "Representation                 "
        "Features   Exact      Bit"
    )

    print(
        "-----------------------------------"
    )

    for result in results:

        print(
            f"{result['name']:<32}"
            f"{result['feature_count']:<11}"
            f"{result['exact_accuracy']:.4f}    "
            f"{result['bit_accuracy']:.4f}"
        )

    # ------------------------------------------------
    # Rank representations
    # ------------------------------------------------

    ranked_results = sorted(
        results,
        key=lambda result:
        result["exact_accuracy"],
        reverse=True
    )

    print()
    print("===================================")
    print(" REPRESENTATION RANKING")
    print("===================================")

    for position, result in enumerate(
        ranked_results,
        start=1
    ):

        print(
            f"{position}. "
            f"{result['name']} "
            f"→ "
            f"{result['exact_accuracy']:.4f}"
        )

    # ------------------------------------------------
    # Compare components
    # ------------------------------------------------

    final_syndrome_result = next(
        result
        for result in results
        if result["name"]
        == "Final syndrome only"
    )

    syndrome_history_result = next(
        result
        for result in results
        if result["name"]
        == "Syndrome history only"
    )

    detection_result = next(
        result
        for result in results
        if result["name"]
        == "Detection events only"
    )

    combined_result = next(
        result
        for result in results
        if result["name"]
        == "Syndrome + detection events"
    )

    print()
    print("===================================")
    print(" COMPONENT ANALYSIS")
    print("===================================")

    print()
    print(
        "Final syndrome → syndrome history"
    )

    print(
        f"{final_syndrome_result['exact_accuracy']:.4f}"
        f" → "
        f"{syndrome_history_result['exact_accuracy']:.4f}"
    )

    print()
    print(
        "Final syndrome → combined temporal"
    )

    print(
        f"{final_syndrome_result['exact_accuracy']:.4f}"
        f" → "
        f"{combined_result['exact_accuracy']:.4f}"
    )

    print()
    print(
        "Syndrome history → combined temporal"
    )

    print(
        f"{syndrome_history_result['exact_accuracy']:.4f}"
        f" → "
        f"{combined_result['exact_accuracy']:.4f}"
    )

    print()
    print(
        "Detection events alone:"
    )

    print(
        f"{detection_result['exact_accuracy']:.4f}"
    )

    # ------------------------------------------------
    # Research interpretation
    # ------------------------------------------------

    print()
    print("===================================")
    print(" RESEARCH INTERPRETATION")
    print("===================================")

    best_result = ranked_results[0]

    print()
    print(
        f"Best representation : "
        f"{best_result['name']}"
    )

    print(
        f"Best exact accuracy : "
        f"{best_result['exact_accuracy']:.4f}"
    )

    print()

    if (
        combined_result["exact_accuracy"]
        >
        syndrome_history_result["exact_accuracy"]
    ):

        improvement = (
            combined_result["exact_accuracy"]
            -
            syndrome_history_result["exact_accuracy"]
        )

        print(
            "Adding detection events improves "
            "the decoder."
        )

        print(
            f"Improvement: "
            f"{improvement:+.4f}"
        )

    else:

        improvement = (
            combined_result["exact_accuracy"]
            -
            syndrome_history_result["exact_accuracy"]
        )

        print(
            "Adding detection events does not "
            "improve the Random Forest in this "
            "experiment."
        )

        print(
            f"Change: "
            f"{improvement:+.4f}"
        )

    print()

    if (
        syndrome_history_result["exact_accuracy"]
        >
        final_syndrome_result["exact_accuracy"]
    ):

        print(
            "Temporal syndrome history contains "
            "more useful information than the "
            "final syndrome alone."
        )

    else:

        print(
            "Temporal syndrome history does not "
            "improve over the final syndrome "
            "for this decoder/configuration."
        )

    print()

    if (
        detection_result["exact_accuracy"]
        >
        final_syndrome_result["exact_accuracy"]
    ):

        print(
            "Detection events alone contain "
            "useful decoding information."
        )

    else:

        print(
            "Detection events alone are weaker "
            "than the final syndrome."
        )

    print()
    print("-----------------------------------")
    print(" IMPORTANT")
    print("-----------------------------------")

    print(
        "All representations use the same "
        "dataset, same target, same training "
        "size, same test set, and same "
        "Random Forest configuration."
    )

    print(
        "Therefore differences are attributable "
        "primarily to the feature representation."
    )

    print()
    print("===================================")
    print(
        "FEATURE ABLATION EXPERIMENT : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()