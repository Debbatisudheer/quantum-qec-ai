from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)


ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10

TOTAL_SAMPLES = 30000
TEST_SAMPLES = 5000

SEED = 42

TRAINING_SIZES = [
    100,
    500,
    1000,
    2500,
    5000,
    10000,
    20000,
]


def encode_features(sample):
    """
    Convert observable information into
    the same 20-feature representation used
    by the existing time-varying AI pipeline.

    Features:

        observed syndrome history
        +
        detection-event history
    """

    features = []

    for syndrome in sample[
        "observed_syndrome_history"
    ]:

        features.extend(
            int(bit)
            for bit in syndrome
        )

    for event in sample[
        "detection_events"
    ]:

        features.extend(
            int(bit)
            for bit in event
        )

    return features


def encode_target(sample):
    """
    AI target:

        final accumulated physical
        X-error state.

    Example:

        [1, 0, 1]
    """

    return list(
        sample["final_error_state"]
    )


def pattern_accuracy(
    predictions,
    targets
):
    """
    Exact physical-error-pattern accuracy.

    A prediction is correct only when
    all three physical-error bits match.

    Example:

        predicted = [1,0,1]
        target    = [1,0,1]

        -> correct

        predicted = [1,0,0]
        target    = [1,0,1]

        -> incorrect
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
    Bit-level accuracy across q0, q1, q2.
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


def target_distribution(targets):
    """
    Return target-pattern distribution.
    """

    counter = Counter(
        tuple(target)
        for target in targets
    )

    return counter


def create_decoder(name):
    """
    Create a fresh decoder for each experiment.
    """

    if name == "Logistic Regression":

        return TimeVaryingLogisticDecoder(
            random_state=SEED
        )

    if name == "Random Forest":

        return TimeVaryingRandomForestDecoder(
            n_estimators=100,
            random_state=SEED
        )

    if name == "MLP":

        return TimeVaryingMLPDecoder(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=SEED
        )

    raise ValueError(
        f"Unknown decoder: {name}"
    )


def train_and_evaluate(
    decoder_name,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train one decoder and evaluate it
    on the fixed test set.
    """

    decoder = create_decoder(
        decoder_name
    )

    decoder.train(
        X_train,
        y_train
    )

    predictions = decoder.predict(
        X_test
    )

    exact_accuracy = pattern_accuracy(
        predictions,
        y_test
    )

    bits = bit_accuracy(
        predictions,
        y_test
    )

    return exact_accuracy, bits


def main():

    print()
    print("===================================")
    print(" AI LEARNING CURVE DIAGNOSTIC")
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
        f"Total generated samples   : "
        f"{TOTAL_SAMPLES}"
    )

    print(
        f"Fixed test samples        : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # ------------------------------------------------
    # Generate one complete dataset
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
    # Convert to ML representation
    # ------------------------------------------------

    X = [
        encode_features(sample)
        for sample in samples
    ]

    y = [
        encode_target(sample)
        for sample in samples
    ]

    # ------------------------------------------------
    # Verify feature representation
    # ------------------------------------------------

    feature_lengths = sorted(
        set(
            len(features)
            for features in X
        )
    )

    print()
    print("-----------------------------------")
    print("FEATURE REPRESENTATION")
    print("-----------------------------------")

    print(
        f"Feature lengths found     : "
        f"{feature_lengths}"
    )

    if feature_lengths == [20]:

        print(
            "Feature representation    : PASS"
        )

    else:

        print(
            "Feature representation    : FAIL"
        )

        raise RuntimeError(
            "Unexpected feature count"
        )

    # ------------------------------------------------
    # Fixed test set
    # ------------------------------------------------

    test_start = (
        TOTAL_SAMPLES
        -
        TEST_SAMPLES
    )

    X_test = X[test_start:]
    y_test = y[test_start:]

    X_pool = X[:test_start]
    y_pool = y[:test_start]

    print()
    print("-----------------------------------")
    print("DATASET SPLIT")
    print("-----------------------------------")

    print(
        f"Training pool samples     : "
        f"{len(X_pool)}"
    )

    print(
        f"Fixed test samples        : "
        f"{len(X_test)}"
    )

    print(
        "Test set remains identical "
        "for every training size."
    )

    # ------------------------------------------------
    # Test distribution
    # ------------------------------------------------

    print()
    print("-----------------------------------")
    print("FIXED TEST TARGET DISTRIBUTION")
    print("-----------------------------------")

    test_distribution = (
        target_distribution(y_test)
    )

    for target, count in (
        test_distribution.most_common()
    ):

        percentage = (
            count / len(y_test)
        ) * 100

        print(
            f"{target} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ------------------------------------------------
    # Decoder list
    # ------------------------------------------------

    decoder_names = [
        "Logistic Regression",
        "Random Forest",
        "MLP",
    ]

    results = {
        decoder_name: []
        for decoder_name in decoder_names
    }

    # ------------------------------------------------
    # Learning curve
    # ------------------------------------------------

    for training_size in TRAINING_SIZES:

        print()
        print("===================================")
        print(
            f"TRAINING SIZE: {training_size}"
        )
        print("===================================")

        X_train = X_pool[
            :training_size
        ]

        y_train = y_pool[
            :training_size
        ]

        for decoder_name in decoder_names:

            print()
            print(
                decoder_name
            )

            exact_accuracy, bits = (
                train_and_evaluate(
                    decoder_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test
                )
            )

            results[
                decoder_name
            ].append(
                {
                    "training_size": (
                        training_size
                    ),
                    "exact_accuracy": (
                        exact_accuracy
                    ),
                    "bit_accuracy": bits
                }
            )

            print(
                f"Exact pattern accuracy : "
                f"{exact_accuracy:.4f}"
            )

            print(
                f"Bit accuracy            : "
                f"{bits:.4f}"
            )

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    print()
    print("===================================")
    print(" LEARNING CURVE SUMMARY")
    print("===================================")

    for decoder_name in decoder_names:

        print()
        print(
            decoder_name
        )

        print(
            "-----------------------------------"
        )

        print(
            "Samples    Exact      Bit"
        )

        for result in results[
            decoder_name
        ]:

            print(
                f"{result['training_size']:<10}"
                f"{result['exact_accuracy']:.4f}    "
                f"{result['bit_accuracy']:.4f}"
            )

    # ------------------------------------------------
    # Analyze improvement
    # ------------------------------------------------

    print()
    print("===================================")
    print(" LEARNING CURVE INTERPRETATION")
    print("===================================")

    for decoder_name in decoder_names:

        decoder_results = results[
            decoder_name
        ]

        first = decoder_results[0]
        last = decoder_results[-1]

        exact_change = (
            last["exact_accuracy"]
            -
            first["exact_accuracy"]
        )

        bit_change = (
            last["bit_accuracy"]
            -
            first["bit_accuracy"]
        )

        print()
        print(
            decoder_name
        )

        print(
            f"Exact accuracy change : "
            f"{exact_change:+.4f}"
        )

        print(
            f"Bit accuracy change   : "
            f"{bit_change:+.4f}"
        )

        if exact_change >= 0.10:

            print(
                "Strong learning with "
                "additional training data."
            )

        elif exact_change >= 0.03:

            print(
                "Moderate learning with "
                "additional training data."
            )

        elif exact_change >= 0.01:

            print(
                "Small learning improvement "
                "with additional data."
            )

        else:

            print(
                "Little or no learning improvement "
                "from additional data."
            )

    # ------------------------------------------------
    # Research diagnosis
    # ------------------------------------------------

    print()
    print("-----------------------------------")
    print(" RESEARCH DIAGNOSIS")
    print("-----------------------------------")

    best_decoder = None
    best_accuracy = -1.0

    for decoder_name in decoder_names:

        final_accuracy = results[
            decoder_name
        ][-1]["exact_accuracy"]

        if final_accuracy > best_accuracy:

            best_accuracy = final_accuracy
            best_decoder = decoder_name

    print(
        f"Best final decoder       : "
        f"{best_decoder}"
    )

    print(
        f"Best final exact accuracy: "
        f"{best_accuracy:.4f}"
    )

    print()

    if best_accuracy >= 0.60:

        print(
            "At least one current model "
            "is learning a substantial amount "
            "of the observable information."
        )

    elif best_accuracy >= 0.50:

        print(
            "Current models learn only a "
            "limited amount of the available "
            "observable information."
        )

    else:

        print(
            "Current models are close to "
            "chance-level exact-pattern "
            "performance."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The information ceiling at the "
        "current noise configuration is "
        "approximately 0.6722 from the "
        "previous diagnostic."
    )

    print(
        "This value is used only as a "
        "reference for the current "
        "10,000-sample diagnostic."
    )

    print()
    print("===================================")
    print(
        "AI LEARNING CURVE DIAGNOSTIC : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()