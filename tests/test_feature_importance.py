from collections import defaultdict

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
    + TEST_SAMPLES
)

SEED = 42

RF_ESTIMATORS = 100


def encode_features(sample):
    """
    Full temporal representation.

    5 rounds × 2 syndrome bits
    +
    5 rounds × 2 detection-event bits

    = 20 features.
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
    return list(
        sample["final_error_state"]
    )


def build_feature_names():
    """
    Feature order MUST exactly match
    encode_features().
    """

    names = []

    for round_number in range(1, ROUNDS + 1):

        names.append(
            f"Syndrome R{round_number} Bit 1"
        )

        names.append(
            f"Syndrome R{round_number} Bit 2"
        )

    for round_number in range(1, ROUNDS + 1):

        names.append(
            f"Detection R{round_number} Bit 1"
        )

        names.append(
            f"Detection R{round_number} Bit 2"
        )

    return names


def group_name(feature_name):
    """
    Group features by their source.
    """

    if feature_name.startswith(
        "Syndrome"
    ):
        return "Syndrome"

    return "Detection"


def round_number(feature_name):
    """
    Extract the round number from
    feature names such as:

        Syndrome R3 Bit 1
    """

    parts = feature_name.split()

    for part in parts:

        if part.startswith("R"):

            return int(
                part[1:]
            )

    raise ValueError(
        f"Could not determine round: "
        f"{feature_name}"
    )


def main():

    print()
    print("===================================")
    print(" RANDOM FOREST FEATURE IMPORTANCE")
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

    # -----------------------------------------
    # Generate one common dataset
    # -----------------------------------------

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

    # -----------------------------------------
    # Build training data
    # -----------------------------------------

    X_train = [
        encode_features(sample)
        for sample in samples[
            :TRAINING_SAMPLES
        ]
    ]

    y_train = [
        encode_target(sample)
        for sample in samples[
            :TRAINING_SAMPLES
        ]
    ]

    X_test = [
        encode_features(sample)
        for sample in samples[
            TRAINING_SAMPLES:
        ]
    ]

    y_test = [
        encode_target(sample)
        for sample in samples[
            TRAINING_SAMPLES:
        ]
    ]

    # -----------------------------------------
    # Train Random Forest
    # -----------------------------------------

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

    # -----------------------------------------
    # Accuracy
    # -----------------------------------------

    exact_correct = 0

    bit_correct = 0
    total_bits = 0

    for prediction, target in zip(
        predictions,
        y_test
    ):

        if list(prediction) == list(target):

            exact_correct += 1

        for predicted_bit, target_bit in zip(
            prediction,
            target
        ):

            if predicted_bit == target_bit:

                bit_correct += 1

            total_bits += 1

    exact_accuracy = (
        exact_correct
        / len(y_test)
    )

    bit_accuracy = (
        bit_correct
        / total_bits
    )

    print()
    print("-----------------------------------")
    print("MODEL PERFORMANCE")
    print("-----------------------------------")

    print(
        f"Exact pattern accuracy : "
        f"{exact_accuracy:.4f}"
    )

    print(
        f"Bit accuracy            : "
        f"{bit_accuracy:.4f}"
    )

    # -----------------------------------------
    # Feature names
    # -----------------------------------------

    feature_names = (
        build_feature_names()
    )

    importances = (
        decoder.model
        .estimators_
    )

    # MultiOutputClassifier contains
    # one Random Forest per target qubit.
    #
    # Average importance across q0, q1, q2.

    feature_importance_by_target = []

    for target_index, estimator in enumerate(
        importances
    ):

        target_importances = (
            estimator.feature_importances_
        )

        feature_importance_by_target.append(
            target_importances
        )

    # -----------------------------------------
    # Average importance
    # -----------------------------------------

    average_importances = []

    for feature_index in range(
        len(feature_names)
    ):

        values = [
            target_importances[
                feature_index
            ]
            for target_importances
            in feature_importance_by_target
        ]

        average_importance = (
            sum(values)
            / len(values)
        )

        average_importances.append(
            average_importance
        )

    ranked_features = sorted(
        zip(
            feature_names,
            average_importances
        ),
        key=lambda item: item[1],
        reverse=True
    )

    # -----------------------------------------
    # Print all features
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("ALL FEATURE IMPORTANCES")
    print("-----------------------------------")

    print()
    print(
        f"{'Rank':<6}"
        f"{'Feature':<30}"
        f"{'Importance':>12}"
    )

    print(
        "-" * 50
    )

    for rank, (
        feature_name,
        importance
    ) in enumerate(
        ranked_features,
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{feature_name:<30}"
            f"{importance:>12.6f}"
        )

    # -----------------------------------------
    # Aggregate by source
    # -----------------------------------------

    source_importance = defaultdict(float)

    for feature_name, importance in zip(
        feature_names,
        average_importances
    ):

        source = group_name(
            feature_name
        )

        source_importance[
            source
        ] += importance

    print()
    print("-----------------------------------")
    print("IMPORTANCE BY FEATURE SOURCE")
    print("-----------------------------------")

    for source, importance in sorted(
        source_importance.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        print(
            f"{source:<20}"
            f"{importance:.6f}"
        )

    # -----------------------------------------
    # Aggregate by round
    # -----------------------------------------

    round_importance = defaultdict(float)

    for feature_name, importance in zip(
        feature_names,
        average_importances
    ):

        current_round = round_number(
            feature_name
        )

        round_importance[
            current_round
        ] += importance

    print()
    print("-----------------------------------")
    print("IMPORTANCE BY ROUND")
    print("-----------------------------------")

    for current_round in sorted(
        round_importance
    ):

        print(
            f"Round {current_round:<12}"
            f"{round_importance[current_round]:.6f}"
        )

    # -----------------------------------------
    # Normalize source importance
    # -----------------------------------------

    total_importance = sum(
        average_importances
    )

    print()
    print("-----------------------------------")
    print("NORMALIZED FEATURE SOURCES")
    print("-----------------------------------")

    for source, importance in sorted(
        source_importance.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        percentage = (
            importance
            / total_importance
        ) * 100

        print(
            f"{source:<20}"
            f"{percentage:6.2f}%"
        )

    # -----------------------------------------
    # Per-target importance
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("FEATURE IMPORTANCE BY TARGET QUBIT")
    print("-----------------------------------")

    for target_index, target_importances in enumerate(
        feature_importance_by_target
    ):

        target_ranked = sorted(
            zip(
                feature_names,
                target_importances
            ),
            key=lambda item: item[1],
            reverse=True
        )

        print()
        print(
            f"Target q{target_index}"
        )

        print(
            "-" * 35
        )

        for rank, (
            feature_name,
            importance
        ) in enumerate(
            target_ranked[:5],
            start=1
        ):

            print(
                f"{rank}. "
                f"{feature_name:<25}"
                f"{importance:.6f}"
            )

    # -----------------------------------------
    # Research interpretation
    # -----------------------------------------

    top_feature = ranked_features[0]

    syndrome_percentage = (
        source_importance["Syndrome"]
        / total_importance
    ) * 100

    detection_percentage = (
        source_importance["Detection"]
        / total_importance
    ) * 100

    most_important_round = max(
        round_importance,
        key=round_importance.get
    )

    print()
    print("===================================")
    print(" RESEARCH INTERPRETATION")
    print("===================================")

    print()
    print(
        f"Most important feature : "
        f"{top_feature[0]}"
    )

    print(
        f"Importance              : "
        f"{top_feature[1]:.6f}"
    )

    print()
    print(
        f"Syndrome features      : "
        f"{syndrome_percentage:.2f}%"
    )

    print(
        f"Detection features     : "
        f"{detection_percentage:.2f}%"
    )

    print()
    print(
        f"Most important round   : "
        f"Round {most_important_round}"
    )

    print(
        f"Round importance       : "
        f"{round_importance[most_important_round]:.6f}"
    )

    print()
    print("-----------------------------------")
    print("IMPORTANT SCIENTIFIC NOTE")
    print("-----------------------------------")

    print(
        "Detection events are deterministically "
        "derived from the observed syndrome "
        "history."
    )

    print(
        "Therefore detection events do not add "
        "new raw information beyond the complete "
        "observed syndrome history."
    )

    print(
        "Their value is mainly a different "
        "representation that may help or hurt "
        "a particular decoder."
    )

    print()
    print("===================================")
    print(
        "FEATURE IMPORTANCE EXPERIMENT : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()