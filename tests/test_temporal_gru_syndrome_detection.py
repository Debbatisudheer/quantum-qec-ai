from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier
)


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10

TOTAL_SAMPLES = 25000
TEST_SAMPLES = 5000

TRAINING_SIZE = 20000

SEED = 42

HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.003


# ============================================================
# FEATURE ENCODING
# ============================================================

def calculate_detection_events(
    observed_syndrome_history
):

    detection_events = []

    previous = "00"

    for syndrome in observed_syndrome_history:

        event = (
            str(
                int(previous[0])
                ^ int(syndrome[0])
            )
            +
            str(
                int(previous[1])
                ^ int(syndrome[1])
            )
        )

        detection_events.append(event)

        previous = syndrome

    return detection_events


def encode_sequence(sample):

    observed_history = (
        sample[
            "observed_syndrome_history"
        ]
    )

    detection_events = (
        calculate_detection_events(
            observed_history
        )
    )

    sequence = []

    for syndrome, detection in zip(
        observed_history,
        detection_events
    ):

        sequence.append(
            [
                int(syndrome[0]),
                int(syndrome[1]),
                int(detection[0]),
                int(detection[1])
            ]
        )

    return sequence


# ============================================================
# TARGET
# ============================================================

def encode_target(sample):

    return list(
        sample["final_error_state"]
    )


# ============================================================
# METRICS
# ============================================================

def exact_pattern_accuracy(
    predictions,
    targets
):

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print(" TEMPORAL GRU: SYNDROME + DETECTION")
    print("==============================================")

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
        f"{TRAINING_SIZE}"
    )

    print(
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"GRU hidden size           : "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Epochs                    : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate             : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # ========================================================
    # GENERATE FIXED DATASET
    # ========================================================

    print()
    print("----------------------------------------------")
    print("GENERATING FIXED DATASET")
    print("----------------------------------------------")

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

    print(
        f"Generated samples        : "
        f"{len(samples)}"
    )

    # ========================================================
    # FIXED TRAIN / TEST SPLIT
    # ========================================================

    train_samples = samples[
        :TRAINING_SIZE
    ]

    test_samples = samples[
        TRAINING_SIZE:
    ]

    X_train = [
        encode_sequence(sample)
        for sample in train_samples
    ]

    y_train = [
        encode_target(sample)
        for sample in train_samples
    ]

    X_test = [
        encode_sequence(sample)
        for sample in test_samples
    ]

    y_test = [
        encode_target(sample)
        for sample in test_samples
    ]

    # ========================================================
    # SHOW INPUT EXAMPLE
    # ========================================================

    print()
    print("----------------------------------------------")
    print("INPUT REPRESENTATION")
    print("----------------------------------------------")

    print(
        "Sequence shape            : "
        f"[batch, {ROUNDS}, 4]"
    )

    print()
    print(
        "Each round:"
    )

    print(
        "[syndrome_bit_1, "
        "syndrome_bit_2, "
        "detection_bit_1, "
        "detection_bit_2]"
    )

    print()
    print(
        "Example sequence:"
    )

    print(
        X_train[0]
    )

    print()
    print(
        "Target:"
    )

    print(
        y_train[0]
    )

    # ========================================================
    # VERIFY DETECTION EVENTS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("DETECTION EVENT VERIFICATION")
    print("----------------------------------------------")

    sample = train_samples[0]

    observed = (
        sample[
            "observed_syndrome_history"
        ]
    )

    detection = (
        calculate_detection_events(
            observed
        )
    )

    print(
        f"Observed syndrome history : "
        f"{observed}"
    )

    print(
        f"Detection events           : "
        f"{detection}"
    )

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TEST TARGET DISTRIBUTION")
    print("----------------------------------------------")

    counter = Counter(
        tuple(target)
        for target in y_test
    )

    for target, count in (
        counter.most_common()
    ):

        percentage = (
            count
            / len(y_test)
            * 100
        )

        print(
            f"{target} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # CREATE GRU
    # ========================================================

    decoder = TemporalGRUClassifier(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=SEED
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRAINING")
    print("----------------------------------------------")

    decoder.train(
        X_train,
        y_train,
        verbose=True
    )

    # ========================================================
    # TRAIN PERFORMANCE
    # ========================================================

    train_predictions = (
        decoder.predict(
            X_train
        )
    )

    train_exact = (
        exact_pattern_accuracy(
            train_predictions,
            y_train
        )
    )

    train_bit = (
        bit_accuracy(
            train_predictions,
            y_train
        )
    )

    # ========================================================
    # TEST PERFORMANCE
    # ========================================================

    test_predictions = (
        decoder.predict(
            X_test
        )
    )

    test_exact = (
        exact_pattern_accuracy(
            test_predictions,
            y_test
        )
    )

    test_bit = (
        bit_accuracy(
            test_predictions,
            y_test
        )
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    prediction_counter = Counter(
        tuple(prediction)
        for prediction in test_predictions
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("==============================================")
    print(" MODEL PERFORMANCE")
    print("==============================================")

    print()
    print(
        f"Train exact              : "
        f"{train_exact:.4f}"
    )

    print(
        f"Train bit                : "
        f"{train_bit:.4f}"
    )

    print(
        f"Test exact               : "
        f"{test_exact:.4f}"
    )

    print(
        f"Test bit                 : "
        f"{test_bit:.4f}"
    )

    print(
        f"Predicted classes        : "
        f"{len(prediction_counter)}"
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TEST PREDICTION DISTRIBUTION")
    print("----------------------------------------------")

    for prediction, count in (
        prediction_counter.most_common()
    ):

        percentage = (
            count
            / len(test_predictions)
            * 100
        )

        print(
            f"{prediction} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("SAMPLE PREDICTIONS")
    print("----------------------------------------------")

    for index in range(20):

        print(
            f"Sample {index + 1:2d}: "
            f"actual={y_test[index]} "
            f"predicted={test_predictions[index]}"
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("==============================================")
    print(" REPRESENTATION COMPARISON")
    print("==============================================")

    print()
    print(
        "Previous GRU:"
    )

    print(
        "Input  = [5 × 2]"
    )

    print(
        "        syndrome history only"
    )

    print(
        "Exact  = 0.6156"
    )

    print(
        "Bit    = 0.7260"
    )

    print()
    print(
        "New GRU:"
    )

    print(
        "Input  = [5 × 4]"
    )

    print(
        "        syndrome + detection"
    )

    print(
        f"Exact  = {test_exact:.4f}"
    )

    print(
        f"Bit    = {test_bit:.4f}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    exact_improvement = (
        test_exact - 0.6156
    )

    bit_improvement = (
        test_bit - 0.7260
    )

    print()
    print("----------------------------------------------")
    print("IMPROVEMENT")
    print("----------------------------------------------")

    print(
        f"Exact improvement        : "
        f"{exact_improvement:+.4f}"
    )

    print(
        f"Bit improvement          : "
        f"{bit_improvement:+.4f}"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("DIAGNOSIS")
    print("----------------------------------------------")

    if exact_improvement >= 0.02:

        print()
        print(
            "SIGNIFICANT IMPROVEMENT"
        )

        print(
            "The explicit detection-event "
            "representation helps the GRU "
            "learn the final error pattern."
        )

    elif exact_improvement > 0.005:

        print()
        print(
            "SMALL IMPROVEMENT"
        )

        print(
            "Detection events provide some "
            "learning benefit, but they are "
            "not the main bottleneck."
        )

    elif exact_improvement >= -0.005:

        print()
        print(
            "NO MEANINGFUL IMPROVEMENT"
        )

        print(
            "The GRU already extracts most of "
            "the useful information from the "
            "syndrome history."
        )

        print(
            "Detection events do not add "
            "meaningful predictive power."
        )

    else:

        print()
        print(
            "PERFORMANCE DECREASE"
        )

        print(
            "The additional representation "
            "appears to make learning harder."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("==============================================")
    print(
        " SYNDROME + DETECTION GRU : COMPLETE"
    )
    print("==============================================")


if __name__ == "__main__":
    main()