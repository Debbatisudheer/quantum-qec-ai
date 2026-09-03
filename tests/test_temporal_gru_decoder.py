from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru import (
    TemporalGRUDecoder
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

HIDDEN_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


def encode_syndrome_sequence(sample):
    """
    Convert:

        ['01', '10', '11', '00', '01']

    into:

        [
            [0,1],
            [1,0],
            [1,1],
            [0,0],
            [0,1]
        ]

    Shape:

        rounds × 2
    """

    sequence = []

    for syndrome in sample[
        "observed_syndrome_history"
    ]:

        sequence.append(
            [
                int(syndrome[0]),
                int(syndrome[1])
            ]
        )

    return sequence


def encode_target(sample):

    return list(
        sample["final_error_state"]
    )


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

    return (
        correct
        / len(targets)
    )


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

    return (
        correct
        / total
    )


def target_distribution(targets):

    counter = Counter(
        tuple(target)
        for target in targets
    )

    for target, count in (
        counter.most_common()
    ):

        percentage = (
            count
            / len(targets)
            * 100
        )

        print(
            f"{target} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


def main():

    print()
    print("===================================")
    print(" TEMPORAL GRU QEC DECODER")
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
        f"GRU hidden size           : "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Training epochs           : "
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

    # -----------------------------------------
    # Generate common dataset
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
    # Fixed split
    # -----------------------------------------

    train_samples = samples[
        :TRAINING_SAMPLES
    ]

    test_samples = samples[
        TRAINING_SAMPLES:
    ]

    X_train = [
        encode_syndrome_sequence(
            sample
        )
        for sample in train_samples
    ]

    y_train = [
        encode_target(sample)
        for sample in train_samples
    ]

    X_test = [
        encode_syndrome_sequence(
            sample
        )
        for sample in test_samples
    ]

    y_test = [
        encode_target(sample)
        for sample in test_samples
    ]

    print()
    print("-----------------------------------")
    print("INPUT SHAPE")
    print("-----------------------------------")

    print(
        f"Rounds per sample         : "
        f"{len(X_train[0])}"
    )

    print(
        f"Features per round        : "
        f"{len(X_train[0][0])}"
    )

    print(
        "Sequence shape            : "
        f"[batch, {ROUNDS}, 2]"
    )

    print(
        "Target shape              : "
        "[batch, 3]"
    )

    # -----------------------------------------
    # Test target distribution
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("TEST TARGET DISTRIBUTION")
    print("-----------------------------------")

    target_distribution(
        y_test
    )

    # -----------------------------------------
    # Create decoder
    # -----------------------------------------

    decoder = TemporalGRUDecoder(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=SEED
    )

    # -----------------------------------------
    # Train
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("TRAINING")
    print("-----------------------------------")

    decoder.train(
        X_train,
        y_train
    )

    print(
        "GRU training completed."
    )

    # -----------------------------------------
    # Predict
    # -----------------------------------------

    predictions = decoder.predict(
        X_test
    )

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

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
        f"{bits:.4f}"
    )

    # -----------------------------------------
    # Show a few predictions
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("SAMPLE PREDICTIONS")
    print("-----------------------------------")

    for index in range(10):

        print(
            f"Sample {index + 1}: "
            f"actual={y_test[index]} "
            f"predicted={predictions[index]}"
        )

    # -----------------------------------------
    # Final comparison
    # -----------------------------------------

    print()
    print("===================================")
    print(" CURRENT DECODER COMPARISON")
    print("===================================")

    print()
    print(
        "Previous Random Forest:"
    )

    print(
        "Exact pattern ≈ 0.584"
    )

    print(
        "Bit accuracy   ≈ 0.728"
    )

    print()

    print(
        "Temporal GRU:"
    )

    print(
        f"Exact pattern = "
        f"{exact_accuracy:.4f}"
    )

    print(
        f"Bit accuracy   = "
        f"{bits:.4f}"
    )

    print()
    print("-----------------------------------")
    print("INTERPRETATION")
    print("-----------------------------------")

    if exact_accuracy > 0.5842:

        print(
            "GRU improves exact-pattern "
            "prediction over the previous "
            "Random Forest result."
        )

    else:

        print(
            "GRU does not yet improve exact-pattern "
            "prediction over the previous "
            "Random Forest result."
        )

    if bits > 0.7277:

        print(
            "GRU improves bit-level prediction."
        )

    else:

        print(
            "GRU does not yet improve bit-level "
            "prediction."
        )

    print()
    print("-----------------------------------")
    print("IMPORTANT")
    print("-----------------------------------")

    print(
        "This experiment uses syndrome history "
        "as an actual sequence."
    )

    print(
        "The GRU processes one QEC round at a "
        "time rather than treating all rounds "
        "as an unrelated flat feature vector."
    )

    print(
        "The target remains the final physical "
        "error state [q0, q1, q2]."
    )

    print()
    print("===================================")
    print(
        "TEMPORAL GRU EXPERIMENT : SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()