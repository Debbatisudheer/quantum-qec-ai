from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier
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
EPOCHS = 50
LEARNING_RATE = 0.001


def encode_syndrome_sequence(sample):

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


def prediction_distribution(
    predictions
):

    counter = Counter(
        tuple(prediction)
        for prediction in predictions
    )

    for prediction, count in (
        counter.most_common()
    ):

        percentage = (
            count
            / len(predictions)
            * 100
        )

        print(
            f"{prediction} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


def main():

    print()
    print("===================================")
    print(" 8-CLASS TEMPORAL GRU QEC DECODER")
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
    # Generate dataset
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
    # Fixed train/test split
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

    # -----------------------------------------
    # Dataset information
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("INPUT REPRESENTATION")
    print("-----------------------------------")

    print(
        "Sequence shape            : "
        f"[batch, {ROUNDS}, 2]"
    )

    print(
        "Output classes            : 8"
    )

    print(
        "Class mapping:"
    )

    for class_index, pattern in enumerate(
        [
            "000",
            "001",
            "010",
            "011",
            "100",
            "101",
            "110",
            "111"
        ]
    ):

        print(
            f"Class {class_index} "
            f"→ {pattern}"
        )

    # -----------------------------------------
    # Target distribution
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

    decoder = TemporalGRUClassifier(
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
        y_train,
        verbose=True
    )

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    predictions = decoder.predict(
        X_test
    )

    # -----------------------------------------
    # Evaluation
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
    # Prediction distribution
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("PREDICTION DISTRIBUTION")
    print("-----------------------------------")

    prediction_distribution(
        predictions
    )

    # -----------------------------------------
    # Sample predictions
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("SAMPLE PREDICTIONS")
    print("-----------------------------------")

    for index in range(20):

        print(
            f"Sample {index + 1:2d}: "
            f"actual={y_test[index]} "
            f"predicted={predictions[index]}"
        )

    # -----------------------------------------
    # Comparison
    # -----------------------------------------

    print()
    print("===================================")
    print(" DECODER COMPARISON")
    print("===================================")

    print()
    print(
        "Traditional Lookup:"
    )

    print(
        "Exact ≈ 0.6070 logical "
        "success in previous experiment"
    )

    print()
    print(
        "Random Forest:"
    )

    print(
        "Exact = 0.5842"
    )

    print(
        "Bit   = 0.7277"
    )

    print()
    print(
        "Previous 3-output GRU:"
    )

    print(
        "Exact = 0.2950"
    )

    print(
        "Bit   = 0.6643"
    )

    print()
    print(
        "New 8-class GRU:"
    )

    print(
        f"Exact = {exact_accuracy:.4f}"
    )

    print(
        f"Bit   = {bits:.4f}"
    )

    # -----------------------------------------
    # Interpretation
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("INTERPRETATION")
    print("-----------------------------------")

    if exact_accuracy > 0.5842:

        print(
            "8-class GRU beats the previous "
            "Random Forest on exact-pattern "
            "prediction."
        )

    else:

        print(
            "8-class GRU does not yet beat "
            "the previous Random Forest."
        )

    print()

    if exact_accuracy > 0.2950:

        print(
            "Changing from three independent "
            "binary outputs to one 8-class "
            "error-pattern target substantially "
            "improves over the previous GRU."
        )

    else:

        print(
            "The 8-class formulation did not "
            "improve over the previous GRU."
        )

    print()
    print("-----------------------------------")
    print("SCIENTIFIC CONTROL")
    print("-----------------------------------")

    print(
        "Dataset, noise model, train/test split, "
        "sequence input and target definition "
        "remain unchanged."
    )

    print(
        "The major change is that the decoder "
        "directly predicts one of eight complete "
        "3-qubit error patterns."
    )

    print()
    print("===================================")
    print(
        "8-CLASS GRU EXPERIMENT : COMPLETE"
    )
    print("===================================")


if __name__ == "__main__":
    main()