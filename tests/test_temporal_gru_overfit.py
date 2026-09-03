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

SAMPLES = 500

SEED = 42

HIDDEN_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.003


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


def main():

    print()
    print("===================================")
    print(" TEMPORAL GRU OVERFIT SANITY TEST")
    print("===================================")

    print()
    print(
        f"Samples                   : {SAMPLES}"
    )

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
        f"Hidden size               : "
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

    # -----------------------------------------
    # Generate small dataset
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
        SAMPLES
    )

    X = [
        encode_syndrome_sequence(
            sample
        )
        for sample in samples
    ]

    y = [
        encode_target(sample)
        for sample in samples
    ]

    print()
    print("-----------------------------------")
    print("TARGET DISTRIBUTION")
    print("-----------------------------------")

    counter = Counter(
        tuple(target)
        for target in y
    )

    for target, count in (
        counter.most_common()
    ):

        print(
            f"{target} -> "
            f"{count}"
        )

    # -----------------------------------------
    # Create model
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
        X,
        y,
        verbose=True
    )

    # -----------------------------------------
    # Predict on SAME training data
    # -----------------------------------------

    predictions = decoder.predict(
        X
    )

    exact = (
        exact_pattern_accuracy(
            predictions,
            y
        )
    )

    bits = bit_accuracy(
        predictions,
        y
    )

    print()
    print("-----------------------------------")
    print("TRAINING SET PERFORMANCE")
    print("-----------------------------------")

    print(
        f"Exact pattern accuracy : "
        f"{exact:.4f}"
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

    prediction_counter = Counter(
        tuple(prediction)
        for prediction in predictions
    )

    for prediction, count in (
        prediction_counter.most_common()
    ):

        percentage = (
            count
            / len(predictions)
            * 100
        )

        print(
            f"{prediction} -> "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------------------
    # Sample predictions
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print("SAMPLE PREDICTIONS")
    print("-----------------------------------")

    for index in range(
        min(20, len(y))
    ):

        print(
            f"Sample {index + 1:2d}: "
            f"actual={y[index]} "
            f"predicted={predictions[index]}"
        )

    # -----------------------------------------
    # Final diagnosis
    # -----------------------------------------

    print()
    print("===================================")
    print(" DIAGNOSIS")
    print("===================================")

    if exact >= 0.90:

        print(
            "PASS: GRU can memorize the "
            "training dataset."
        )

        print(
            "The implementation can learn "
            "the mapping."
        )

        print(
            "The remaining problem is likely "
            "generalization/information."
        )

    elif exact >= 0.70:

        print(
            "PARTIAL: GRU learns the training "
            "data but does not fully memorize it."
        )

        print(
            "Training configuration may need "
            "further investigation."
        )

    else:

        print(
            "FAIL: GRU cannot sufficiently "
            "memorize the training dataset."
        )

        print(
            "Do NOT move to more advanced "
            "architectures yet."
        )

        print(
            "We need to investigate the "
            "training implementation."
        )

    print()
    print("===================================")


if __name__ == "__main__":
    main()