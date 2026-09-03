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

TRAINING_SIZES = [
    500,
    1000,
    2500,
    5000,
    10000,
    15000,
    20000
]

SEED = 42

HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.003


# ============================================================
# FEATURE ENCODING
# ============================================================

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


# ============================================================
# TARGET ENCODING
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
    print(" TEMPORAL GRU LEARNING CURVE")
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
        f"Total samples             : "
        f"{TOTAL_SAMPLES}"
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
        f"Epochs per model          : "
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

    print()
    print(
        "Training sizes:"
    )

    print(
        TRAINING_SIZES
    )

    # ========================================================
    # GENERATE ONE FIXED DATASET
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
    # FIXED TRAIN / TEST DATA
    # ========================================================

    train_pool = samples[
        :TOTAL_SAMPLES - TEST_SAMPLES
    ]

    test_samples = samples[
        TOTAL_SAMPLES - TEST_SAMPLES:
    ]

    X_test = [
        encode_syndrome_sequence(
            sample
        )
        for sample in test_samples
    ]

    y_test = [
        encode_target(
            sample
        )
        for sample in test_samples
    ]

    # ========================================================
    # TEST DISTRIBUTION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("FIXED TEST TARGET DISTRIBUTION")
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
    # LEARNING CURVE
    # ========================================================

    results = []

    print()
    print("==============================================")
    print(" RUNNING GRU LEARNING CURVE")
    print("==============================================")

    for training_size in TRAINING_SIZES:

        print()
        print()
        print("==============================================")
        print(
            f" TRAINING SIZE: {training_size}"
        )
        print("==============================================")

        # ----------------------------------------------------
        # Select first N samples from SAME training pool
        # ----------------------------------------------------

        train_samples = train_pool[
            :training_size
        ]

        X_train = [
            encode_syndrome_sequence(
                sample
            )
            for sample in train_samples
        ]

        y_train = [
            encode_target(
                sample
            )
            for sample in train_samples
        ]

        print()
        print(
            f"Training samples          : "
            f"{len(X_train)}"
        )

        # ----------------------------------------------------
        # Create a NEW model for each training size
        # ----------------------------------------------------

        decoder = TemporalGRUClassifier(
            input_size=2,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            random_seed=SEED
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        decoder.train(
            X_train,
            y_train,
            verbose=False
        )

        # ----------------------------------------------------
        # Training performance
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Test performance
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------

        prediction_counter = Counter(
            tuple(prediction)
            for prediction in test_predictions
        )

        number_of_predicted_classes = (
            len(prediction_counter)
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append(
            {
                "training_size": training_size,
                "train_exact": train_exact,
                "train_bit": train_bit,
                "test_exact": test_exact,
                "test_bit": test_bit,
                "predicted_classes": (
                    number_of_predicted_classes
                )
            }
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            "RESULT"
        )

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
            f"{number_of_predicted_classes}"
        )

        print()
        print(
            "Test prediction distribution:"
        )

        for prediction, count in (
            prediction_counter.most_common()
        ):

            percentage = (
                count
                / len(test_predictions)
                * 100
            )

            print(
                f"  {prediction} -> "
                f"{count:5d} "
                f"({percentage:6.2f}%)"
            )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print()
    print("==============================================")
    print(" FINAL GRU LEARNING CURVE")
    print("==============================================")

    print()

    print(
        "Training | "
        "Train Exact | "
        "Train Bit | "
        "Test Exact | "
        "Test Bit | "
        "Classes"
    )

    print(
        "---------|-------------|-----------|"
        "------------|----------|--------"
    )

    for result in results:

        print(
            f"{result['training_size']:8d} | "
            f"{result['train_exact']:.4f}      | "
            f"{result['train_bit']:.4f}    | "
            f"{result['test_exact']:.4f}     | "
            f"{result['test_bit']:.4f}   | "
            f"{result['predicted_classes']}"
        )

    # ========================================================
    # BEST RESULT
    # ========================================================

    best_result = max(
        results,
        key=lambda result: result["test_exact"]
    )

    print()
    print("----------------------------------------------")
    print("BEST GRU RESULT")
    print("----------------------------------------------")

    print(
        f"Training size            : "
        f"{best_result['training_size']}"
    )

    print(
        f"Test exact accuracy      : "
        f"{best_result['test_exact']:.4f}"
    )

    print(
        f"Test bit accuracy        : "
        f"{best_result['test_bit']:.4f}"
    )

    print(
        f"Predicted classes        : "
        f"{best_result['predicted_classes']}"
    )

    # ========================================================
    # PREVIOUS RESULTS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("PREVIOUS BENCHMARKS")
    print("----------------------------------------------")

    print(
        "Majority baseline        : 0.2950"
    )

    print(
        "Random Forest            : 0.5842 exact"
    )

    print(
        "Traditional lookup       : "
        "≈ 0.6070 logical success"
    )

    print(
        "Information ceiling      : "
        "≈ 0.6722"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("DIAGNOSIS")
    print("----------------------------------------------")

    first_test = results[0]["test_exact"]
    last_test = results[-1]["test_exact"]

    improvement = (
        last_test - first_test
    )

    print(
        f"Test exact improvement   : "
        f"{improvement:+.4f}"
    )

    if improvement > 0.05:

        print()
        print(
            "DATA BENEFIT DETECTED"
        )

        print(
            "More training data is "
            "improving GRU generalization."
        )

        print(
            "The learning curve has "
            "not completely saturated."
        )

    elif improvement > 0.01:

        print()
        print(
            "SMALL DATA BENEFIT"
        )

        print(
            "More data provides some "
            "generalization improvement."
        )

        print(
            "The curve may be approaching "
            "a plateau."
        )

    else:

        print()
        print(
            "DATA SATURATION"
        )

        print(
            "Increasing training data "
            "provides little benefit."
        )

        print(
            "The limitation may be "
            "representation/information "
            "rather than dataset size."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("==============================================")
    print(
        " GRU LEARNING CURVE EXPERIMENT : COMPLETE"
    )
    print("==============================================")


if __name__ == "__main__":
    main()