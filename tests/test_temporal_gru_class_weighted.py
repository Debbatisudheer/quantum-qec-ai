import torch
import torch.nn as nn
from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    GRUClassifierNetwork,
    ERROR_PATTERNS,
    PATTERN_TO_CLASS
)


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10

TOTAL_SAMPLES = 25000
TRAINING_SIZE = 20000
TEST_SAMPLES = 5000

SEED = 42

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1

LEARNING_RATE = 0.003
EPOCHS = 100


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
        sample["observed_syndrome_history"]
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
# TARGET ENCODING
# ============================================================

def pattern_to_class(pattern):

    pattern = tuple(
        int(bit)
        for bit in pattern
    )

    if pattern not in PATTERN_TO_CLASS:
        raise ValueError(
            f"Unknown error pattern: {pattern}"
        )

    return PATTERN_TO_CLASS[pattern]


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
# CLASS-WEIGHT CALCULATION
# ============================================================

def calculate_class_weights(y_train):

    class_counts = Counter(
        pattern_to_class(target)
        for target in y_train
    )

    number_of_classes = len(
        ERROR_PATTERNS
    )

    total_samples = len(y_train)

    weights = []

    print()
    print("----------------------------------------------")
    print("TRAINING CLASS DISTRIBUTION")
    print("----------------------------------------------")

    for class_index in range(
        number_of_classes
    ):

        count = class_counts.get(
            class_index,
            0
        )

        pattern = ERROR_PATTERNS[
            class_index
        ]

        if count == 0:
            weight = 0.0
        else:
            weight = (
                total_samples
                /
                (
                    number_of_classes
                    * count
                )
            )

        weights.append(weight)

        percentage = (
            count
            / total_samples
            * 100
        )

        print(
            f"Class {class_index}: "
            f"{pattern} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%) "
            f"weight={weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# TRAINING
# ============================================================

def train_weighted_model(
    X_train,
    y_train,
    class_weights
):

    torch.manual_seed(SEED)

    model = GRUClassifierNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=8
    )

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_classes = [
        pattern_to_class(target)
        for target in y_train
    ]

    y_tensor = torch.tensor(
        y_classes,
        dtype=torch.long
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()

        logits = model(X_tensor)

        loss = loss_function(
            logits,
            y_tensor
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            predictions = torch.argmax(
                logits,
                dim=1
            )

            accuracy = (
                (
                    predictions
                    == y_tensor
                )
                .float()
                .mean()
                .item()
            )

            print(
                f"Epoch {epoch:3d}/{EPOCHS} "
                f"Loss={loss.item():.6f} "
                f"Train exact={accuracy:.4f}"
            )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        classes = torch.argmax(
            logits,
            dim=1
        )

    predictions = []

    for class_index in classes.tolist():

        predictions.append(
            list(
                ERROR_PATTERNS[
                    class_index
                ]
            )
        )

    return predictions


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

def print_per_class_accuracy(
    predictions,
    targets
):

    print()
    print("----------------------------------------------")
    print("PER-CLASS TEST ACCURACY")
    print("----------------------------------------------")

    for pattern in ERROR_PATTERNS:

        pattern_tuple = tuple(pattern)

        indices = [
            index
            for index, target in enumerate(
                targets
            )
            if tuple(target) == pattern_tuple
        ]

        if not indices:
            continue

        correct = sum(
            1
            for index in indices
            if tuple(
                predictions[index]
            ) == pattern_tuple
        )

        accuracy = (
            correct
            / len(indices)
        )

        print(
            f"{pattern} -> "
            f"{correct:4d}/"
            f"{len(indices):4d} "
            f"({accuracy:.4f})"
        )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

def print_prediction_distribution(
    predictions,
    total
):

    counter = Counter(
        tuple(prediction)
        for prediction in predictions
    )

    print()
    print("----------------------------------------------")
    print("TEST PREDICTION DISTRIBUTION")
    print("----------------------------------------------")

    for pattern, count in (
        counter.most_common()
    ):

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"{pattern} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print(" CLASS-WEIGHTED TEMPORAL GRU")
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
        f"Input size                : "
        f"{INPUT_SIZE}"
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
    # GENERATE DATASET
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
    # FIXED SPLIT
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
        list(sample["final_error_state"])
        for sample in train_samples
    ]

    X_test = [
        encode_sequence(sample)
        for sample in test_samples
    ]

    y_test = [
        list(sample["final_error_state"])
        for sample in test_samples
    ]

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            y_train
        )
    )

    print()
    print(
        "Class weights:"
    )

    print(
        class_weights.tolist()
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRAINING CLASS-WEIGHTED GRU")
    print("----------------------------------------------")

    model = train_weighted_model(
        X_train,
        y_train,
        class_weights
    )

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_predictions = predict(
        model,
        X_train
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
    # TEST METRICS
    # ========================================================

    test_predictions = predict(
        model,
        X_test
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

    # ========================================================
    # PER CLASS
    # ========================================================

    print_per_class_accuracy(
        test_predictions,
        y_test
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print_prediction_distribution(
        test_predictions,
        len(y_test)
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("==============================================")
    print(" COMPARISON")
    print("==============================================")

    print()
    print(
        "Previous unweighted GRU:"
    )

    print(
        "Input  = [5 × 4]"
    )

    print(
        "Exact  = 0.6222"
    )

    print(
        "Bit    = 0.7357"
    )

    print()
    print(
        "Class-weighted GRU:"
    )

    print(
        f"Input  = [5 × 4]"
    )

    print(
        f"Exact  = {test_exact:.4f}"
    )

    print(
        f"Bit    = {test_bit:.4f}"
    )

    exact_improvement = (
        test_exact - 0.6222
    )

    bit_improvement = (
        test_bit - 0.7357
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
            "Class imbalance was a meaningful "
            "limitation."
        )

        print(
            "The weighted loss improved learning "
            "of the error-pattern classes."
        )

    elif exact_improvement > 0.005:

        print()
        print(
            "SMALL IMPROVEMENT"
        )

        print(
            "Class weighting helps somewhat, "
            "but imbalance is not the main "
            "bottleneck."
        )

    elif exact_improvement >= -0.005:

        print()
        print(
            "NO MEANINGFUL IMPROVEMENT"
        )

        print(
            "Class weighting does not materially "
            "improve exact-pattern prediction."
        )

        print(
            "The main limitation is probably "
            "information/representation rather "
            "than class imbalance."
        )

    else:

        print()
        print(
            "PERFORMANCE DECREASE"
        )

        print(
            "Class weighting hurts overall "
            "exact-pattern accuracy."
        )

        print(
            "The model may be trading common-class "
            "accuracy for rare-class predictions."
        )

    print()
    print("==============================================")
    print(" CLASS-WEIGHTED GRU : COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()