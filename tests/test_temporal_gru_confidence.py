import torch
from collections import Counter, defaultdict

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
# TARGET
# ============================================================

def pattern_to_class(pattern):

    pattern = tuple(
        int(bit)
        for bit in pattern
    )

    return PATTERN_TO_CLASS[pattern]


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
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

    loss_function = (
        torch.nn.CrossEntropyLoss()
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
            or epoch % 10 == 0
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
# PREDICTION WITH PROBABILITIES
# ============================================================

def predict_with_probabilities(
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

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidences, classes = torch.max(
            probabilities,
            dim=1
        )

    return (
        classes.tolist(),
        confidences.tolist(),
        probabilities.tolist()
    )


# ============================================================
# EXACT ACCURACY
# ============================================================

def exact_accuracy(
    predictions,
    targets
):

    correct = 0

    for prediction, target in zip(
        predictions,
        targets
    ):

        if tuple(prediction) == tuple(target):
            correct += 1

    return correct / len(targets)


# ============================================================
# BIT ACCURACY
# ============================================================

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

        for p, t in zip(
            prediction,
            target
        ):

            if p == t:
                correct += 1

            total += 1

    return correct / total


# ============================================================
# CONFIDENCE BUCKET ANALYSIS
# ============================================================

def confidence_bucket_analysis(
    confidences,
    predictions,
    targets
):

    buckets = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 1.00)
    ]

    print()
    print("----------------------------------------------")
    print("CONFIDENCE → ACCURACY")
    print("----------------------------------------------")

    for lower, upper in buckets:

        indices = []

        for index, confidence in enumerate(
            confidences
        ):

            if lower <= confidence < upper:
                indices.append(index)

        if not indices:

            print(
                f"{lower:.2f}-{upper:.2f}: "
                f"no samples"
            )

            continue

        correct = 0

        for index in indices:

            predicted_pattern = tuple(
                ERROR_PATTERNS[
                    predictions[index]
                ]
            )

            target_pattern = tuple(
                targets[index]
            )

            if (
                predicted_pattern
                == target_pattern
            ):

                correct += 1

        accuracy = (
            correct
            / len(indices)
        )

        average_confidence = (
            sum(
                confidences[index]
                for index in indices
            )
            / len(indices)
        )

        print(
            f"{lower:.2f}-{upper:.2f}: "
            f"samples={len(indices):4d} "
            f"avg_conf={average_confidence:.4f} "
            f"accuracy={accuracy:.4f}"
        )


# ============================================================
# ACCEPTANCE THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    confidences,
    predictions,
    targets
):

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95
    ]

    print()
    print("----------------------------------------------")
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("----------------------------------------------")

    for threshold in thresholds:

        accepted = []

        for index, confidence in enumerate(
            confidences
        ):

            if confidence >= threshold:
                accepted.append(index)

        if not accepted:

            print(
                f">= {threshold:.2f}: "
                f"no samples"
            )

            continue

        correct = 0

        for index in accepted:

            predicted_pattern = tuple(
                ERROR_PATTERNS[
                    predictions[index]
                ]
            )

            target_pattern = tuple(
                targets[index]
            )

            if (
                predicted_pattern
                == target_pattern
            ):

                correct += 1

        accuracy = (
            correct
            / len(accepted)
        )

        coverage = (
            len(accepted)
            / len(targets)
        )

        print(
            f">= {threshold:.2f}: "
            f"accepted={len(accepted):4d} "
            f"coverage={coverage:.4f} "
            f"accuracy={accuracy:.4f}"
        )


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(
    confidences,
    predictions,
    targets,
    number_of_bins=10
):

    bins = [
        []
        for _ in range(number_of_bins)
    ]

    for index, confidence in enumerate(
        confidences
    ):

        bin_index = int(
            confidence
            * number_of_bins
        )

        if bin_index >= number_of_bins:
            bin_index = (
                number_of_bins - 1
            )

        predicted_pattern = tuple(
            ERROR_PATTERNS[
                predictions[index]
            ]
        )

        target_pattern = tuple(
            targets[index]
        )

        correct = (
            predicted_pattern
            == target_pattern
        )

        bins[bin_index].append(
            (
                confidence,
                correct
            )
        )

    ece = 0.0

    print()
    print("----------------------------------------------")
    print("CALIBRATION")
    print("----------------------------------------------")

    print(
        "Bin       Samples   Confidence   Accuracy"
    )

    for bin_index, values in enumerate(
        bins
    ):

        if not values:
            continue

        average_confidence = (
            sum(
                confidence
                for confidence, _
                in values
            )
            / len(values)
        )

        accuracy = (
            sum(
                int(correct)
                for _, correct
                in values
            )
            / len(values)
        )

        fraction = (
            len(values)
            / len(targets)
        )

        ece += (
            fraction
            * abs(
                average_confidence
                - accuracy
            )
        )

        lower = (
            bin_index
            / number_of_bins
        )

        upper = (
            (bin_index + 1)
            / number_of_bins
        )

        print(
            f"{lower:.1f}-{upper:.1f}   "
            f"{len(values):6d}   "
            f"{average_confidence:.4f}       "
            f"{accuracy:.4f}"
        )

    return ece


# ============================================================
# TOP-2 ANALYSIS
# ============================================================

def top_k_accuracy(
    probabilities,
    targets,
    k
):

    correct = 0

    for probability_vector, target in zip(
        probabilities,
        targets
    ):

        ranked = sorted(
            range(len(probability_vector)),
            key=lambda index:
                probability_vector[index],
            reverse=True
        )

        target_class = pattern_to_class(
            target
        )

        if target_class in ranked[:k]:
            correct += 1

    return correct / len(targets)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print(" TEMPORAL GRU CONFIDENCE ANALYSIS")
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

    print(
        f"Seed                      : "
        f"{SEED}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("----------------------------------------------")
    print("GENERATING DATASET")
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
        tuple(sample["final_error_state"])
        for sample in train_samples
    ]

    X_test = [
        encode_sequence(sample)
        for sample in test_samples
    ]

    y_test = [
        tuple(sample["final_error_state"])
        for sample in test_samples
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRAINING")
    print("----------------------------------------------")

    model = train_model(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICT
    # ========================================================

    (
        predicted_classes,
        confidences,
        probabilities
    ) = predict_with_probabilities(
        model,
        X_test
    )

    predictions = [
        list(
            ERROR_PATTERNS[
                class_index
            ]
        )
        for class_index in predicted_classes
    ]

    # ========================================================
    # BASIC METRICS
    # ========================================================

    exact = exact_accuracy(
        predictions,
        y_test
    )

    bit = bit_accuracy(
        predictions,
        y_test
    )

    print()
    print("==============================================")
    print("MODEL PERFORMANCE")
    print("==============================================")

    print()
    print(
        f"Test exact               : "
        f"{exact:.4f}"
    )

    print(
        f"Test bit                 : "
        f"{bit:.4f}"
    )

    print(
        f"Average confidence       : "
        f"{sum(confidences) / len(confidences):.4f}"
    )

    print(
        f"Minimum confidence       : "
        f"{min(confidences):.4f}"
    )

    print(
        f"Maximum confidence       : "
        f"{max(confidences):.4f}"
    )

    # ========================================================
    # CONFIDENCE BUCKETS
    # ========================================================

    confidence_bucket_analysis(
        confidences,
        predicted_classes,
        y_test
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    threshold_analysis(
        confidences,
        predicted_classes,
        y_test
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    ece = calculate_ece(
        confidences,
        predicted_classes,
        y_test
    )

    print()
    print(
        f"Expected Calibration Error : "
        f"{ece:.4f}"
    )

    # ========================================================
    # TOP-K
    # ========================================================

    top1 = top_k_accuracy(
        probabilities,
        y_test,
        1
    )

    top2 = top_k_accuracy(
        probabilities,
        y_test,
        2
    )

    top3 = top_k_accuracy(
        probabilities,
        y_test,
        3
    )

    print()
    print("----------------------------------------------")
    print("TOP-K ACCURACY")
    print("----------------------------------------------")

    print(
        f"Top-1 accuracy            : "
        f"{top1:.4f}"
    )

    print(
        f"Top-2 accuracy            : "
        f"{top2:.4f}"
    )

    print(
        f"Top-3 accuracy            : "
        f"{top3:.4f}"
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("PREDICTION DISTRIBUTION")
    print("----------------------------------------------")

    prediction_counter = Counter(
        tuple(prediction)
        for prediction in predictions
    )

    for pattern, count in (
        prediction_counter.most_common()
    ):

        percentage = (
            count
            / len(predictions)
            * 100
        )

        print(
            f"{pattern} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # HIGH-CONFIDENCE EXAMPLES
    # ========================================================

    print()
    print("----------------------------------------------")
    print("HIGH-CONFIDENCE EXAMPLES")
    print("----------------------------------------------")

    ranked_indices = sorted(
        range(len(confidences)),
        key=lambda index:
            confidences[index],
        reverse=True
    )

    for index in ranked_indices[:10]:

        predicted = predictions[index]
        actual = list(y_test[index])
        confidence = confidences[index]

        print(
            f"confidence={confidence:.4f} "
            f"predicted={predicted} "
            f"actual={actual} "
            f"correct={predicted == actual}"
        )

    # ========================================================
    # LOW-CONFIDENCE EXAMPLES
    # ========================================================

    print()
    print("----------------------------------------------")
    print("LOW-CONFIDENCE EXAMPLES")
    print("----------------------------------------------")

    low_ranked_indices = sorted(
        range(len(confidences)),
        key=lambda index:
            confidences[index]
    )

    for index in low_ranked_indices[:10]:

        predicted = predictions[index]
        actual = list(y_test[index])
        confidence = confidences[index]

        print(
            f"confidence={confidence:.4f} "
            f"predicted={predicted} "
            f"actual={actual} "
            f"correct={predicted == actual}"
        )

    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    print()
    print("==============================================")
    print("DIAGNOSIS")
    print("==============================================")

    if ece < 0.05:

        print()
        print(
            "GOOD CALIBRATION"
        )

        print(
            "The GRU confidence is reasonably "
            "aligned with its actual accuracy."
        )

    elif ece < 0.10:

        print()
        print(
            "MODERATE CALIBRATION ERROR"
        )

        print(
            "Confidence contains useful information "
            "but is not perfectly calibrated."
        )

    else:

        print()
        print(
            "POOR CALIBRATION"
        )

        print(
            "Raw neural-network confidence should "
            "not be treated as a reliable probability."
        )

    print()

    if top2 > top1 + 0.05:

        print(
            "Top-2 contains substantial additional "
            "information over Top-1."
        )

    else:

        print(
            "Top-2 provides limited improvement "
            "over Top-1."
        )

    print()
    print("==============================================")
    print(
        " GRU CONFIDENCE ANALYSIS : COMPLETE"
    )
    print("==============================================")


if __name__ == "__main__":
    main()