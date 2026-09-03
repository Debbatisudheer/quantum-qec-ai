import torch
from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    GRUClassifierNetwork,
    ERROR_PATTERNS,
    PATTERN_TO_CLASS
)

from decoders.repeated_lookup import (
    RepeatedLookupDecoder
)

from evaluation.logical_recovery import (
    LogicalRecovery
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

CONFIDENCE_THRESHOLDS = [
    0.00,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


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

    if pattern not in PATTERN_TO_CLASS:
        raise ValueError(
            f"Unknown target pattern: {pattern}"
        )

    return PATTERN_TO_CLASS[pattern]


# ============================================================
# TRAIN GRU
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
# GRU PREDICTION
# ============================================================

def gru_predictions(
    model,
    X_test
):

    X_tensor = torch.tensor(
        X_test,
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
# XOR STATES
# ============================================================

def xor_states(
    actual,
    predicted
):

    if len(actual) != 3:
        raise ValueError(
            "actual must contain 3 bits"
        )

    if len(predicted) != 3:
        raise ValueError(
            "predicted must contain 3 bits"
        )

    return [
        int(a) ^ int(p)
        for a, p in zip(
            actual,
            predicted
        )
    ]


# ============================================================
# LOGICAL RECOVERY
# ============================================================

def logical_success(
    corrected_state,
    logical_state
):

    recovery = LogicalRecovery()

    recovered = recovery.recover(
        corrected_state
    )

    return (
        recovered == logical_state
    )


# ============================================================
# DECODER EVALUATION
# ============================================================

def evaluate_strategy(
    test_samples,
    gru_classes,
    gru_confidences,
    lookup_decoder,
    threshold
):

    physical_correct = 0
    logical_correct = 0

    gru_used = 0
    fallback_used = 0

    accepted_confidences = []

    for index, sample in enumerate(
        test_samples
    ):

        actual_error = list(
            sample["final_error_state"]
        )

        logical_state = int(
            sample["logical_state"]
        )

        confidence = (
            gru_confidences[index]
        )

        # ----------------------------------------------------
        # GRU prediction
        # ----------------------------------------------------

        gru_prediction = list(
            ERROR_PATTERNS[
                gru_classes[index]
            ]
        )

        # ----------------------------------------------------
        # Traditional fallback
        # ----------------------------------------------------

        fallback_prediction = (
            lookup_decoder.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        # ----------------------------------------------------
        # Confidence gate
        # ----------------------------------------------------

        if confidence >= threshold:

            prediction = (
                gru_prediction
            )

            gru_used += 1

            accepted_confidences.append(
                confidence
            )

        else:

            prediction = (
                fallback_prediction
            )

            fallback_used += 1

        # ----------------------------------------------------
        # Apply correction
        # ----------------------------------------------------

        corrected_state = xor_states(
            actual_error,
            prediction
        )

        # ----------------------------------------------------
        # Physical success
        # ----------------------------------------------------

        if corrected_state == [0, 0, 0]:

            physical_correct += 1

        # ----------------------------------------------------
        # Logical recovery
        # ----------------------------------------------------

        if logical_success(
            corrected_state,
            logical_state
        ):

            logical_correct += 1

    total = len(test_samples)

    return {
        "physical_success":
            physical_correct / total,

        "logical_success":
            logical_correct / total,

        "gru_used":
            gru_used,

        "fallback_used":
            fallback_used,

        "gru_coverage":
            gru_used / total,

        "fallback_rate":
            fallback_used / total,

        "average_accepted_confidence":
            (
                sum(accepted_confidences)
                / len(accepted_confidences)
                if accepted_confidences
                else 0.0
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print(" CONFIDENCE-GATED HYBRID DECODER")
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
    # GENERATE DATASET
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
        tuple(
            sample["final_error_state"]
        )
        for sample in train_samples
    ]

    X_test = [
        encode_sequence(sample)
        for sample in test_samples
    ]

    y_test = [
        tuple(
            sample["final_error_state"]
        )
        for sample in test_samples
    ]

    print(
        f"Generated samples        : "
        f"{len(samples)}"
    )

    print(
        f"Training samples         : "
        f"{len(train_samples)}"
    )

    print(
        f"Test samples             : "
        f"{len(test_samples)}"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRAINING GRU")
    print("----------------------------------------------")

    model = train_model(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    (
        gru_classes,
        gru_confidences,
        probabilities
    ) = gru_predictions(
        model,
        X_test
    )

    # ========================================================
    # LOOKUP DECODER
    # ========================================================

    lookup_decoder = (
        RepeatedLookupDecoder()
    )

    # ========================================================
    # GRU BASELINE
    # ========================================================

    print()
    print("----------------------------------------------")
    print("GRU BASELINE")
    print("----------------------------------------------")

    gru_result = evaluate_strategy(
        test_samples,
        gru_classes,
        gru_confidences,
        lookup_decoder,
        threshold=0.00
    )

    print(
        f"Physical success         : "
        f"{gru_result['physical_success']:.4f}"
    )

    print(
        f"Logical success          : "
        f"{gru_result['logical_success']:.4f}"
    )

    print(
        f"GRU coverage             : "
        f"{gru_result['gru_coverage']:.4f}"
    )

    # ========================================================
    # TRADITIONAL BASELINE
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRADITIONAL LOOKUP BASELINE")
    print("----------------------------------------------")

    traditional_result = evaluate_strategy(
        test_samples,
        gru_classes,
        gru_confidences,
        lookup_decoder,
        threshold=1.01
    )

    print(
        f"Physical success         : "
        f"{traditional_result['physical_success']:.4f}"
    )

    print(
        f"Logical success          : "
        f"{traditional_result['logical_success']:.4f}"
    )

    # ========================================================
    # CONFIDENCE-GATED HYBRID
    # ========================================================

    print()
    print("==============================================")
    print("CONFIDENCE-GATED RESULTS")
    print("==============================================")

    print()

    print(
        "Threshold | GRU Cov | Fallback | "
        "Physical | Logical"
    )

    print(
        "----------|---------|----------|"
        "----------|--------"
    )

    results = {}

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        result = evaluate_strategy(
            test_samples,
            gru_classes,
            gru_confidences,
            lookup_decoder,
            threshold
        )

        results[threshold] = result

        print(
            f"{threshold:9.2f} | "
            f"{result['gru_coverage']:.4f}  | "
            f"{result['fallback_rate']:.4f}   | "
            f"{result['physical_success']:.4f}   | "
            f"{result['logical_success']:.4f}"
        )

    # ========================================================
    # BEST LOGICAL STRATEGY
    # ========================================================

    best_threshold = max(
        results,
        key=lambda threshold:
            results[threshold][
                "logical_success"
            ]
    )

    best_result = results[
        best_threshold
    ]

    # ========================================================
    # BEST PHYSICAL STRATEGY
    # ========================================================

    best_physical_threshold = max(
        results,
        key=lambda threshold:
            results[threshold][
                "physical_success"
            ]
    )

    best_physical_result = results[
        best_physical_threshold
    ]

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("==============================================")
    print("FINAL COMPARISON")
    print("==============================================")

    print()

    print(
        "Strategy                  Physical    Logical"
    )

    print(
        "------------------------------------------------"
    )

    print(
        f"Traditional Lookup        "
        f"{traditional_result['physical_success']:.4f}      "
        f"{traditional_result['logical_success']:.4f}"
    )

    print(
        f"GRU Always                "
        f"{gru_result['physical_success']:.4f}      "
        f"{gru_result['logical_success']:.4f}"
    )

    print(
        f"Best Hybrid "
        f"(threshold={best_threshold:.2f})     "
        f"{best_result['physical_success']:.4f}      "
        f"{best_result['logical_success']:.4f}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    hybrid_logical_gain = (
        best_result["logical_success"]
        - traditional_result["logical_success"]
    )

    hybrid_physical_gain = (
        best_result["physical_success"]
        - traditional_result["physical_success"]
    )

    print()
    print("----------------------------------------------")
    print("HYBRID IMPROVEMENT")
    print("----------------------------------------------")

    print(
        f"Logical improvement      : "
        f"{hybrid_logical_gain:+.4f}"
    )

    print(
        f"Physical improvement     : "
        f"{hybrid_physical_gain:+.4f}"
    )

    print()
    print(
        f"Best logical threshold   : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best physical threshold  : "
        f"{best_physical_threshold:.2f}"
    )

    print(
        f"Best logical success     : "
        f"{best_result['logical_success']:.4f}"
    )

    print(
        f"Best physical success    : "
        f"{best_physical_result['physical_success']:.4f}"
    )

    print(
        f"Best hybrid GRU coverage : "
        f"{best_result['gru_coverage']:.4f}"
    )

    print(
        f"Best hybrid fallback    : "
        f"{best_result['fallback_rate']:.4f}"
    )

    # ========================================================
    # ADDITIONAL INTERPRETATION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("INTERPRETATION")
    print("----------------------------------------------")

    if (
        best_result["logical_success"]
        >
        gru_result["logical_success"]
    ):

        print(
            "Confidence gating improves "
            "logical success compared with "
            "GRU always."
        )

    elif (
        best_result["logical_success"]
        ==
        gru_result["logical_success"]
    ):

        print(
            "Confidence gating produces the "
            "same logical success as GRU always."
        )

    else:

        print(
            "Confidence gating does not improve "
            "logical success compared with GRU always."
        )

    if (
        best_result["logical_success"]
        >
        traditional_result["logical_success"]
    ):

        print(
            "Hybrid decoding beats the traditional "
            "lookup baseline."
        )

    elif (
        best_result["logical_success"]
        ==
        traditional_result["logical_success"]
    ):

        print(
            "Hybrid decoding matches the traditional "
            "lookup baseline."
        )

    else:

        print(
            "Traditional lookup remains better "
            "than the hybrid decoder."
        )

    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    print()
    print("==============================================")
    print("DIAGNOSIS")
    print("==============================================")

    if hybrid_logical_gain > 0:

        print()
        print(
            "HYBRID DECODER IMPROVES LOGICAL SUCCESS"
        )

        print(
            "The confidence-gated GRU provides "
            "a measurable improvement over the "
            "traditional decoder."
        )

        print(
            "This strategy should be retained "
            "for further testing."
        )

    elif hybrid_logical_gain == 0:

        print()
        print(
            "HYBRID DECODER MATCHES BASELINE"
        )

        print(
            "Confidence gating does not provide "
            "an improvement under this configuration."
        )

    else:

        print()
        print(
            "HYBRID DECODER DOES NOT IMPROVE"
        )

        print(
            "The traditional decoder remains "
            "stronger for logical QEC performance."
        )

    print()
    print("==============================================")
    print(
        " CONFIDENCE-GATED HYBRID : COMPLETE"
    )
    print("==============================================")


if __name__ == "__main__":
    main()