import torch

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

PHYSICAL_NOISE_LEVELS = [
    0.01,
    0.05,
    0.10,
    0.15,
    0.20
]

MEASUREMENT_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20
]

TOTAL_SAMPLES = 25000
TRAINING_SIZE = 20000
TEST_SAMPLES = 5000

SEED = 42

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 1

LEARNING_RATE = 0.003
EPOCHS = 100

HYBRID_THRESHOLD = 0.50


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

    return model


# ============================================================
# GRU PREDICTIONS
# ============================================================

def get_gru_predictions(
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
        confidences.tolist()
    )


# ============================================================
# XOR
# ============================================================

def xor_states(
    actual,
    predicted
):

    return [
        int(a) ^ int(p)
        for a, p in zip(
            actual,
            predicted
        )
    ]


# ============================================================
# LOGICAL SUCCESS
# ============================================================

def is_logical_success(
    corrected_state,
    logical_state
):

    recovery = LogicalRecovery()

    recovered = recovery.recover(
        corrected_state
    )

    return recovered == logical_state


# ============================================================
# EVALUATE GRU
# ============================================================

def evaluate_gru(
    test_samples,
    gru_classes
):

    physical_correct = 0
    logical_correct = 0

    for index, sample in enumerate(
        test_samples
    ):

        actual_error = list(
            sample["final_error_state"]
        )

        predicted = list(
            ERROR_PATTERNS[
                gru_classes[index]
            ]
        )

        corrected_state = xor_states(
            actual_error,
            predicted
        )

        if corrected_state == [
            0,
            0,
            0
        ]:

            physical_correct += 1

        if is_logical_success(
            corrected_state,
            sample["logical_state"]
        ):

            logical_correct += 1

    total = len(test_samples)

    return (
        physical_correct / total,
        logical_correct / total
    )


# ============================================================
# EVALUATE TRADITIONAL
# ============================================================

def evaluate_traditional(
    test_samples
):

    decoder = (
        RepeatedLookupDecoder()
    )

    physical_correct = 0
    logical_correct = 0

    for sample in test_samples:

        actual_error = list(
            sample["final_error_state"]
        )

        prediction = (
            decoder.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        corrected_state = xor_states(
            actual_error,
            prediction
        )

        if corrected_state == [
            0,
            0,
            0
        ]:

            physical_correct += 1

        if is_logical_success(
            corrected_state,
            sample["logical_state"]
        ):

            logical_correct += 1

    total = len(test_samples)

    return (
        physical_correct / total,
        logical_correct / total
    )


# ============================================================
# EVALUATE HYBRID
# ============================================================

def evaluate_hybrid(
    test_samples,
    gru_classes,
    gru_confidences,
    threshold
):

    decoder = (
        RepeatedLookupDecoder()
    )

    physical_correct = 0
    logical_correct = 0

    gru_used = 0
    fallback_used = 0

    for index, sample in enumerate(
        test_samples
    ):

        actual_error = list(
            sample["final_error_state"]
        )

        confidence = (
            gru_confidences[index]
        )

        gru_prediction = list(
            ERROR_PATTERNS[
                gru_classes[index]
            ]
        )

        fallback_prediction = (
            decoder.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        if confidence >= threshold:

            prediction = (
                gru_prediction
            )

            gru_used += 1

        else:

            prediction = (
                fallback_prediction
            )

            fallback_used += 1

        corrected_state = xor_states(
            actual_error,
            prediction
        )

        if corrected_state == [
            0,
            0,
            0
        ]:

            physical_correct += 1

        if is_logical_success(
            corrected_state,
            sample["logical_state"]
        ):

            logical_correct += 1

    total = len(test_samples)

    return {
        "physical":
            physical_correct / total,

        "logical":
            logical_correct / total,

        "gru_coverage":
            gru_used / total,

        "fallback":
            fallback_used / total
    }


# ============================================================
# RUN ONE NOISE CONFIGURATION
# ============================================================

def run_noise_configuration(
    physical_noise,
    measurement_noise
):

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                physical_noise
            ),
            measurement_noise_probability=(
                measurement_noise
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

    model = train_model(
        X_train,
        y_train
    )

    (
        gru_classes,
        gru_confidences
    ) = get_gru_predictions(
        model,
        X_test
    )

    # --------------------------------------------------------
    # Traditional
    # --------------------------------------------------------

    (
        traditional_physical,
        traditional_logical
    ) = evaluate_traditional(
        test_samples
    )

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    (
        gru_physical,
        gru_logical
    ) = evaluate_gru(
        test_samples,
        gru_classes
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid = evaluate_hybrid(
        test_samples,
        gru_classes,
        gru_confidences,
        HYBRID_THRESHOLD
    )

    return {
        "physical_noise":
            physical_noise,

        "measurement_noise":
            measurement_noise,

        "traditional_physical":
            traditional_physical,

        "traditional_logical":
            traditional_logical,

        "gru_physical":
            gru_physical,

        "gru_logical":
            gru_logical,

        "hybrid_physical":
            hybrid["physical"],

        "hybrid_logical":
            hybrid["logical"],

        "hybrid_coverage":
            hybrid["gru_coverage"],

        "hybrid_fallback":
            hybrid["fallback"]
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print()
    print(
        f"Physical={result['physical_noise']:.2f} "
        f"Measurement={result['measurement_noise']:.2f}"
    )

    print(
        f"Traditional  "
        f"Physical={result['traditional_physical']:.4f} "
        f"Logical={result['traditional_logical']:.4f}"
    )

    print(
        f"GRU          "
        f"Physical={result['gru_physical']:.4f} "
        f"Logical={result['gru_logical']:.4f}"
    )

    print(
        f"Hybrid       "
        f"Physical={result['hybrid_physical']:.4f} "
        f"Logical={result['hybrid_logical']:.4f} "
        f"Coverage={result['hybrid_coverage']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("============================================================")
    print(" QEC AI NOISE SWEEP")
    print("============================================================")

    print()
    print(
        f"Rounds                 : {ROUNDS}"
    )

    print(
        f"Training samples       : {TRAINING_SIZE}"
    )

    print(
        f"Test samples           : {TEST_SAMPLES}"
    )

    print(
        f"GRU hidden size        : {HIDDEN_SIZE}"
    )

    print(
        f"Epochs                 : {EPOCHS}"
    )

    print(
        f"Hybrid threshold       : "
        f"{HYBRID_THRESHOLD:.2f}"
    )

    print()
    print(
        "Physical noise levels : "
        f"{PHYSICAL_NOISE_LEVELS}"
    )

    print(
        "Measurement levels     : "
        f"{MEASUREMENT_NOISE_LEVELS}"
    )

    results = []

    # ========================================================
    # FULL GRID
    # ========================================================

    total_runs = (
        len(PHYSICAL_NOISE_LEVELS)
        *
        len(MEASUREMENT_NOISE_LEVELS)
    )

    current_run = 0

    for physical_noise in (
        PHYSICAL_NOISE_LEVELS
    ):

        for measurement_noise in (
            MEASUREMENT_NOISE_LEVELS
        ):

            current_run += 1

            print()
            print(
                "============================================================"
            )

            print(
                f"RUN {current_run}/{total_runs}"
            )

            print(
                f"Physical noise      : "
                f"{physical_noise:.2f}"
            )

            print(
                f"Measurement noise   : "
                f"{measurement_noise:.2f}"
            )

            print(
                "============================================================"
            )

            result = run_noise_configuration(
                physical_noise,
                measurement_noise
            )

            results.append(result)

            print_result(
                result
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("============================================================")
    print(" NOISE SWEEP SUMMARY")
    print("============================================================")

    print()

    print(
        "PNoise MNoise | "
        "Traditional Logical | "
        "GRU Logical | "
        "Hybrid Logical | "
        "Hybrid Cov"
    )

    print(
        "--------------|"
        "--------------------|"
        "------------|"
        "---------------|"
        "-----------"
    )

    for result in results:

        print(
            f"{result['physical_noise']:.2f}    "
            f"{result['measurement_noise']:.2f}   | "
            f"{result['traditional_logical']:.4f}              | "
            f"{result['gru_logical']:.4f}      | "
            f"{result['hybrid_logical']:.4f}         | "
            f"{result['hybrid_coverage']:.4f}"
        )

    # ========================================================
    # BEST RESULTS
    # ========================================================

    best_traditional = max(
        results,
        key=lambda r:
            r["traditional_logical"]
    )

    best_gru = max(
        results,
        key=lambda r:
            r["gru_logical"]
    )

    best_hybrid = max(
        results,
        key=lambda r:
            r["hybrid_logical"]
    )

    print()
    print("============================================================")
    print(" BEST RESULTS")
    print("============================================================")

    print()

    print(
        "Traditional best:"
    )

    print(
        f"Noise = "
        f"({best_traditional['physical_noise']:.2f}, "
        f"{best_traditional['measurement_noise']:.2f}) "
        f"Logical = "
        f"{best_traditional['traditional_logical']:.4f}"
    )

    print()

    print(
        "GRU best:"
    )

    print(
        f"Noise = "
        f"({best_gru['physical_noise']:.2f}, "
        f"{best_gru['measurement_noise']:.2f}) "
        f"Logical = "
        f"{best_gru['gru_logical']:.4f}"
    )

    print()

    print(
        "Hybrid best:"
    )

    print(
        f"Noise = "
        f"({best_hybrid['physical_noise']:.2f}, "
        f"{best_hybrid['measurement_noise']:.2f}) "
        f"Logical = "
        f"{best_hybrid['hybrid_logical']:.4f}"
    )

    # ========================================================
    # WIN COUNTS
    # ========================================================

    traditional_wins = 0
    gru_wins = 0
    hybrid_wins = 0

    for result in results:

        values = {
            "traditional":
                result["traditional_logical"],

            "gru":
                result["gru_logical"],

            "hybrid":
                result["hybrid_logical"]
        }

        winner = max(
            values,
            key=values.get
        )

        if winner == "traditional":
            traditional_wins += 1

        elif winner == "gru":
            gru_wins += 1

        elif winner == "hybrid":
            hybrid_wins += 1

    print()
    print("============================================================")
    print(" WIN COUNT")
    print("============================================================")

    print(
        f"Traditional wins : "
        f"{traditional_wins}/{total_runs}"
    )

    print(
        f"GRU wins         : "
        f"{gru_wins}/{total_runs}"
    )

    print(
        f"Hybrid wins      : "
        f"{hybrid_wins}/{total_runs}"
    )

    # ========================================================
    # AVERAGES
    # ========================================================

    average_traditional = (
        sum(
            r["traditional_logical"]
            for r in results
        )
        / len(results)
    )

    average_gru = (
        sum(
            r["gru_logical"]
            for r in results
        )
        / len(results)
    )

    average_hybrid = (
        sum(
            r["hybrid_logical"]
            for r in results
        )
        / len(results)
    )

    average_hybrid_coverage = (
        sum(
            r["hybrid_coverage"]
            for r in results
        )
        / len(results)
    )

    print()
    print("============================================================")
    print(" AVERAGE LOGICAL SUCCESS")
    print("============================================================")

    print(
        f"Traditional average : "
        f"{average_traditional:.4f}"
    )

    print(
        f"GRU average         : "
        f"{average_gru:.4f}"
    )

    print(
        f"Hybrid average      : "
        f"{average_hybrid:.4f}"
    )

    print(
        f"Hybrid coverage     : "
        f"{average_hybrid_coverage:.4f}"
    )

    # ========================================================
    # FINAL DIAGNOSIS
    # ========================================================

    print()
    print("============================================================")
    print(" DIAGNOSIS")
    print("============================================================")

    if average_hybrid > average_traditional:

        print()
        print(
            "HYBRID BEATS TRADITIONAL ON AVERAGE"
        )

    else:

        print()
        print(
            "TRADITIONAL BEATS OR MATCHES HYBRID ON AVERAGE"
        )

    if average_gru > average_traditional:

        print(
            "GRU BEATS TRADITIONAL ON AVERAGE"
        )

    else:

        print(
            "GRU DOES NOT BEAT TRADITIONAL ON AVERAGE"
        )

    if average_hybrid > average_gru:

        print(
            "HYBRID BEATS GRU-ALWAYS ON AVERAGE"
        )

    else:

        print(
            "GRU-ALWAYS BEATS OR MATCHES HYBRID ON AVERAGE"
        )

    print()
    print("============================================================")
    print(" QEC AI NOISE SWEEP : COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    main()