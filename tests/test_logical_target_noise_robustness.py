from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_gru import (
    LogicalTargetGRUDecoder
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

TRAINING_SAMPLES = 5000

TEST_SAMPLES = 1000

HIDDEN_SIZE = 64

LEARNING_RATE = 0.003

EPOCHS = 100

SEED = 42


PHYSICAL_NOISE_LEVELS = [
    0.01,
    0.05,
    0.10,
    0.20
]


MEASUREMENT_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20
]


# ============================================================
# HELPERS
# ============================================================

def xor_states(a, b):
    """
    XOR two binary states.
    """

    if len(a) != len(b):
        raise ValueError(
            "States must have the same length"
        )

    return [
        int(x) ^ int(y)
        for x, y in zip(a, b)
    ]


def create_encoded_state(logical_state):
    """
    3-qubit repetition-code encoding.

        0 -> 000
        1 -> 111
    """

    if logical_state not in (0, 1):
        raise ValueError(
            "logical_state must be 0 or 1"
        )

    return [
        logical_state,
        logical_state,
        logical_state
    ]


def calculate_logical_success(
    logical_state,
    actual_error,
    correction,
    recovery
):
    """
    Determine whether a correction
    preserves the original logical state.
    """

    encoded_state = create_encoded_state(
        logical_state
    )

    corrupted_state = xor_states(
        encoded_state,
        actual_error
    )

    corrected_state = xor_states(
        corrupted_state,
        correction
    )

    recovered_logical = recovery.recover(
        corrected_state
    )

    return recovered_logical == logical_state


# ============================================================
# TRAIN ONE AI DECODER
# ============================================================

def train_decoder(
    physical_noise,
    measurement_noise,
    seed
):
    """
    Generate training data and train
    one logical-target GRU.
    """

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                physical_noise
            ),
            measurement_noise_probability=(
                measurement_noise
            ),
            seed=seed
        )
    )

    training_samples = (
        generator.generate_dataset(
            TRAINING_SAMPLES
        )
    )

    decoder = LogicalTargetGRUDecoder(
        rounds=ROUNDS,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=seed
    )

    decoder.train(
        training_samples,
        verbose=False
    )

    return decoder


# ============================================================
# EVALUATE ONE DECODER
# ============================================================

def evaluate_decoder(
    decoder,
    test_samples,
    recovery
):
    """
    Evaluate logical success on independent
    test samples.
    """

    success_count = 0

    for sample in test_samples:

        logical_state = int(
            sample["logical_state"]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        correction = decoder.decode(
            sample
        )

        success = calculate_logical_success(
            logical_state,
            actual_error,
            correction,
            recovery
        )

        if success:
            success_count += 1

    return (
        success_count
        / len(test_samples)
    )


# ============================================================
# EVALUATE TRADITIONAL DECODER
# ============================================================

def evaluate_traditional(
    test_samples,
    recovery
):
    """
    Evaluate the traditional repeated
    lookup decoder.
    """

    decoder = RepeatedLookupDecoder()

    success_count = 0

    for sample in test_samples:

        logical_state = int(
            sample["logical_state"]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        correction = (
            decoder.decode_history(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

        success = calculate_logical_success(
            logical_state,
            actual_error,
            correction,
            recovery
        )

        if success:
            success_count += 1

    return (
        success_count
        / len(test_samples)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        " LOGICAL-TARGET GRU "
        "NOISE ROBUSTNESS EXPERIMENT"
    )
    print("=" * 78)

    print()
    print(
        f"Rounds              : {ROUNDS}"
    )

    print(
        f"Training samples    : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples        : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"GRU hidden size     : "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"GRU epochs          : "
        f"{EPOCHS}"
    )

    print()

    recovery = LogicalRecovery()

    results = []

    # ========================================================
    # NOISE GRID
    # ========================================================

    for physical_noise in (
        PHYSICAL_NOISE_LEVELS
    ):

        for measurement_noise in (
            MEASUREMENT_NOISE_LEVELS
        ):

            print()
            print(
                "-" * 78
            )

            print(
                f"Physical noise     : "
                f"{physical_noise:.2f}"
            )

            print(
                f"Measurement noise  : "
                f"{measurement_noise:.2f}"
            )

            # ------------------------------------------------
            # TRAIN AI ON THIS NOISE CONDITION
            # ------------------------------------------------

            ai_decoder = train_decoder(
                physical_noise,
                measurement_noise,
                SEED
            )

            # ------------------------------------------------
            # CREATE INDEPENDENT TEST SET
            # ------------------------------------------------

            test_generator = (
                TimeVaryingQECDatasetGenerator(
                    rounds=ROUNDS,
                    physical_error_probability=(
                        physical_noise
                    ),
                    measurement_noise_probability=(
                        measurement_noise
                    ),
                    seed=12345
                )
            )

            test_samples = (
                test_generator.generate_dataset(
                    TEST_SAMPLES
                )
            )

            # ------------------------------------------------
            # TRADITIONAL
            # ------------------------------------------------

            traditional_score = (
                evaluate_traditional(
                    test_samples,
                    recovery
                )
            )

            # ------------------------------------------------
            # AI
            # ------------------------------------------------

            ai_score = evaluate_decoder(
                ai_decoder,
                test_samples,
                recovery
            )

            # ------------------------------------------------
            # GAIN
            # ------------------------------------------------

            gain = (
                ai_score
                - traditional_score
            )

            results.append(
                {
                    "physical_noise":
                        physical_noise,

                    "measurement_noise":
                        measurement_noise,

                    "traditional":
                        traditional_score,

                    "logical_target_gru":
                        ai_score,

                    "gain":
                        gain
                }
            )

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print()

            print(
                f"Traditional logical : "
                f"{traditional_score:.4f}"
            )

            print(
                f"Logical-target GRU  : "
                f"{ai_score:.4f}"
            )

            print(
                f"AI gain             : "
                f"{gain:+.4f}"
            )

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    print()
    print()
    print("=" * 78)
    print(
        " ROBUSTNESS SUMMARY"
    )
    print("=" * 78)

    print()

    print(
        "Physical  Measurement  "
        "Traditional  Logical-GRU  Gain"
    )

    print(
        "-" * 78
    )

    for result in results:

        print(
            f"{result['physical_noise']:<9.2f}"
            f"{result['measurement_noise']:<12.2f}"
            f"{result['traditional']:<13.4f}"
            f"{result['logical_target_gru']:<13.4f}"
            f"{result['gain']:+.4f}"
        )

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    traditional_scores = [
        result["traditional"]
        for result in results
    ]

    ai_scores = [
        result["logical_target_gru"]
        for result in results
    ]

    gains = [
        result["gain"]
        for result in results
    ]

    mean_traditional = (
        sum(traditional_scores)
        / len(traditional_scores)
    )

    mean_ai = (
        sum(ai_scores)
        / len(ai_scores)
    )

    mean_gain = (
        sum(gains)
        / len(gains)
    )

    ai_wins = sum(
        1
        for gain in gains
        if gain > 0
    )

    ties = sum(
        1
        for gain in gains
        if gain == 0
    )

    losses = sum(
        1
        for gain in gains
        if gain < 0
    )

    # ========================================================
    # OVERALL RESULT
    # ========================================================

    print()
    print("=" * 78)
    print(
        " OVERALL ROBUSTNESS"
    )
    print("=" * 78)

    print()

    print(
        f"Mean traditional logical : "
        f"{mean_traditional:.4f}"
    )

    print(
        f"Mean logical-target GRU  : "
        f"{mean_ai:.4f}"
    )

    print(
        f"Mean AI gain             : "
        f"{mean_gain:+.4f}"
    )

    print()

    print(
        f"AI wins                  : "
        f"{ai_wins}/{len(results)}"
    )

    print(
        f"Ties                     : "
        f"{ties}/{len(results)}"
    )

    print(
        f"AI losses                : "
        f"{losses}/{len(results)}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert len(results) == (
        len(PHYSICAL_NOISE_LEVELS)
        *
        len(MEASUREMENT_NOISE_LEVELS)
    )

    assert all(
        0.0 <= score <= 1.0
        for score in traditional_scores
    )

    assert all(
        0.0 <= score <= 1.0
        for score in ai_scores
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()

    if ai_wins > 0:

        print(
            "Noise robustness experiment : PASS"
        )

    else:

        print(
            "Noise robustness experiment : REVIEW"
        )

    print()

    print(
        "RESULT : SUCCESS"
    )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()