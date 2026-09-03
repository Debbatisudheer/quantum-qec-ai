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

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITY = 0.10

TRAINING_SAMPLES = 5000

TEST_SAMPLES = 1000

HIDDEN_SIZE = 64

LEARNING_RATE = 0.003

EPOCHS = 100

SEED = 42

TEST_SEED = 12345


# ============================================================
# QEC ROUND CONFIGURATIONS
# ============================================================

ROUND_CONFIGURATIONS = [
    1,
    2,
    3,
    5,
    7,
    10
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

        logical 0 -> 000
        logical 1 -> 111
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
    Determine whether the predicted correction
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

    return (
        recovered_logical
        == logical_state
    )


# ============================================================
# GENERATE DATA
# ============================================================

def generate_training_samples(
    rounds,
    seed
):
    """
    Generate training samples for a
    particular number of QEC rounds.
    """

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=seed
        )
    )

    return generator.generate_dataset(
        TRAINING_SAMPLES
    )


def generate_test_samples(
    rounds,
    seed
):
    """
    Generate an independent test dataset.
    """

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=seed
        )
    )

    return generator.generate_dataset(
        TEST_SAMPLES
    )


# ============================================================
# TRAIN LOGICAL-TARGET GRU
# ============================================================

def train_gru(
    rounds,
    training_samples
):
    """
    Train a logical-target GRU for the
    requested number of QEC rounds.
    """

    decoder = LogicalTargetGRUDecoder(
        rounds=rounds,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=SEED
    )

    decoder.train(
        training_samples,
        verbose=False
    )

    return decoder


# ============================================================
# EVALUATE AI DECODER
# ============================================================

def evaluate_ai(
    decoder,
    test_samples,
    recovery
):
    """
    Evaluate logical success of the
    logical-target GRU.
    """

    success_count = 0

    for sample in test_samples:

        logical_state = int(
            sample[
                "logical_state"
            ]
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

        success = (
            calculate_logical_success(
                logical_state,
                actual_error,
                correction,
                recovery
            )
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

    It uses the final observed syndrome.
    """

    decoder = RepeatedLookupDecoder()

    success_count = 0

    for sample in test_samples:

        logical_state = int(
            sample[
                "logical_state"
            ]
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

        success = (
            calculate_logical_success(
                logical_state,
                actual_error,
                correction,
                recovery
            )
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
        "QEC ROUND ROBUSTNESS EXPERIMENT"
    )

    print("=" * 78)

    print()

    print(
        f"Physical noise     : "
        f"{PHYSICAL_ERROR_PROBABILITY:.2f}"
    )

    print(
        f"Measurement noise  : "
        f"{MEASUREMENT_NOISE_PROBABILITY:.2f}"
    )

    print(
        f"Training samples   : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples       : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"GRU hidden size    : "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"GRU epochs         : "
        f"{EPOCHS}"
    )

    print()

    recovery = LogicalRecovery()

    results = []

    # ========================================================
    # TEST EVERY ROUND CONFIGURATION
    # ========================================================

    for rounds in ROUND_CONFIGURATIONS:

        print()
        print(
            "-" * 78
        )

        print(
            f"QEC rounds : {rounds}"
        )

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        training_samples = (
            generate_training_samples(
                rounds,
                SEED
            )
        )

        print(
            f"Training samples : "
            f"{len(training_samples)}"
        )

        # ----------------------------------------------------
        # TRAIN AI
        # ----------------------------------------------------

        ai_decoder = train_gru(
            rounds,
            training_samples
        )

        print(
            "Logical-target GRU : TRAINED"
        )

        # ----------------------------------------------------
        # INDEPENDENT TEST DATA
        # ----------------------------------------------------

        test_samples = (
            generate_test_samples(
                rounds,
                TEST_SEED
            )
        )

        print(
            f"Test samples : "
            f"{len(test_samples)}"
        )

        # ----------------------------------------------------
        # TRADITIONAL DECODER
        # ----------------------------------------------------

        traditional_score = (
            evaluate_traditional(
                test_samples,
                recovery
            )
        )

        # ----------------------------------------------------
        # AI DECODER
        # ----------------------------------------------------

        ai_score = evaluate_ai(
            ai_decoder,
            test_samples,
            recovery
        )

        # ----------------------------------------------------
        # GAIN
        # ----------------------------------------------------

        gain = (
            ai_score
            - traditional_score
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "rounds":
                    rounds,

                "traditional":
                    traditional_score,

                "logical_target_gru":
                    ai_score,

                "gain":
                    gain
            }
        )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

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
    # SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 78)

    print(
        " ROUND ROBUSTNESS SUMMARY"
    )

    print("=" * 78)

    print()

    print(
        "Rounds   Traditional   Logical-GRU   Gain"
    )

    print(
        "-" * 78
    )

    for result in results:

        print(
            f"{result['rounds']:<9}"
            f"{result['traditional']:<14.4f}"
            f"{result['logical_target_gru']:<14.4f}"
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
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 78)

    print(
        " OVERALL ROUND ROBUSTNESS"
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

    expected_results = len(
        ROUND_CONFIGURATIONS
    )

    assert len(results) == (
        expected_results
    )

    assert all(
        0.0 <= score <= 1.0
        for score in traditional_scores
    )

    assert all(
        0.0 <= score <= 1.0
        for score in ai_scores
    )

    assert all(
        len(
            sample[
                "observed_syndrome_history"
            ]
        ) == sample["rounds"]
        for result_rounds in ROUND_CONFIGURATIONS
        for sample in (
            generate_test_samples(
                result_rounds,
                TEST_SEED
            )[:5]
        )
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()

    if ai_wins > 0:

        print(
            "Round robustness experiment : PASS"
        )

    else:

        print(
            "Round robustness experiment : REVIEW"
        )

    print()

    print(
        "RESULT : SUCCESS"
    )

    print()

    print("=" * 78)


if __name__ == "__main__":
    main()