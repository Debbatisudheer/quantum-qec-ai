from collections import Counter, defaultdict

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
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

SAMPLES = 25000

SEED = 42


# ============================================================
# HELPERS
# ============================================================

def observation_to_key(
    observed_syndrome_history
):
    """
    Convert the complete observed syndrome history
    into a hashable key.

    Example:

        ['11', '10', '01', '01', '01']

    becomes:

        11|10|01|01|01
    """

    return "|".join(
        observed_syndrome_history
    )


def xor_states(a, b):
    """
    XOR two 3-qubit states.
    """

    return [
        int(x) ^ int(y)
        for x, y in zip(a, b)
    ]


def encoded_state_to_list(
    encoded_state
):
    """
    Convert:

        '000' -> [0,0,0]
        '111' -> [1,1,1]
    """

    return [
        int(bit)
        for bit in encoded_state
    ]


def state_to_string(state):
    return "".join(
        str(int(bit))
        for bit in state
    )


# ============================================================
# BUILD OBSERVATION -> ERROR DISTRIBUTION
# ============================================================

def build_error_observation_groups(
    samples
):
    """
    Build:

        observed syndrome history
                    ↓
             final error pattern

    For every observation we store how frequently
    each final error pattern occurred.
    """

    groups = defaultdict(Counter)

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        error_state = tuple(
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        )

        groups[
            observation
        ][error_state] += 1

    return groups


# ============================================================
# TEST 1
#
# ERROR INFORMATION CEILING
# ============================================================

def test_error_information_ceiling(
    samples
):

    print()
    print("=" * 60)
    print(
        " TEST 1: FINAL ERROR INFORMATION CEILING"
    )
    print("=" * 60)

    groups = (
        build_error_observation_groups(
            samples
        )
    )

    total_samples = 0

    correct_predictions = 0

    ambiguous_observations = 0

    ambiguous_samples = 0

    maximum_targets = 0

    for error_counts in groups.values():

        group_total = sum(
            error_counts.values()
        )

        total_samples += group_total

        number_of_targets = len(
            error_counts
        )

        maximum_targets = max(
            maximum_targets,
            number_of_targets
        )

        if number_of_targets > 1:

            ambiguous_observations += 1

            ambiguous_samples += (
                group_total
            )

        best_count = max(
            error_counts.values()
        )

        correct_predictions += (
            best_count
        )

    ceiling = (
        correct_predictions
        / total_samples
    )

    print()

    print(
        f"Unique observations       : "
        f"{len(groups)}"
    )

    print(
        f"Ambiguous observations    : "
        f"{ambiguous_observations}"
    )

    print(
        f"Samples in ambiguous groups: "
        f"{ambiguous_samples}"
    )

    print(
        f"Maximum error patterns/observation: "
        f"{maximum_targets}"
    )

    print()

    print(
        f"Exact error-pattern ceiling: "
        f"{ceiling:.4f}"
    )

    return groups, ceiling


# ============================================================
# TEST 2
#
# EMPIRICAL BAYES ERROR DECODER
#
# For every syndrome-history observation:
#
#     choose the most frequent final error.
#
# This is NOT an AI model.
#
# It is the best deterministic prediction available
# from the empirical observation groups.
# ============================================================

def test_empirical_error_decoder(
    samples,
    groups
):

    print()
    print("=" * 60)
    print(
        " TEST 2: EMPIRICAL ERROR DECODER"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    exact_error_correct = 0

    bit_correct = 0

    total_bits = 0

    physical_recovery = 0

    logical_success = 0

    prediction_distribution = Counter()

    confidence_buckets = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        logical_state = int(
            sample["logical_state"]
        )

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        error_counts = groups[
            observation
        ]

        # ----------------------------------------------------
        # BEST EMPIRICAL ERROR PREDICTION
        # ----------------------------------------------------

        predicted_error_tuple = max(
            error_counts,
            key=error_counts.get
        )

        predicted_error = list(
            predicted_error_tuple
        )

        prediction_distribution[
            state_to_string(
                predicted_error
            )
        ] += 1

        # ----------------------------------------------------
        # ERROR-PATTERN EXACT ACCURACY
        # ----------------------------------------------------

        if predicted_error == actual_error:

            exact_error_correct += 1

        # ----------------------------------------------------
        # BIT ACCURACY
        # ----------------------------------------------------

        for predicted_bit, actual_bit in zip(
            predicted_error,
            actual_error
        ):

            if predicted_bit == actual_bit:

                bit_correct += 1

            total_bits += 1

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        group_total = sum(
            error_counts.values()
        )

        confidence = (
            error_counts[
                predicted_error_tuple
            ]
            / group_total
        )

        bucket = int(
            confidence * 10
        ) / 10

        if bucket >= 1.0:
            bucket = 0.9

        confidence_buckets[
            bucket
        ]["total"] += 1

        if predicted_error == actual_error:

            confidence_buckets[
                bucket
            ]["correct"] += 1

        # ----------------------------------------------------
        # CORRUPTED PHYSICAL STATE
        # ----------------------------------------------------

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        # ----------------------------------------------------
        # APPLY PREDICTED CORRECTION
        # ----------------------------------------------------

        corrected_state = xor_states(
            corrupted_state,
            predicted_error
        )

        # ----------------------------------------------------
        # EXACT PHYSICAL RECOVERY
        # ----------------------------------------------------

        if corrected_state == encoded_state:

            physical_recovery += 1

        # ----------------------------------------------------
        # LOGICAL RECOVERY
        # ----------------------------------------------------

        recovered_logical = recovery.recover(
            corrected_state
        )

        if recovered_logical == logical_state:

            logical_success += 1

    exact_accuracy = (
        exact_error_correct
        / len(samples)
    )

    bit_accuracy = (
        bit_correct
        / total_bits
    )

    physical_accuracy = (
        physical_recovery
        / len(samples)
    )

    logical_accuracy = (
        logical_success
        / len(samples)
    )

    print()

    print(
        f"Exact error accuracy : "
        f"{exact_accuracy:.4f}"
    )

    print(
        f"Bit error accuracy   : "
        f"{bit_accuracy:.4f}"
    )

    print(
        f"Physical recovery    : "
        f"{physical_accuracy:.4f}"
    )

    print(
        f"Logical success      : "
        f"{logical_accuracy:.4f}"
    )

    print()

    print(
        "Prediction distribution:"
    )

    for error_state in sorted(
        prediction_distribution
    ):

        count = prediction_distribution[
            error_state
        ]

        print(
            f"{error_state}: "
            f"{count:5d} "
            f"({count / len(samples):.2%})"
        )

    print()

    print(
        "Confidence -> error accuracy:"
    )

    for bucket in sorted(
        confidence_buckets
    ):

        data = confidence_buckets[
            bucket
        ]

        if data["total"] == 0:
            continue

        accuracy = (
            data["correct"]
            / data["total"]
        )

        print(
            f"{bucket:.1f}+ "
            f"samples={data['total']:5d} "
            f"accuracy={accuracy:.4f}"
        )

    return {
        "exact_accuracy":
            exact_accuracy,

        "bit_accuracy":
            bit_accuracy,

        "physical_accuracy":
            physical_accuracy,

        "logical_accuracy":
            logical_accuracy
    }


# ============================================================
# TEST 3
#
# FINAL SYNDROME ONLY
#
# Compare the full temporal history against only
# the final observed syndrome.
# ============================================================

def test_final_syndrome_error_ceiling(
    samples
):

    print()
    print("=" * 60)
    print(
        " TEST 3: FINAL SYNDROME ERROR CEILING"
    )
    print("=" * 60)

    groups = defaultdict(Counter)

    for sample in samples:

        syndrome = (
            sample[
                "final_observed_syndrome"
            ]
        )

        error_state = tuple(
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        )

        groups[
            syndrome
        ][error_state] += 1

    correct = 0

    total = 0

    for error_counts in groups.values():

        total += sum(
            error_counts.values()
        )

        correct += max(
            error_counts.values()
        )

    ceiling = correct / total

    print()

    print(
        f"Unique final syndromes : "
        f"{len(groups)}"
    )

    print(
        f"Final syndrome error ceiling: "
        f"{ceiling:.4f}"
    )

    print()

    for syndrome in sorted(
        groups
    ):

        counts = groups[
            syndrome
        ]

        total_count = sum(
            counts.values()
        )

        best_error = max(
            counts,
            key=counts.get
        )

        best_probability = (
            counts[best_error]
            / total_count
        )

        print(
            f"{syndrome}: "
            f"best={state_to_string(best_error)} "
            f"P={best_probability:.4f} "
            f"n={total_count}"
        )

    return ceiling


# ============================================================
# TEST 4
#
# LOGICAL SUCCESS OF THE INFORMATION CEILING
#
# This is the MOST IMPORTANT TEST.
#
# We don't care only about exact physical-error
# prediction.
#
# We care about whether the best empirical correction
# preserves the logical state.
# ============================================================

def test_logical_ceiling(
    samples,
    groups
):

    print()
    print("=" * 60)
    print(
        " TEST 4: LOGICAL SUCCESS OF INFORMATION CEILING"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    logical_success = 0

    physical_recovery = 0

    exact_error_prediction = 0

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        logical_state = int(
            sample["logical_state"]
        )

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        error_counts = groups[
            observation
        ]

        predicted_error = list(
            max(
                error_counts,
                key=error_counts.get
            )
        )

        if predicted_error == actual_error:

            exact_error_prediction += 1

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        corrected_state = xor_states(
            corrupted_state,
            predicted_error
        )

        if corrected_state == encoded_state:

            physical_recovery += 1

        recovered_logical = recovery.recover(
            corrected_state
        )

        if recovered_logical == logical_state:

            logical_success += 1

    exact_error_accuracy = (
        exact_error_prediction
        / len(samples)
    )

    physical_accuracy = (
        physical_recovery
        / len(samples)
    )

    logical_accuracy = (
        logical_success
        / len(samples)
    )

    print()

    print(
        f"Exact error prediction : "
        f"{exact_error_accuracy:.4f}"
    )

    print(
        f"Physical recovery      : "
        f"{physical_accuracy:.4f}"
    )

    print(
        f"Logical success        : "
        f"{logical_accuracy:.4f}"
    )

    print()

    return logical_accuracy


# ============================================================
# TEST 5
#
# COMPARE AGAINST PREVIOUS AI RESULTS
# ============================================================

def compare_with_previous_ai(
    logical_ceiling
):

    print()
    print("=" * 60)
    print(
        " TEST 5: INFORMATION CEILING VS PREVIOUS AI"
    )
    print("=" * 60)

    # Previous best GRU logical success from our
    # earlier experiment.

    previous_gru_logical = 0.5066

    # Previous best hybrid logical success.

    previous_hybrid_logical = 0.5114

    print()

    print(
        f"Empirical logical ceiling : "
        f"{logical_ceiling:.4f}"
    )

    print(
        f"Previous GRU logical      : "
        f"{previous_gru_logical:.4f}"
    )

    print(
        f"Previous Hybrid logical   : "
        f"{previous_hybrid_logical:.4f}"
    )

    print()

    gru_gap = (
        logical_ceiling
        - previous_gru_logical
    )

    hybrid_gap = (
        logical_ceiling
        - previous_hybrid_logical
    )

    print(
        f"GRU gap to ceiling        : "
        f"{gru_gap:+.4f}"
    )

    print(
        f"Hybrid gap to ceiling     : "
        f"{hybrid_gap:+.4f}"
    )

    print()

    if logical_ceiling < 0.55:

        print(
            "The observable data provides "
            "limited logical-decoding information."
        )

    elif logical_ceiling < 0.70:

        print(
            "The observable data contains "
            "moderate logical-decoding information."
        )

    else:

        print(
            "The observable data contains "
            "strong logical-decoding information."
        )

    print()

    if gru_gap < 0.03:

        print(
            "GRU is already close to the "
            "empirical logical ceiling."
        )

    else:

        print(
            "There is meaningful room between "
            "the GRU and the empirical ceiling."
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " ERROR DECODING INFORMATION CEILING"
    )
    print("=" * 60)

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
        f"Samples                   : "
        f"{SAMPLES}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # --------------------------------------------------------
    # Generate one common dataset.
    # --------------------------------------------------------

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

    print()

    print(
        f"Generated samples         : "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    (
        groups,
        error_ceiling
    ) = test_error_information_ceiling(
        samples
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    decoder_results = (
        test_empirical_error_decoder(
            samples,
            groups
        )
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    final_syndrome_ceiling = (
        test_final_syndrome_error_ceiling(
            samples
        )
    )

    # --------------------------------------------------------
    # Test 4
    # --------------------------------------------------------

    logical_ceiling = (
        test_logical_ceiling(
            samples,
            groups
        )
    )

    # --------------------------------------------------------
    # Test 5
    # --------------------------------------------------------

    compare_with_previous_ai(
        logical_ceiling
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " FINAL SUMMARY"
    )
    print("=" * 60)

    print()

    print(
        f"Final syndrome error ceiling : "
        f"{final_syndrome_ceiling:.4f}"
    )

    print(
        f"Temporal error ceiling       : "
        f"{error_ceiling:.4f}"
    )

    print(
        f"Empirical error decoder      : "
        f"{decoder_results['exact_accuracy']:.4f}"
    )

    print(
        f"Empirical logical ceiling    : "
        f"{logical_ceiling:.4f}"
    )

    print(
        f"Previous GRU logical         : "
        f"0.5066"
    )

    print(
        f"Previous Hybrid logical      : "
        f"0.5114"
    )

    print()

    temporal_gain = (
        error_ceiling
        - final_syndrome_ceiling
    )

    print(
        f"Temporal error information gain: "
        f"{temporal_gain:+.4f}"
    )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " ERROR DECODING INFORMATION CEILING : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()