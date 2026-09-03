from collections import Counter, defaultdict

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
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

def observation_to_key(observed_syndrome_history):
    """
    Convert the complete observed syndrome history
    into one hashable representation.

    Example:

        ['11', '10', '01', '01', '01']

    becomes:

        '11|10|01|01|01'
    """

    return "|".join(
        observed_syndrome_history
    )


def calculate_logical_state_accuracy(
    observation_groups
):
    """
    For every observation, predict the most frequent
    logical state.

    This gives the empirical best possible deterministic
    logical prediction for the observed data.
    """

    correct = 0

    total = 0

    for logical_counts in observation_groups.values():

        group_total = sum(
            logical_counts.values()
        )

        best_count = max(
            logical_counts.values()
        )

        correct += best_count

        total += group_total

    if total == 0:
        return 0.0

    return correct / total


# ============================================================
# TEST 1
#
# LOGICAL STATE DISTRIBUTION
# ============================================================

def test_logical_distribution(samples):

    print()
    print("=" * 60)
    print(
        " TEST 1: LOGICAL STATE DISTRIBUTION"
    )
    print("=" * 60)

    counts = Counter()

    for sample in samples:

        logical_state = int(
            sample["logical_state"]
        )

        counts[logical_state] += 1

    for logical_state in [0, 1]:

        count = counts[logical_state]

        print(
            f"Logical {logical_state}: "
            f"{count:5d} "
            f"({count / len(samples):.2%})"
        )

    passed = (
        counts[0] > 0
        and counts[1] > 0
    )

    print()

    print(
        "Logical states represented : "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    return passed


# ============================================================
# TEST 2
#
# OBSERVATION -> LOGICAL STATE
#
# For each observed syndrome history:
#
#     P(logical 0 | observation)
#     P(logical 1 | observation)
#
# is calculated.
# ============================================================

def test_observation_logical_information(samples):

    print()
    print("=" * 60)
    print(
        " TEST 2: OBSERVATION -> LOGICAL STATE"
    )
    print("=" * 60)

    observation_groups = defaultdict(Counter)

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        logical_state = int(
            sample["logical_state"]
        )

        observation_groups[
            observation
        ][logical_state] += 1

    total_observations = len(
        observation_groups
    )

    ambiguous_observations = 0

    samples_in_ambiguous_groups = 0

    maximum_logical_probability = 0.0

    for logical_counts in (
        observation_groups.values()
    ):

        number_of_logical_states = len(
            logical_counts
        )

        group_total = sum(
            logical_counts.values()
        )

        best_count = max(
            logical_counts.values()
        )

        best_probability = (
            best_count / group_total
        )

        maximum_logical_probability = max(
            maximum_logical_probability,
            best_probability
        )

        if number_of_logical_states > 1:

            ambiguous_observations += 1

            samples_in_ambiguous_groups += (
                group_total
            )

    ceiling = (
        calculate_logical_state_accuracy(
            observation_groups
        )
    )

    print()
    print(
        f"Unique observations       : "
        f"{total_observations}"
    )

    print(
        f"Ambiguous observations    : "
        f"{ambiguous_observations}"
    )

    print(
        f"Samples in ambiguous groups: "
        f"{samples_in_ambiguous_groups}"
    )

    print(
        f"Maximum P(logical|obs)     : "
        f"{maximum_logical_probability:.4f}"
    )

    print(
        f"Logical information ceiling: "
        f"{ceiling:.4f}"
    )

    return observation_groups, ceiling


# ============================================================
# TEST 3
#
# FINAL SYNDROME ONLY
#
# Compare:
#
#     complete temporal history
#
# against:
#
#     final syndrome only
#
# This tells us whether temporal information is actually
# useful for logical-state inference.
# ============================================================

def test_final_syndrome_ceiling(samples):

    print()
    print("=" * 60)
    print(
        " TEST 3: FINAL SYNDROME CEILING"
    )
    print("=" * 60)

    observation_groups = defaultdict(Counter)

    for sample in samples:

        final_syndrome = (
            sample[
                "final_observed_syndrome"
            ]
        )

        logical_state = int(
            sample["logical_state"]
        )

        observation_groups[
            final_syndrome
        ][logical_state] += 1

    ceiling = (
        calculate_logical_state_accuracy(
            observation_groups
        )
    )

    print()

    print(
        f"Unique final syndromes : "
        f"{len(observation_groups)}"
    )

    print(
        f"Final-syndrome logical ceiling: "
        f"{ceiling:.4f}"
    )

    print()

    print(
        "Final syndrome probabilities:"
    )

    for syndrome in sorted(
        observation_groups
    ):

        counts = observation_groups[
            syndrome
        ]

        total = sum(
            counts.values()
        )

        probability_0 = (
            counts[0] / total
        )

        probability_1 = (
            counts[1] / total
        )

        print(
            f"{syndrome}: "
            f"P(0)={probability_0:.4f} "
            f"P(1)={probability_1:.4f} "
            f"n={total}"
        )

    return ceiling


# ============================================================
# TEST 4
#
# BEST LOGICAL PREDICTION
#
# We explicitly construct the empirical Bayes-style
# prediction:
#
#     observation -> most frequent logical state
#
# This is NOT an AI model.
#
# It is an information diagnostic.
# ============================================================

def test_logical_bayes_prediction(
    samples
):

    print()
    print("=" * 60)
    print(
        " TEST 4: EMPIRICAL LOGICAL BAYES PREDICTION"
    )
    print("=" * 60)

    observation_groups = defaultdict(Counter)

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        logical_state = int(
            sample["logical_state"]
        )

        observation_groups[
            observation
        ][logical_state] += 1

    correct = 0

    total = 0

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

        actual_logical = int(
            sample["logical_state"]
        )

        counts = observation_groups[
            observation
        ]

        group_total = sum(
            counts.values()
        )

        predicted_logical = max(
            counts,
            key=counts.get
        )

        confidence = (
            counts[predicted_logical]
            / group_total
        )

        prediction_distribution[
            predicted_logical
        ] += 1

        if predicted_logical == actual_logical:

            correct += 1

        total += 1

        bucket = int(
            confidence * 10
        ) / 10

        if bucket >= 1.0:
            bucket = 0.9

        confidence_buckets[
            bucket
        ]["total"] += 1

        if predicted_logical == actual_logical:

            confidence_buckets[
                bucket
            ]["correct"] += 1

    accuracy = correct / total

    print()

    print(
        f"Logical prediction accuracy : "
        f"{accuracy:.4f}"
    )

    print()

    print(
        "Prediction distribution:"
    )

    for logical_state in [0, 1]:

        count = prediction_distribution[
            logical_state
        ]

        print(
            f"Logical {logical_state}: "
            f"{count:5d} "
            f"({count / total:.2%})"
        )

    print()

    print(
        "Confidence -> accuracy:"
    )

    for bucket in sorted(
        confidence_buckets
    ):

        bucket_data = (
            confidence_buckets[bucket]
        )

        bucket_total = (
            bucket_data["total"]
        )

        bucket_correct = (
            bucket_data["correct"]
        )

        if bucket_total == 0:
            continue

        bucket_accuracy = (
            bucket_correct
            / bucket_total
        )

        print(
            f"{bucket:.1f}+ "
            f"samples={bucket_total:5d} "
            f"accuracy={bucket_accuracy:.4f}"
        )

    return accuracy


# ============================================================
# TEST 5
#
# SAMPLE SOME AMBIGUOUS OBSERVATIONS
#
# This lets us directly see cases where exactly the
# same syndrome history occurs for BOTH logical states.
# ============================================================

def show_ambiguous_examples(
    samples,
    observation_groups,
    maximum_examples=10
):

    print()
    print("=" * 60)
    print(
        " TEST 5: AMBIGUOUS OBSERVATION EXAMPLES"
    )
    print("=" * 60)

    shown = 0

    for observation in sorted(
        observation_groups
    ):

        counts = observation_groups[
            observation
        ]

        if len(counts) < 2:
            continue

        total = sum(
            counts.values()
        )

        probability_0 = (
            counts[0] / total
        )

        probability_1 = (
            counts[1] / total
        )

        print()

        print(
            f"Observation: {observation}"
        )

        print(
            f"  Logical 0: "
            f"{counts[0]} "
            f"({probability_0:.2%})"
        )

        print(
            f"  Logical 1: "
            f"{counts[1]} "
            f"({probability_1:.2%})"
        )

        shown += 1

        if shown >= maximum_examples:
            break

    if shown == 0:

        print()
        print(
            "No ambiguous observations found."
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " LOGICAL INFORMATION DIAGNOSTIC"
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

    test_logical_distribution(
        samples
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    (
        observation_groups,
        temporal_ceiling
    ) = test_observation_logical_information(
        samples
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    final_syndrome_ceiling = (
        test_final_syndrome_ceiling(
            samples
        )
    )

    # --------------------------------------------------------
    # Test 4
    # --------------------------------------------------------

    bayes_accuracy = (
        test_logical_bayes_prediction(
            samples
        )
    )

    # --------------------------------------------------------
    # Test 5
    # --------------------------------------------------------

    show_ambiguous_examples(
        samples,
        observation_groups
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " FINAL COMPARISON"
    )
    print("=" * 60)

    print()

    print(
        f"Final syndrome ceiling : "
        f"{final_syndrome_ceiling:.4f}"
    )

    print(
        f"Temporal history ceiling: "
        f"{temporal_ceiling:.4f}"
    )

    print(
        f"Empirical Bayes accuracy : "
        f"{bayes_accuracy:.4f}"
    )

    print()

    temporal_gain = (
        temporal_ceiling
        - final_syndrome_ceiling
    )

    print(
        f"Temporal information gain: "
        f"{temporal_gain:+.4f}"
    )

    print()

    print(
        "INTERPRETATION"
    )

    if temporal_gain > 0.01:

        print(
            "Temporal syndrome history contains "
            "useful logical information beyond "
            "the final syndrome."
        )

    else:

        print(
            "Temporal history provides little "
            "additional logical information."
        )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " LOGICAL INFORMATION DIAGNOSTIC : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()