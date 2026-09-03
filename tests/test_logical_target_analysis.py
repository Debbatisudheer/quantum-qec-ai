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
# ALL POSSIBLE 3-QUBIT CORRECTIONS
# ============================================================

CORRECTIONS = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]


# ============================================================
# HELPERS
# ============================================================

def observation_to_key(
    observed_syndrome_history
):
    return "|".join(
        observed_syndrome_history
    )


def xor_states(a, b):
    return [
        int(x) ^ int(y)
        for x, y in zip(a, b)
    ]


def state_to_tuple(state):
    return tuple(
        int(bit)
        for bit in state
    )


def state_to_string(state):
    return "".join(
        str(int(bit))
        for bit in state
    )


def encoded_state_to_list(
    encoded_state
):
    return [
        int(bit)
        for bit in encoded_state
    ]


# ============================================================
# TEST 1
#
# VERIFY THE LOGICAL EFFECT OF ALL POSSIBLE CORRECTIONS
#
# A correction is evaluated by its residual error:
#
#     actual_error XOR correction
#
# Majority(residual_error) == 0
#
# means the logical information is preserved.
# ============================================================

def test_correction_logical_effects():

    print()
    print("=" * 60)
    print(
        " TEST 1: CORRECTION -> LOGICAL EFFECT"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    print()

    print(
        "Correction | Residual majority effect"
    )

    print(
        "-" * 45
    )

    # --------------------------------------------------------
    # For every correction, show which residual errors
    # preserve the logical state.
    # --------------------------------------------------------

    for correction in CORRECTIONS:

        logical_preserving = []

        logical_flipping = []

        for actual_error in CORRECTIONS:

            residual = xor_states(
                actual_error,
                correction
            )

            recovered = recovery.recover(
                residual
            )

            if recovered == 0:

                logical_preserving.append(
                    state_to_string(
                        actual_error
                    )
                )

            else:

                logical_flipping.append(
                    state_to_string(
                        actual_error
                    )
                )

        print()

        print(
            f"Correction "
            f"{state_to_string(correction)}"
        )

        print(
            "  Preserves logical : "
            + ", ".join(
                logical_preserving
            )
        )

        print(
            "  Flips logical     : "
            + ", ".join(
                logical_flipping
            )
        )

    return True


# ============================================================
# BUILD OBSERVATION -> ERROR DISTRIBUTION
# ============================================================

def build_observation_groups(
    samples
):

    groups = defaultdict(Counter)

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        error_state = state_to_tuple(
            sample[
                "final_error_state"
            ]
        )

        groups[
            observation
        ][error_state] += 1

    return groups


# ============================================================
# TEST 2
#
# FOR EVERY OBSERVATION:
#
# Evaluate ALL 8 possible corrections.
#
# This gives:
#
#     P(logical success | observation, correction)
#
# ============================================================

def calculate_correction_probabilities(
    groups
):

    correction_statistics = {}

    for observation, error_counts in (
        groups.items()
    ):

        group_total = sum(
            error_counts.values()
        )

        correction_statistics[
            observation
        ] = {}

        for correction in CORRECTIONS:

            logical_success_count = 0

            exact_error_count = 0

            bit_correct_count = 0

            total_bits = 0

            for actual_error, count in (
                error_counts.items()
            ):

                # ------------------------------------------------
                # Exact physical error prediction
                # ------------------------------------------------

                if correction == actual_error:

                    exact_error_count += count

                # ------------------------------------------------
                # Bit accuracy
                # ------------------------------------------------

                for predicted_bit, actual_bit in zip(
                    correction,
                    actual_error
                ):

                    if predicted_bit == actual_bit:

                        bit_correct_count += count

                    total_bits += count

                # ------------------------------------------------
                # Residual error
                # ------------------------------------------------

                residual_error = xor_states(
                    actual_error,
                    correction
                )

                # ------------------------------------------------
                # Logical success
                #
                # The encoded logical state is either:
                #
                #     000
                #
                # or:
                #
                #     111
                #
                # A residual error preserves the logical
                # state when its majority is 0.
                # ------------------------------------------------

                residual_logical = (
                    LogicalRecovery().recover(
                        residual_error
                    )
                )

                if residual_logical == 0:

                    logical_success_count += count

            logical_probability = (
                logical_success_count
                / group_total
            )

            exact_probability = (
                exact_error_count
                / group_total
            )

            bit_probability = (
                bit_correct_count
                / total_bits
            )

            correction_statistics[
                observation
            ][correction] = {
                "logical_probability":
                    logical_probability,

                "exact_probability":
                    exact_probability,

                "bit_probability":
                    bit_probability
            }

    return correction_statistics


# ============================================================
# TEST 2 OUTPUT
# ============================================================

def test_correction_probability_summary(
    groups,
    correction_statistics
):

    print()
    print("=" * 60)
    print(
        " TEST 2: CORRECTION PROBABILITY SUMMARY"
    )
    print("=" * 60)

    logical_probability_totals = {
        correction: 0.0
        for correction in CORRECTIONS
    }

    exact_probability_totals = {
        correction: 0.0
        for correction in CORRECTIONS
    }

    number_of_observations = len(
        groups
    )

    for observation in groups:

        statistics = (
            correction_statistics[
                observation
            ]
        )

        for correction in CORRECTIONS:

            logical_probability_totals[
                correction
            ] += statistics[
                correction
            ][
                "logical_probability"
            ]

            exact_probability_totals[
                correction
            ] += statistics[
                correction
            ][
                "exact_probability"
            ]

    print()

    print(
        "Average probability across "
        "observations:"
    )

    print()

    for correction in CORRECTIONS:

        average_logical = (
            logical_probability_totals[
                correction
            ]
            / number_of_observations
        )

        average_exact = (
            exact_probability_totals[
                correction
            ]
            / number_of_observations
        )

        print(
            f"{state_to_string(correction)} "
            f"-> "
            f"logical={average_logical:.4f} "
            f"exact={average_exact:.4f}"
        )

    return True


# ============================================================
# TEST 3
#
# BUILD THREE DIFFERENT TARGETS
#
# TARGET A:
#
# Most likely physical error.
#
# TARGET B:
#
# Correction that maximizes logical success.
#
# TARGET C:
#
# Correction that maximizes exact physical prediction.
#
# A and C should normally be identical.
# ============================================================

def build_targets(
    groups,
    correction_statistics
):

    physical_error_target = {}

    logical_target = {}

    exact_target = {}

    for observation, error_counts in (
        groups.items()
    ):

        # ----------------------------------------------------
        # Target A:
        # Most likely physical error.
        # ----------------------------------------------------

        physical_error_target[
            observation
        ] = max(
            error_counts,
            key=error_counts.get
        )

        # ----------------------------------------------------
        # Target B:
        # Maximum logical success.
        # ----------------------------------------------------

        logical_target[
            observation
        ] = max(
            CORRECTIONS,
            key=lambda correction:
                correction_statistics[
                    observation
                ][
                    correction
                ][
                    "logical_probability"
                ]
        )

        # ----------------------------------------------------
        # Target C:
        # Maximum exact physical accuracy.
        # ----------------------------------------------------

        exact_target[
            observation
        ] = max(
            CORRECTIONS,
            key=lambda correction:
                correction_statistics[
                    observation
                ][
                    correction
                ][
                    "exact_probability"
                ]
        )

    return (
        physical_error_target,
        logical_target,
        exact_target
    )


# ============================================================
# TEST 4
#
# HOW OFTEN DOES THE LOGICAL TARGET DIFFER FROM
# THE MOST LIKELY PHYSICAL ERROR?
# ============================================================

def compare_targets(
    physical_error_target,
    logical_target,
    exact_target
):

    print()
    print("=" * 60)
    print(
        " TEST 3: TARGET COMPARISON"
    )
    print("=" * 60)

    total = len(
        physical_error_target
    )

    logical_diff = 0

    exact_diff = 0

    for observation in (
        physical_error_target
    ):

        physical_target = (
            physical_error_target[
                observation
            ]
        )

        logical_correction = (
            logical_target[
                observation
            ]
        )

        exact_correction = (
            exact_target[
                observation
            ]
        )

        if (
            physical_target
            != logical_correction
        ):

            logical_diff += 1

        if (
            physical_target
            != exact_correction
        ):

            exact_diff += 1

    print()

    print(
        f"Observations                 : "
        f"{total}"
    )

    print(
        f"Physical vs logical target   : "
        f"{logical_diff} "
        f"({logical_diff / total:.2%})"
    )

    print(
        f"Physical vs exact target     : "
        f"{exact_diff} "
        f"({exact_diff / total:.2%})"
    )

    print()

    return True


# ============================================================
# TEST 5
#
# GLOBAL PERFORMANCE OF EACH TARGET STRATEGY
# ============================================================

def evaluate_target(
    samples,
    target
):

    recovery = LogicalRecovery()

    exact_error = 0

    bit_correct = 0

    total_bits = 0

    physical_recovery = 0

    logical_success = 0

    total = len(samples)

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        predicted_correction = list(
            target[
                observation
            ]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        encoded_state = (
            encoded_state_to_list(
                sample[
                    "encoded_state"
                ]
            )
        )

        logical_state = int(
            sample[
                "logical_state"
            ]
        )

        # ----------------------------------------------------
        # Exact error accuracy
        # ----------------------------------------------------

        if (
            predicted_correction
            == actual_error
        ):

            exact_error += 1

        # ----------------------------------------------------
        # Bit accuracy
        # ----------------------------------------------------

        for predicted_bit, actual_bit in zip(
            predicted_correction,
            actual_error
        ):

            if predicted_bit == actual_bit:

                bit_correct += 1

            total_bits += 1

        # ----------------------------------------------------
        # Corrupted state
        # ----------------------------------------------------

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        # ----------------------------------------------------
        # Apply correction
        # ----------------------------------------------------

        corrected_state = xor_states(
            corrupted_state,
            predicted_correction
        )

        # ----------------------------------------------------
        # Physical recovery
        # ----------------------------------------------------

        if (
            corrected_state
            == encoded_state
        ):

            physical_recovery += 1

        # ----------------------------------------------------
        # Logical recovery
        # ----------------------------------------------------

        recovered_logical = (
            recovery.recover(
                corrected_state
            )
        )

        if (
            recovered_logical
            == logical_state
        ):

            logical_success += 1

    return {
        "exact_error":
            exact_error / total,

        "bit_accuracy":
            bit_correct / total_bits,

        "physical_recovery":
            physical_recovery / total,

        "logical_success":
            logical_success / total
    }


# ============================================================
# TEST 6
#
# SHOW OBSERVATIONS WHERE THE TWO TARGETS DIFFER
# ============================================================

def show_target_differences(
    groups,
    correction_statistics,
    physical_error_target,
    logical_target,
    maximum_examples=15
):

    print()
    print("=" * 60)
    print(
        " TEST 4: PHYSICAL VS LOGICAL TARGET DIFFERENCES"
    )
    print("=" * 60)

    shown = 0

    for observation in sorted(
        groups
    ):

        physical_target = (
            physical_error_target[
                observation
            ]
        )

        logical_correction = (
            logical_target[
                observation
            ]
        )

        if (
            physical_target
            == logical_correction
        ):

            continue

        print()

        print(
            f"Observation: {observation}"
        )

        print(
            f"  Most likely error: "
            f"{state_to_string(physical_target)}"
        )

        print(
            f"  Logical-optimal correction: "
            f"{state_to_string(logical_correction)}"
        )

        physical_probability = (
            correction_statistics[
                observation
            ][
                physical_target
            ][
                "logical_probability"
            ]
        )

        logical_probability = (
            correction_statistics[
                observation
            ][
                logical_correction
            ][
                "logical_probability"
            ]
        )

        print(
            f"  Logical success using "
            f"physical target: "
            f"{physical_probability:.4f}"
        )

        print(
            f"  Logical success using "
            f"logical target: "
            f"{logical_probability:.4f}"
        )

        shown += 1

        if shown >= maximum_examples:

            break

    if shown == 0:

        print()

        print(
            "No target differences found."
        )

    return True


# ============================================================
# TEST 7
#
# GLOBAL ERROR-PATTERN DISTRIBUTION
# ============================================================

def show_error_distribution(
    samples
):

    print()
    print("=" * 60)
    print(
        " TEST 5: FINAL ERROR DISTRIBUTION"
    )
    print("=" * 60)

    counts = Counter()

    for sample in samples:

        error = state_to_string(
            sample[
                "final_error_state"
            ]
        )

        counts[
            error
        ] += 1

    print()

    for error in sorted(
        counts
    ):

        count = counts[
            error
        ]

        print(
            f"{error}: "
            f"{count:5d} "
            f"({count / len(samples):.2%})"
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " LOGICAL TARGET ANALYSIS"
    )
    print("=" * 60)

    print()

    print(
        f"Rounds                    : "
        f"{ROUNDS}"
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
    # TEST 1
    # --------------------------------------------------------

    mapping_passed = (
        test_correction_logical_effects()
    )

    if not mapping_passed:

        print()

        print(
            "RESULT : FAIL"
        )

        return

    # --------------------------------------------------------
    # Generate common dataset.
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
    # Show distribution.
    # --------------------------------------------------------

    show_error_distribution(
        samples
    )

    # --------------------------------------------------------
    # Build observation groups.
    # --------------------------------------------------------

    groups = (
        build_observation_groups(
            samples
        )
    )

    print()

    print(
        f"Unique observations       : "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Calculate probability of every correction
    # for every observation.
    # --------------------------------------------------------

    correction_statistics = (
        calculate_correction_probabilities(
            groups
        )
    )

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    test_correction_probability_summary(
        groups,
        correction_statistics
    )

    # --------------------------------------------------------
    # Build the three targets.
    # --------------------------------------------------------

    (
        physical_error_target,
        logical_target,
        exact_target
    ) = build_targets(
        groups,
        correction_statistics
    )

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    compare_targets(
        physical_error_target,
        logical_target,
        exact_target
    )

    # --------------------------------------------------------
    # Evaluate physical-error target.
    # --------------------------------------------------------

    physical_results = evaluate_target(
        samples,
        physical_error_target
    )

    # --------------------------------------------------------
    # Evaluate logical target.
    # --------------------------------------------------------

    logical_results = evaluate_target(
        samples,
        logical_target
    )

    # --------------------------------------------------------
    # Evaluate exact target.
    # --------------------------------------------------------

    exact_results = evaluate_target(
        samples,
        exact_target
    )

    # --------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------

    show_target_differences(
        groups,
        correction_statistics,
        physical_error_target,
        logical_target
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " FINAL TARGET COMPARISON"
    )
    print("=" * 60)

    print()

    print(
        "Most-likely physical-error target:"
    )

    print(
        f"  Exact error      : "
        f"{physical_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{physical_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{physical_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{physical_results['logical_success']:.4f}"
    )

    print()

    print(
        "Logical-objective target:"
    )

    print(
        f"  Exact error      : "
        f"{logical_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{logical_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{logical_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{logical_results['logical_success']:.4f}"
    )

    print()

    print(
        "Exact-error target:"
    )

    print(
        f"  Exact error      : "
        f"{exact_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{exact_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{exact_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{exact_results['logical_success']:.4f}"
    )

    print()

    gain = (
        logical_results[
            "logical_success"
        ]
        - physical_results[
            "logical_success"
        ]
    )

    print(
        f"Logical-target gain: "
        f"{gain:+.4f}"
    )

    print()

    if gain > 0.01:

        print(
            "CONCLUSION:"
        )

        print(
            "The logical objective produces a "
            "meaningfully different and better "
            "correction target."
        )

        print()

        print(
            "The future AI decoder should be "
            "trained/evaluated around logical "
            "QEC success rather than only "
            "exact physical-error prediction."
        )

    else:

        print(
            "CONCLUSION:"
        )

        print(
            "The physical-error and logical-objective "
            "targets behave similarly."
        )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " LOGICAL TARGET ANALYSIS : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()