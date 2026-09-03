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
# 3-QUBIT BIT-FLIP CODE
#
# Error patterns sharing the same syndrome:
#
#     00 -> 000 / 111
#     01 -> 001 / 110
#     10 -> 100 / 011
#     11 -> 010 / 101
#
# Within each pair, one representative is the
# low-weight / logical-preserving error and the
# other differs by logical X.
# ============================================================

SYNDROME_ERROR_PAIRS = {
    "00": [
        (0, 0, 0),
        (1, 1, 1),
    ],
    "01": [
        (0, 0, 1),
        (1, 1, 0),
    ],
    "10": [
        (1, 0, 0),
        (0, 1, 1),
    ],
    "11": [
        (0, 1, 0),
        (1, 0, 1),
    ],
}


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
# VERIFY THE FOUR SYNDROME ERROR PAIRS
# ============================================================

def test_syndrome_error_pairs():

    print()
    print("=" * 60)
    print(
        " TEST 1: SYNDROME -> ERROR PAIRS"
    )
    print("=" * 60)

    expected = {
        "00": {
            (0, 0, 0),
            (1, 1, 1),
        },
        "01": {
            (0, 0, 1),
            (1, 1, 0),
        },
        "10": {
            (1, 0, 0),
            (0, 1, 1),
        },
        "11": {
            (0, 1, 0),
            (1, 0, 1),
        },
    }

    passed = True

    for syndrome in sorted(
        SYNDROME_ERROR_PAIRS
    ):

        actual = set(
            SYNDROME_ERROR_PAIRS[
                syndrome
            ]
        )

        print()
        print(
            f"Syndrome {syndrome}:"
        )

        for error in actual:

            q0, q1, q2 = error

            calculated = (
                f"{q0 ^ q1}{q1 ^ q2}"
            )

            print(
                f"  {state_to_string(error)} "
                f"-> syndrome {calculated}"
            )

            if calculated != syndrome:
                passed = False

        if actual != expected[syndrome]:
            passed = False

    print()

    print(
        "Syndrome/error mapping : "
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
# TEST 3
#
# FOR EACH OBSERVATION:
#
#     P(error pattern | observation)
#
# AND:
#
#     Which of the four syndrome pairs
#     does the observation favor?
#
# We calculate the probability that the actual
# error belongs to each error pattern.
# ============================================================

def test_observation_error_probabilities(
    samples,
    groups
):

    print()
    print("=" * 60)
    print(
        " TEST 2: OBSERVATION -> ERROR PROBABILITIES"
    )
    print("=" * 60)

    total = 0

    ambiguous = 0

    pair_probability_groups = defaultdict(
        Counter
    )

    for observation, error_counts in (
        groups.items()
    ):

        group_total = sum(
            error_counts.values()
        )

        total += group_total

        if len(error_counts) > 1:
            ambiguous += 1

        for error_state, count in (
            error_counts.items()
        ):

            syndrome = (
                f"{error_state[0] ^ error_state[1]}"
                f"{error_state[1] ^ error_state[2]}"
            )

            pair_probability_groups[
                syndrome
            ][error_state] += count

    print()

    print(
        f"Unique observations    : "
        f"{len(groups)}"
    )

    print(
        f"Ambiguous observations : "
        f"{ambiguous}"
    )

    print()

    print(
        "Global error probabilities "
        "inside syndrome pairs:"
    )

    for syndrome in sorted(
        pair_probability_groups
    ):

        counts = pair_probability_groups[
            syndrome
        ]

        pair_total = sum(
            counts.values()
        )

        print()

        print(
            f"Syndrome {syndrome}"
        )

        for error_state in sorted(
            counts
        ):

            probability = (
                counts[error_state]
                / pair_total
            )

            print(
                f"  "
                f"{state_to_string(error_state)} "
                f"-> "
                f"{probability:.4f}"
            )

    return True


# ============================================================
# TEST 4
#
# TEMPORAL EMPIRICAL ERROR DECODER
#
# This chooses the most frequent error pattern for
# each COMPLETE observed syndrome history.
# ============================================================

def build_temporal_decoder(
    groups
):

    decoder = {}

    for observation, counts in (
        groups.items()
    ):

        decoder[
            observation
        ] = max(
            counts,
            key=counts.get
        )

    return decoder


# ============================================================
# TEST 5
#
# LOGICAL EQUIVALENCE DECODER
#
# Instead of requiring exact physical-error prediction,
# evaluate whether the prediction preserves the logical
# state.
# ============================================================

def evaluate_decoder(
    samples,
    decoder
):

    recovery = LogicalRecovery()

    exact_error = 0

    physical_recovery = 0

    logical_success = 0

    total_bits = 0

    correct_bits = 0

    for sample in samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        predicted_error = list(
            decoder[observation]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        logical_state = int(
            sample["logical_state"]
        )

        # ----------------------------------------------------
        # Exact error prediction
        # ----------------------------------------------------

        if predicted_error == actual_error:

            exact_error += 1

        # ----------------------------------------------------
        # Bit accuracy
        # ----------------------------------------------------

        for predicted_bit, actual_bit in zip(
            predicted_error,
            actual_error
        ):

            if predicted_bit == actual_bit:

                correct_bits += 1

            total_bits += 1

        # ----------------------------------------------------
        # Actual corrupted physical state
        # ----------------------------------------------------

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        # ----------------------------------------------------
        # Apply predicted correction
        # ----------------------------------------------------

        corrected_state = xor_states(
            corrupted_state,
            predicted_error
        )

        # ----------------------------------------------------
        # Exact physical recovery
        # ----------------------------------------------------

        if corrected_state == encoded_state:

            physical_recovery += 1

        # ----------------------------------------------------
        # Logical recovery
        # ----------------------------------------------------

        recovered_logical = (
            recovery.recover(
                corrected_state
            )
        )

        if recovered_logical == logical_state:

            logical_success += 1

    total = len(samples)

    return {
        "exact_error":
            exact_error / total,

        "bit_accuracy":
            correct_bits / total_bits,

        "physical_recovery":
            physical_recovery / total,

        "logical_success":
            logical_success / total
    }


# ============================================================
# TEST 6
#
# FINAL SYNDROME LOOKUP
#
# Traditional decoder using ONLY the final observed
# syndrome.
# ============================================================

def build_final_syndrome_decoder():

    return {
        "00": (0, 0, 0),
        "01": (0, 0, 1),
        "10": (1, 0, 0),
        "11": (0, 1, 0),
    }


def evaluate_final_syndrome_decoder(
    samples
):

    decoder = (
        build_final_syndrome_decoder()
    )

    recovery = LogicalRecovery()

    exact_error = 0

    physical_recovery = 0

    logical_success = 0

    total = len(samples)

    for sample in samples:

        syndrome = (
            sample[
                "final_observed_syndrome"
            ]
        )

        predicted_error = list(
            decoder[syndrome]
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        logical_state = int(
            sample["logical_state"]
        )

        if predicted_error == actual_error:

            exact_error += 1

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

        recovered_logical = (
            recovery.recover(
                corrected_state
            )
        )

        if recovered_logical == logical_state:

            logical_success += 1

    return {
        "exact_error":
            exact_error / total,

        "physical_recovery":
            physical_recovery / total,

        "logical_success":
            logical_success / total
    }


# ============================================================
# TEST 7
#
# DIRECT LOGICAL-EQUIVALENCE ANALYSIS
#
# For every observation, choose the error pattern that
# produces the highest empirical logical success.
#
# Since logical_state is independent of the error,
# we evaluate the physical correction itself:
#
#     actual error
#          XOR
#     predicted correction
#          ↓
#     residual error
#
# Then majority recovery determines whether the
# logical state survives.
# ============================================================

def test_best_logical_correction_per_observation(
    samples,
    groups
):

    print()
    print("=" * 60)
    print(
        " TEST 3: BEST LOGICAL CORRECTION PER OBSERVATION"
    )
    print("=" * 60)

    recovery = LogicalRecovery()

    # Candidate corrections are all eight possible
    # 3-bit physical corrections.

    candidate_corrections = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]

    logical_decoder = {}

    total_observations = len(groups)

    for observation, error_counts in (
        groups.items()
    ):

        # ----------------------------------------------------
        # Evaluate every possible correction using the
        # empirical error distribution for this observation.
        # ----------------------------------------------------

        best_correction = None

        best_success = -1.0

        for correction in candidate_corrections:

            success_count = 0

            group_total = sum(
                error_counts.values()
            )

            for actual_error, count in (
                error_counts.items()
            ):

                # Residual error after correction.
                residual_error = xor_states(
                    actual_error,
                    correction
                )

                # The logical state is preserved when
                # majority of the residual error is zero.
                #
                # For logical 0:
                #   encoded = 000
                #   residual must have majority 0
                #
                # For logical 1:
                #   encoded = 111
                #   residual must have majority 0
                #
                # Therefore the logical criterion is simply
                # majority(residual) == 0.

                recovered_residual = (
                    recovery.recover(
                        residual_error
                    )
                )

                if recovered_residual == 0:

                    success_count += count

            success_probability = (
                success_count
                / group_total
            )

            if (
                success_probability
                > best_success
            ):

                best_success = (
                    success_probability
                )

                best_correction = correction

        logical_decoder[
            observation
        ] = best_correction

    # --------------------------------------------------------
    # Evaluate the resulting logical decoder.
    # --------------------------------------------------------

    results = evaluate_decoder(
        samples,
        logical_decoder
    )

    print()

    print(
        f"Observations evaluated : "
        f"{total_observations}"
    )

    print(
        f"Exact error accuracy   : "
        f"{results['exact_error']:.4f}"
    )

    print(
        f"Bit accuracy           : "
        f"{results['bit_accuracy']:.4f}"
    )

    print(
        f"Physical recovery      : "
        f"{results['physical_recovery']:.4f}"
    )

    print(
        f"Logical success        : "
        f"{results['logical_success']:.4f}"
    )

    return logical_decoder, results


# ============================================================
# TEST 8
#
# SHOW REPRESENTATIVE OBSERVATIONS
# ============================================================

def show_decoder_examples(
    samples,
    groups,
    logical_decoder,
    maximum_examples=12
):

    print()
    print("=" * 60)
    print(
        " TEST 4: LOGICAL DECODER EXAMPLES"
    )
    print("=" * 60)

    shown = 0

    for observation in sorted(
        groups
    ):

        if shown >= maximum_examples:
            break

        counts = groups[
            observation
        ]

        best_error = max(
            counts,
            key=counts.get
        )

        correction = logical_decoder[
            observation
        ]

        total = sum(
            counts.values()
        )

        best_error_probability = (
            counts[best_error]
            / total
        )

        print()

        print(
            f"Observation: {observation}"
        )

        print(
            f"  Most likely error : "
            f"{state_to_string(best_error)} "
            f"({best_error_probability:.2%})"
        )

        print(
            f"  Logical correction: "
            f"{state_to_string(correction)}"
        )

        shown += 1

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " LOGICAL EQUIVALENCE DECODER DIAGNOSTIC"
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
    # TEST 1
    # --------------------------------------------------------

    mapping_passed = (
        test_syndrome_error_pairs()
    )

    if not mapping_passed:

        print()
        print(
            "RESULT : FAIL"
        )

        return

    # --------------------------------------------------------
    # Generate dataset.
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
    # Build temporal observation groups.
    # --------------------------------------------------------

    groups = (
        build_observation_groups(
            samples
        )
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    test_observation_error_probabilities(
        samples,
        groups
    )

    # --------------------------------------------------------
    # Build standard temporal empirical decoder.
    # --------------------------------------------------------

    temporal_decoder = (
        build_temporal_decoder(
            groups
        )
    )

    temporal_results = evaluate_decoder(
        samples,
        temporal_decoder
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    (
        logical_decoder,
        logical_results
    ) = (
        test_best_logical_correction_per_observation(
            samples,
            groups
        )
    )

    # --------------------------------------------------------
    # Test 4
    # --------------------------------------------------------

    show_decoder_examples(
        samples,
        groups,
        logical_decoder
    )

    # --------------------------------------------------------
    # Traditional final-syndrome decoder.
    # --------------------------------------------------------

    traditional_results = (
        evaluate_final_syndrome_decoder(
            samples
        )
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        " FINAL COMPARISON"
    )
    print("=" * 60)

    print()

    print(
        "Traditional final syndrome:"
    )

    print(
        f"  Exact error      : "
        f"{traditional_results['exact_error']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{traditional_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{traditional_results['logical_success']:.4f}"
    )

    print()

    print(
        "Temporal empirical error decoder:"
    )

    print(
        f"  Exact error      : "
        f"{temporal_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{temporal_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{temporal_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{temporal_results['logical_success']:.4f}"
    )

    print()

    print(
        "Logical-objective decoder:"
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

    logical_gain = (
        logical_results["logical_success"]
        - temporal_results["logical_success"]
    )

    print(
        f"Logical-objective gain over "
        f"exact-error decoder: "
        f"{logical_gain:+.4f}"
    )

    print()

    if (
        logical_results["logical_success"]
        > temporal_results["logical_success"]
        + 0.01
    ):

        print(
            "CONCLUSION:"
        )

        print(
            "Optimizing the decoder for the logical "
            "QEC objective is materially better than "
            "optimizing exact physical-error prediction."
        )

    else:

        print(
            "CONCLUSION:"
        )

        print(
            "Exact physical-error prediction and "
            "logical-objective optimization perform "
            "similarly under this configuration."
        )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " LOGICAL EQUIVALENCE DECODER : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()