from collections import Counter

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from evaluation.logical_recovery import LogicalRecovery


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITIES = [
    0.00,
    0.10,
]

SAMPLES = 10000

SEED = 42


# ============================================================
# HELPERS
# ============================================================

def xor_states(a, b):
    return [
        int(x) ^ int(y)
        for x, y in zip(a, b)
    ]


def encoded_state_to_list(encoded_state):
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
# TEST 1
#
# PERFECT DECODER
#
# The perfect decoder knows the exact final error.
#
# Correct pipeline:
#
# encoded state
#      XOR
# actual error
#      =
# corrupted physical state
#
# corrupted physical state
#      XOR
# predicted error
#      =
# corrected physical state
#
# corrected physical state
#      ->
# LogicalRecovery
# ============================================================

def test_perfect_decoder(
    measurement_noise_probability
):

    print()
    print("=" * 60)
    print(" PERFECT DECODER TEST")
    print("=" * 60)

    print(
        f"Rounds                 : {ROUNDS}"
    )

    print(
        f"Physical error         : "
        f"{PHYSICAL_ERROR_PROBABILITY}"
    )

    print(
        f"Measurement noise      : "
        f"{measurement_noise_probability}"
    )

    print(
        f"Samples                : {SAMPLES}"
    )

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            seed=SEED
        )
    )

    recovery = LogicalRecovery()

    error_cancellation_success = 0

    physical_state_success = 0

    logical_success = 0

    target_distribution = Counter()

    corrected_state_distribution = Counter()

    logical_predictions = Counter()

    for sample_id in range(SAMPLES):

        sample = generator.generate_sample(
            sample_id
        )

        logical_state = int(
            sample["logical_state"]
        )

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        actual_error = list(
            sample["final_error_state"]
        )

        target_distribution[
            state_to_string(actual_error)
        ] += 1

        # ----------------------------------------------------
        # CORRUPTED PHYSICAL STATE
        # ----------------------------------------------------

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        # ----------------------------------------------------
        # PERFECT DECODER
        #
        # It knows the exact actual error.
        # ----------------------------------------------------

        predicted_error = (
            actual_error.copy()
        )

        # ----------------------------------------------------
        # APPLY CORRECTION TO THE CORRUPTED
        # PHYSICAL STATE
        # ----------------------------------------------------

        corrected_state = xor_states(
            corrupted_state,
            predicted_error
        )

        corrected_state_distribution[
            state_to_string(corrected_state)
        ] += 1

        # ----------------------------------------------------
        # ERROR-CANCELLATION METRIC
        #
        # actual error XOR predicted error
        #
        # must be 000.
        # ----------------------------------------------------

        remaining_error = xor_states(
            actual_error,
            predicted_error
        )

        if remaining_error == [
            0,
            0,
            0
        ]:

            error_cancellation_success += 1

        # ----------------------------------------------------
        # PHYSICAL STATE SUCCESS
        #
        # corrected physical state must equal
        # the original encoded physical state.
        # ----------------------------------------------------

        if corrected_state == encoded_state:

            physical_state_success += 1

        # ----------------------------------------------------
        # LOGICAL RECOVERY
        #
        # IMPORTANT:
        #
        # recover() receives the corrected PHYSICAL STATE,
        # not the error pattern.
        # ----------------------------------------------------

        recovered_logical = recovery.recover(
            corrected_state
        )

        logical_predictions[
            recovered_logical
        ] += 1

        if recovered_logical == logical_state:

            logical_success += 1

    error_cancellation_accuracy = (
        error_cancellation_success
        / SAMPLES
    )

    physical_accuracy = (
        physical_state_success
        / SAMPLES
    )

    logical_accuracy = (
        logical_success
        / SAMPLES
    )

    print()
    print(
        "Actual error distribution:"
    )

    for state in sorted(
        target_distribution
    ):

        count = target_distribution[state]

        print(
            f"{state} "
            f"{count:5d} "
            f"({count / SAMPLES:.2%})"
        )

    print()
    print(
        "Corrected physical-state distribution:"
    )

    for state in sorted(
        corrected_state_distribution
    ):

        count = corrected_state_distribution[
            state
        ]

        print(
            f"{state} "
            f"{count:5d} "
            f"({count / SAMPLES:.2%})"
        )

    print()
    print(
        "Recovered logical-state distribution:"
    )

    for logical_state in [0, 1]:

        count = logical_predictions[
            logical_state
        ]

        print(
            f"Logical {logical_state}: "
            f"{count:5d} "
            f"({count / SAMPLES:.2%})"
        )

    print()

    print(
        f"Error cancellation : "
        f"{error_cancellation_accuracy:.4f}"
    )

    print(
        f"Physical recovery  : "
        f"{physical_accuracy:.4f}"
    )

    print(
        f"Logical success    : "
        f"{logical_accuracy:.4f}"
    )

    passed = (
        error_cancellation_accuracy == 1.0
        and physical_accuracy == 1.0
        and logical_accuracy == 1.0
    )

    print()

    print(
        "PERFECT DECODER : "
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
# MEASUREMENT NOISE MUST NOT AFFECT
# THE PERFECT DECODER
#
# The perfect decoder uses ground truth directly.
# Therefore measurement noise is irrelevant.
# ============================================================

def test_measurement_noise_independence():

    print()
    print("=" * 60)
    print(
        " TEST 2: MEASUREMENT NOISE INDEPENDENCE"
    )
    print("=" * 60)

    results = []

    for measurement_noise_probability in (
        MEASUREMENT_NOISE_PROBABILITIES
    ):

        generator = (
            TimeVaryingQECDatasetGenerator(
                rounds=ROUNDS,
                physical_error_probability=(
                    PHYSICAL_ERROR_PROBABILITY
                ),
                measurement_noise_probability=(
                    measurement_noise_probability
                ),
                seed=SEED
            )
        )

        recovery = LogicalRecovery()

        error_cancellation_success = 0

        physical_success = 0

        logical_success = 0

        for sample_id in range(SAMPLES):

            sample = generator.generate_sample(
                sample_id
            )

            logical_state = int(
                sample["logical_state"]
            )

            encoded_state = (
                encoded_state_to_list(
                    sample["encoded_state"]
                )
            )

            actual_error = list(
                sample["final_error_state"]
            )

            # Corrupt the encoded physical state.

            corrupted_state = xor_states(
                encoded_state,
                actual_error
            )

            # Perfect prediction.

            predicted_error = (
                actual_error.copy()
            )

            # Apply correction.

            corrected_state = xor_states(
                corrupted_state,
                predicted_error
            )

            # Error cancellation.

            remaining_error = xor_states(
                actual_error,
                predicted_error
            )

            if remaining_error == [
                0,
                0,
                0
            ]:

                error_cancellation_success += 1

            # Exact physical recovery.

            if corrected_state == encoded_state:

                physical_success += 1

            # Logical recovery.

            recovered = recovery.recover(
                corrected_state
            )

            if recovered == logical_state:

                logical_success += 1

        error_accuracy = (
            error_cancellation_success
            / SAMPLES
        )

        physical_accuracy = (
            physical_success
            / SAMPLES
        )

        logical_accuracy = (
            logical_success
            / SAMPLES
        )

        results.append(
            (
                error_accuracy,
                physical_accuracy,
                logical_accuracy
            )
        )

        print(
            f"Measurement noise="
            f"{measurement_noise_probability:.2f} "
            f"Error cancellation="
            f"{error_accuracy:.4f} "
            f"Physical="
            f"{physical_accuracy:.4f} "
            f"Logical="
            f"{logical_accuracy:.4f}"
        )

    passed = all(
        error == 1.0
        and physical == 1.0
        and logical == 1.0
        for (
            error,
            physical,
            logical
        ) in results
    )

    print()

    print(
        "MEASUREMENT NOISE INDEPENDENCE : "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    return passed


# ============================================================
# TEST 3
#
# ZERO-ERROR DECODER
#
# Always predicts [0,0,0].
#
# This is a negative-control baseline.
# ============================================================

def test_zero_decoder():

    print()
    print("=" * 60)
    print(
        " TEST 3: ZERO-ERROR DECODER BASELINE"
    )
    print("=" * 60)

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=0.10,
            seed=SEED
        )
    )

    recovery = LogicalRecovery()

    physical_success = 0

    logical_success = 0

    for sample_id in range(SAMPLES):

        sample = generator.generate_sample(
            sample_id
        )

        logical_state = int(
            sample["logical_state"]
        )

        encoded_state = (
            encoded_state_to_list(
                sample["encoded_state"]
            )
        )

        actual_error = list(
            sample["final_error_state"]
        )

        # Corrupted physical state.

        corrupted_state = xor_states(
            encoded_state,
            actual_error
        )

        # Bad decoder always predicts no error.

        predicted_error = [
            0,
            0,
            0
        ]

        # Apply incorrect correction.

        corrected_state = xor_states(
            corrupted_state,
            predicted_error
        )

        if corrected_state == encoded_state:

            physical_success += 1

        recovered = recovery.recover(
            corrected_state
        )

        if recovered == logical_state:

            logical_success += 1

    physical_accuracy = (
        physical_success
        / SAMPLES
    )

    logical_accuracy = (
        logical_success
        / SAMPLES
    )

    print()

    print(
        f"Physical recovery : "
        f"{physical_accuracy:.4f}"
    )

    print(
        f"Logical success   : "
        f"{logical_accuracy:.4f}"
    )

    print()

    print(
        "ZERO DECODER : BASELINE ONLY"
    )

    return True


# ============================================================
# TEST 4
#
# LOGICAL STATE DISTRIBUTION
# ============================================================

def test_logical_state_distribution():

    print()
    print("=" * 60)
    print(
        " TEST 4: LOGICAL STATE DISTRIBUTION"
    )
    print("=" * 60)

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=0.10,
            seed=SEED
        )
    )

    logical_distribution = Counter()

    error_distribution = Counter()

    joint_distribution = Counter()

    for sample_id in range(SAMPLES):

        sample = generator.generate_sample(
            sample_id
        )

        logical_state = int(
            sample["logical_state"]
        )

        error_state = state_to_string(
            sample["final_error_state"]
        )

        logical_distribution[
            logical_state
        ] += 1

        error_distribution[
            error_state
        ] += 1

        joint_distribution[
            (
                logical_state,
                error_state
            )
        ] += 1

    print()

    print(
        "Logical state distribution:"
    )

    for logical_state in [0, 1]:

        count = logical_distribution[
            logical_state
        ]

        print(
            f"Logical {logical_state}: "
            f"{count} "
            f"({count / SAMPLES:.2%})"
        )

    print()

    print(
        "Joint logical/error distribution:"
    )

    for logical_state in [0, 1]:

        total = logical_distribution[
            logical_state
        ]

        print()

        print(
            f"Logical {logical_state}"
        )

        for error_state in sorted(
            error_distribution
        ):

            count = joint_distribution[
                (
                    logical_state,
                    error_state
                )
            ]

            if count == 0:
                continue

            print(
                f"  {error_state}: "
                f"{count:5d} "
                f"({count / total:.2%})"
            )

    both_present = (
        logical_distribution[0] > 0
        and logical_distribution[1] > 0
    )

    print()

    print(
        "Both logical states present : "
        + (
            "PASS"
            if both_present
            else "FAIL"
        )
    )

    return both_present


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " PERFECT DECODER END-TO-END DIAGNOSTIC"
    )
    print("=" * 60)

    results = []

    # Test 1: no measurement noise.

    results.append(
        test_perfect_decoder(
            0.00
        )
    )

    # Test 1: measurement noise.

    results.append(
        test_perfect_decoder(
            0.10
        )
    )

    # Test 2.

    results.append(
        test_measurement_noise_independence()
    )

    # Test 3.

    results.append(
        test_zero_decoder()
    )

    # Test 4.

    results.append(
        test_logical_state_distribution()
    )

    print()
    print("=" * 60)
    print(
        " FINAL DIAGNOSTIC RESULT"
    )
    print("=" * 60)

    passed = sum(results)

    total = len(results)

    print(
        f"Tests passed : "
        f"{passed}/{total}"
    )

    if passed == total:

        print()
        print(
            "RESULT : SUCCESS"
        )

        print()
        print(
            "The end-to-end logical recovery "
            "pipeline is consistent."
        )

    else:

        print()
        print(
            "RESULT : FAILURE"
        )

        print()
        print(
            "There is still an inconsistency "
            "in the end-to-end pipeline."
        )

    print()
    print("=" * 60)
    print(
        " PERFECT DECODER DIAGNOSTIC : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()