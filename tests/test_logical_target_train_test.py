from collections import Counter, defaultdict

import numpy as np

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier,
    ERROR_PATTERNS
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

TRAIN_SAMPLES = 20000

TEST_SAMPLES = 5000

SEED = 42

HIDDEN_SIZE = 64

EPOCHS = 100

LEARNING_RATE = 0.003


# ============================================================
# CORRECTION PATTERNS
# ============================================================

CORRECTIONS = [
    tuple(pattern)
    for pattern in ERROR_PATTERNS
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


def encode_features(
    sample
):
    """
    Convert:

        observed syndrome history
        +
        detection event history

    into a temporal feature sequence.

    Shape:

        rounds x 4

    For 5 rounds:

        5 x 4
    """

    features = []

    syndrome_history = sample[
        "observed_syndrome_history"
    ]

    detection_events = sample[
        "detection_events"
    ]

    for syndrome, detection in zip(
        syndrome_history,
        detection_events
    ):

        features.append([
            int(syndrome[0]),
            int(syndrome[1]),
            int(detection[0]),
            int(detection[1])
        ])

    return features


# ============================================================
# LOGICAL SUCCESS
# ============================================================

def calculate_logical_success(
    actual_error,
    correction,
    logical_state
):

    # --------------------------------------------------------
    # Encoded physical state
    # --------------------------------------------------------

    if logical_state == 0:

        encoded_state = [
            0, 0, 0
        ]

    else:

        encoded_state = [
            1, 1, 1
        ]

    # --------------------------------------------------------
    # Error corrupts encoded state
    # --------------------------------------------------------

    corrupted_state = xor_states(
        encoded_state,
        actual_error
    )

    # --------------------------------------------------------
    # Correction is applied
    # --------------------------------------------------------

    corrected_state = xor_states(
        corrupted_state,
        correction
    )

    # --------------------------------------------------------
    # Majority recovery
    # --------------------------------------------------------

    recovered_logical = (
        LogicalRecovery().recover(
            corrected_state
        )
    )

    return (
        recovered_logical
        == logical_state
    )


# ============================================================
# BUILD LOGICAL-OPTIMAL TARGETS
#
# IMPORTANT:
#
# These targets are created ONLY from the
# TRAINING SET.
#
# The TEST SET is never used here.
# ============================================================

def build_logical_targets(
    training_samples
):

    observation_groups = (
        defaultdict(Counter)
    )

    logical_states = (
        defaultdict(Counter)
    )

    # --------------------------------------------------------
    # Group training observations by:
    #
    # observed syndrome history
    #
    # and record final errors.
    # --------------------------------------------------------

    for sample in training_samples:

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

        observation_groups[
            observation
        ][
            error_state
        ] += 1

        logical_states[
            observation
        ][
            int(
                sample[
                    "logical_state"
                ]
            )
        ] += 1

    # --------------------------------------------------------
    # Find best correction for every training observation.
    # --------------------------------------------------------

    logical_targets = {}

    target_scores = {}

    for observation, error_counts in (
        observation_groups.items()
    ):

        best_correction = None

        best_score = -1.0

        total = sum(
            error_counts.values()
        )

        for correction in CORRECTIONS:

            success_count = 0

            for actual_error, count in (
                error_counts.items()
            ):

                # Logical state does not affect
                # whether the correction preserves
                # the encoded information.
                #
                # We nevertheless evaluate both
                # logical states to make this explicit.

                success_for_error = True

                for logical_state in (
                    0,
                    1
                ):

                    if not calculate_logical_success(
                        actual_error,
                        correction,
                        logical_state
                    ):

                        success_for_error = False

                        break

                if success_for_error:

                    success_count += count

            score = (
                success_count
                / total
            )

            if score > best_score:

                best_score = score

                best_correction = correction

        logical_targets[
            observation
        ] = best_correction

        target_scores[
            observation
        ] = best_score

    return (
        logical_targets,
        target_scores,
        observation_groups
    )


# ============================================================
# BUILD EXACT-ERROR TARGETS
#
# This is the normal maximum-likelihood physical-error
# decoder target.
# ============================================================

def build_exact_error_targets(
    training_samples
):

    observation_groups = (
        defaultdict(Counter)
    )

    for sample in training_samples:

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

        observation_groups[
            observation
        ][
            error_state
        ] += 1

    targets = {}

    for observation, counts in (
        observation_groups.items()
    ):

        targets[
            observation
        ] = max(
            counts,
            key=counts.get
        )

    return targets


# ============================================================
# EVALUATE A CORRECTION DECODER
# ============================================================

def evaluate_decoder(
    samples,
    decoder_function
):

    recovery = LogicalRecovery()

    total = len(samples)

    exact_error = 0

    bit_correct = 0

    total_bits = 0

    physical_recovery = 0

    logical_success = 0

    for sample in samples:

        correction = list(
            decoder_function(
                sample
            )
        )

        actual_error = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        logical_state = int(
            sample[
                "logical_state"
            ]
        )

        # ----------------------------------------------------
        # Exact error prediction
        # ----------------------------------------------------

        if (
            correction
            == actual_error
        ):

            exact_error += 1

        # ----------------------------------------------------
        # Bit accuracy
        # ----------------------------------------------------

        for predicted_bit, actual_bit in zip(
            correction,
            actual_error
        ):

            if predicted_bit == actual_bit:

                bit_correct += 1

            total_bits += 1

        # ----------------------------------------------------
        # Encoded state
        # ----------------------------------------------------

        if logical_state == 0:

            encoded_state = [
                0, 0, 0
            ]

        else:

            encoded_state = [
                1, 1, 1
            ]

        # ----------------------------------------------------
        # Corrupted state
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
            correction
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
# TRADITIONAL LOOKUP DECODER
# ============================================================

def traditional_decoder_function(
    sample
):

    decoder = RepeatedLookupDecoder()

    return decoder.decode_history(
        sample[
            "observed_syndrome_history"
        ]
    )


# ============================================================
# TRAIN GRU
#
# TARGET:
#
# logical-optimal correction
# ============================================================

def train_logical_gru(
    training_samples,
    logical_targets
):

    X = []

    y = []

    for sample in training_samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        X.append(
            encode_features(
                sample
            )
        )

        y.append(
            list(
                logical_targets[
                    observation
                ]
            )
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    decoder = TemporalGRUClassifier(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=SEED
    )

    decoder.train(
        X,
        y,
        verbose=True
    )

    return decoder


# ============================================================
# TRAIN EXACT-ERROR GRU
#
# This reproduces our existing physical-error objective.
# ============================================================

def train_exact_error_gru(
    training_samples
):

    X = []

    y = []

    exact_targets = (
        build_exact_error_targets(
            training_samples
        )
    )

    for sample in training_samples:

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        X.append(
            encode_features(
                sample
            )
        )

        y.append(
            list(
                exact_targets[
                    observation
                ]
            )
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    decoder = TemporalGRUClassifier(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        random_seed=SEED
    )

    decoder.train(
        X,
        y,
        verbose=True
    )

    return decoder


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " HELD-OUT LOGICAL-TARGET AI EXPERIMENT"
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
        f"Total samples             : "
        f"{TOTAL_SAMPLES}"
    )

    print(
        f"Training samples          : "
        f"{TRAIN_SAMPLES}"
    )

    print(
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # ========================================================
    # GENERATE ONE COMMON DATASET
    # ========================================================

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

    print()

    print(
        f"Generated samples         : "
        f"{len(samples)}"
    )

    # ========================================================
    # FIXED TRAIN / TEST SPLIT
    # ========================================================

    training_samples = (
        samples[
            :TRAIN_SAMPLES
        ]
    )

    test_samples = (
        samples[
            TRAIN_SAMPLES:
            TRAIN_SAMPLES + TEST_SAMPLES
        ]
    )

    print()

    print(
        f"Training set              : "
        f"{len(training_samples)}"
    )

    print(
        f"Test set                  : "
        f"{len(test_samples)}"
    )

    # ========================================================
    # BUILD LOGICAL TARGETS FROM TRAINING ONLY
    # ========================================================

    (
        logical_targets,
        target_scores,
        training_groups
    ) = build_logical_targets(
        training_samples
    )

    print()

    print("=" * 60)
    print(
        " LOGICAL TARGET GENERATION"
    )
    print("=" * 60)

    print()

    print(
        f"Training observations     : "
        f"{len(training_groups)}"
    )

    print(
        f"Logical target count      : "
        f"{len(logical_targets)}"
    )

    average_target_score = (
        sum(
            target_scores.values()
        )
        / len(target_scores)
    )

    print(
        f"Average training logical "
        f"target score             : "
        f"{average_target_score:.4f}"
    )

    # ========================================================
    # TEST OBSERVATION COVERAGE
    # ========================================================

    test_observations = set()

    for sample in test_samples:

        test_observations.add(
            observation_to_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

    seen_test_observations = (
        test_observations
        & set(logical_targets.keys())
    )

    unseen_test_observations = (
        test_observations
        - set(logical_targets.keys())
    )

    print()

    print(
        f"Unique test observations   : "
        f"{len(test_observations)}"
    )

    print(
        f"Seen in training           : "
        f"{len(seen_test_observations)}"
    )

    print(
        f"Unseen in training         : "
        f"{len(unseen_test_observations)}"
    )

    # ========================================================
    # EMPIRICAL TRAINING-TARGET DECODER
    #
    # This is the cleanest reference:
    #
    # train observation -> logical target
    # test observation -> same target
    #
    # Unseen observations use 000 as a neutral fallback.
    # ========================================================

    def training_logical_target_decoder(
        sample
    ):

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        if observation in logical_targets:

            return logical_targets[
                observation
            ]

        return (
            0,
            0,
            0
        )

    # ========================================================
    # EMPIRICAL EXACT-ERROR DECODER
    # ========================================================

    exact_targets = (
        build_exact_error_targets(
            training_samples
        )
    )

    def training_exact_decoder(
        sample
    ):

        observation = observation_to_key(
            sample[
                "observed_syndrome_history"
            ]
        )

        if observation in exact_targets:

            return exact_targets[
                observation
            ]

        return (
            0,
            0,
            0
        )

    # ========================================================
    # TEST 1:
    # EMPIRICAL BASELINES
    # ========================================================

    print()
    print("=" * 60)
    print(
        " TEST 1: HELD-OUT EMPIRICAL BASELINES"
    )
    print("=" * 60)

    traditional_results = (
        evaluate_decoder(
            test_samples,
            traditional_decoder_function
        )
    )

    exact_results = (
        evaluate_decoder(
            test_samples,
            training_exact_decoder
        )
    )

    logical_results = (
        evaluate_decoder(
            test_samples,
            training_logical_target_decoder
        )
    )

    print()

    print(
        "Traditional lookup:"
    )

    print(
        f"  Exact error      : "
        f"{traditional_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{traditional_results['bit_accuracy']:.4f}"
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
        "Training-derived exact-error target:"
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

    print(
        "Training-derived logical target:"
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

    # ========================================================
    # TRAIN EXACT-ERROR GRU
    # ========================================================

    print()
    print("=" * 60)
    print(
        " TEST 2: TRAIN EXACT-ERROR GRU"
    )
    print("=" * 60)

    exact_gru = train_exact_error_gru(
        training_samples
    )

    # ========================================================
    # TRAIN LOGICAL-TARGET GRU
    # ========================================================

    print()
    print("=" * 60)
    print(
        " TEST 3: TRAIN LOGICAL-TARGET GRU"
    )
    print("=" * 60)

    logical_gru = train_logical_gru(
        training_samples,
        logical_targets
    )

    # ========================================================
    # GRU EVALUATION
    # ========================================================

    def exact_gru_decoder(
        sample
    ):

        features = encode_features(
            sample
        )

        return exact_gru.decode(
            features
        )

    def logical_gru_decoder(
        sample
    ):

        features = encode_features(
            sample
        )

        return logical_gru.decode(
            features
        )

    # ========================================================
    # TEST 4:
    # AI RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print(
        " TEST 4: HELD-OUT AI RESULTS"
    )
    print("=" * 60)

    exact_gru_results = (
        evaluate_decoder(
            test_samples,
            exact_gru_decoder
        )
    )

    logical_gru_results = (
        evaluate_decoder(
            test_samples,
            logical_gru_decoder
        )
    )

    print()

    print(
        "Exact-error GRU:"
    )

    print(
        f"  Exact error      : "
        f"{exact_gru_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{exact_gru_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{exact_gru_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{exact_gru_results['logical_success']:.4f}"
    )

    print()

    print(
        "Logical-target GRU:"
    )

    print(
        f"  Exact error      : "
        f"{logical_gru_results['exact_error']:.4f}"
    )

    print(
        f"  Bit accuracy     : "
        f"{logical_gru_results['bit_accuracy']:.4f}"
    )

    print(
        f"  Physical recovery: "
        f"{logical_gru_results['physical_recovery']:.4f}"
    )

    print(
        f"  Logical success  : "
        f"{logical_gru_results['logical_success']:.4f}"
    )

    # ========================================================
    # TEST 5:
    # AI IMPROVEMENT
    # ========================================================

    logical_gain = (
        logical_gru_results[
            "logical_success"
        ]
        -
        exact_gru_results[
            "logical_success"
        ]
    )

    print()
    print("=" * 60)
    print(
        " FINAL COMPARISON"
    )
    print("=" * 60)

    print()

    print(
        "Decoder                         "
        "Logical Success"
    )

    print(
        "-" * 60
    )

    print(
        f"Traditional lookup             "
        f"{traditional_results['logical_success']:.4f}"
    )

    print(
        f"Empirical exact-error target   "
        f"{exact_results['logical_success']:.4f}"
    )

    print(
        f"Empirical logical target       "
        f"{logical_results['logical_success']:.4f}"
    )

    print(
        f"Exact-error GRU                 "
        f"{exact_gru_results['logical_success']:.4f}"
    )

    print(
        f"Logical-target GRU              "
        f"{logical_gru_results['logical_success']:.4f}"
    )

    print()

    print(
        f"Logical-target GRU gain over "
        f"exact-error GRU              : "
        f"{logical_gain:+.4f}"
    )

    print()

    if logical_gain > 0.01:

        print(
            "CONCLUSION:"
        )

        print(
            "The logical-target GRU improves "
            "held-out logical QEC performance."
        )

        print()

        print(
            "This supports changing the AI "
            "decoder objective from exact "
            "physical-error prediction to "
            "logical-preserving correction."
        )

    elif logical_gain > 0:

        print(
            "CONCLUSION:"
        )

        print(
            "The logical-target GRU shows a "
            "positive held-out improvement, "
            "but the gain is currently small."
        )

    else:

        print(
            "CONCLUSION:"
        )

        print(
            "The logical-target GRU does not "
            "yet outperform the exact-error "
            "GRU on this held-out experiment."
        )

        print(
            "The logical objective may still "
            "be useful, but the current model "
            "or target representation needs "
            "further investigation."
        )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " HELD-OUT LOGICAL-TARGET AI : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()