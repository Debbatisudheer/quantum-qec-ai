from collections import Counter, defaultdict

import numpy as np

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.temporal_gru_classifier import (
    TemporalGRUClassifier
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

HIDDEN_SIZE = 64

EPOCHS = 100

LEARNING_RATE = 0.003

SEEDS = [
    42,
    123,
    456
]


# ============================================================
# CORRECTION PATTERNS
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


def encode_features(sample):

    syndrome_history = sample[
        "observed_syndrome_history"
    ]

    detection_events = sample[
        "detection_events"
    ]

    features = []

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

def logical_success(
    actual_error,
    correction,
    logical_state
):

    if logical_state == 0:

        encoded_state = [
            0,
            0,
            0
        ]

    else:

        encoded_state = [
            1,
            1,
            1
        ]

    corrupted_state = xor_states(
        encoded_state,
        actual_error
    )

    corrected_state = xor_states(
        corrupted_state,
        correction
    )

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
# BUILD LOGICAL TARGETS FROM TRAINING ONLY
# ============================================================

def build_logical_targets(
    training_samples
):

    groups = defaultdict(Counter)

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

        groups[
            observation
        ][
            error_state
        ] += 1

    targets = {}

    scores = {}

    for observation, error_counts in (
        groups.items()
    ):

        total = sum(
            error_counts.values()
        )

        best_correction = None

        best_score = -1.0

        for correction in CORRECTIONS:

            success_count = 0

            for actual_error, count in (
                error_counts.items()
            ):

                # Logical preservation is independent
                # of whether the encoded logical state
                # is 0 or 1, so evaluate the residual
                # logical effect once.

                residual = xor_states(
                    actual_error,
                    correction
                )

                recovered = (
                    LogicalRecovery().recover(
                        residual
                    )
                )

                if recovered == 0:

                    success_count += count

            score = (
                success_count
                / total
            )

            if score > best_score:

                best_score = score

                best_correction = correction

        targets[
            observation
        ] = best_correction

        scores[
            observation
        ] = best_score

    return targets, scores


# ============================================================
# BUILD EXACT-ERROR TARGETS
# ============================================================

def build_exact_targets(
    training_samples
):

    groups = defaultdict(Counter)

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

        groups[
            observation
        ][
            error_state
        ] += 1

    targets = {}

    for observation, counts in (
        groups.items()
    ):

        targets[
            observation
        ] = max(
            counts,
            key=counts.get
        )

    return targets


# ============================================================
# EVALUATE DECODER
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

    logical_success_count = 0

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
        # Exact error
        # ----------------------------------------------------

        if correction == actual_error:

            exact_error += 1

        # ----------------------------------------------------
        # Bit accuracy
        # ----------------------------------------------------

        for predicted, actual in zip(
            correction,
            actual_error
        ):

            if predicted == actual:

                bit_correct += 1

            total_bits += 1

        # ----------------------------------------------------
        # Encoded state
        # ----------------------------------------------------

        encoded_state = [
            logical_state,
            logical_state,
            logical_state
        ]

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
            correction
        )

        # ----------------------------------------------------
        # Physical recovery
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

            logical_success_count += 1

    return {

        "exact_error":
            exact_error / total,

        "bit_accuracy":
            bit_correct / total_bits,

        "physical_recovery":
            physical_recovery / total,

        "logical_success":
            logical_success_count / total
    }


# ============================================================
# TRADITIONAL LOOKUP
# ============================================================

def traditional_decoder(
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
# ============================================================

def train_gru(
    training_samples,
    targets,
    seed
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
                targets[
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
        random_seed=seed
    )

    decoder.train(
        X,
        y,
        verbose=False
    )

    return decoder


# ============================================================
# RUN ONE SEED
# ============================================================

def run_seed(seed):

    print()
    print("=" * 60)
    print(
        f" SEED {seed}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Generate data
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
            seed=seed
        )
    )

    samples = generator.generate_dataset(
        TOTAL_SAMPLES
    )

    training_samples = samples[
        :TRAIN_SAMPLES
    ]

    test_samples = samples[
        TRAIN_SAMPLES:
        TRAIN_SAMPLES + TEST_SAMPLES
    ]

    # --------------------------------------------------------
    # Build targets using TRAINING DATA ONLY
    # --------------------------------------------------------

    logical_targets, target_scores = (
        build_logical_targets(
            training_samples
        )
    )

    exact_targets = (
        build_exact_targets(
            training_samples
        )
    )

    # --------------------------------------------------------
    # Test observation coverage
    # --------------------------------------------------------

    test_observations = set()

    for sample in test_samples:

        test_observations.add(
            observation_to_key(
                sample[
                    "observed_syndrome_history"
                ]
            )
        )

    seen = (
        test_observations
        & set(logical_targets.keys())
    )

    unseen = (
        test_observations
        - set(logical_targets.keys())
    )

    print()

    print(
        f"Training observations     : "
        f"{len(logical_targets)}"
    )

    print(
        f"Unique test observations  : "
        f"{len(test_observations)}"
    )

    print(
        f"Seen test observations    : "
        f"{len(seen)}"
    )

    print(
        f"Unseen test observations  : "
        f"{len(unseen)}"
    )

    # --------------------------------------------------------
    # Empirical logical target decoder
    # --------------------------------------------------------

    def empirical_logical(
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

    # --------------------------------------------------------
    # Empirical exact-error decoder
    # --------------------------------------------------------

    def empirical_exact(
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

    # --------------------------------------------------------
    # Train exact-error GRU
    # --------------------------------------------------------

    exact_gru = train_gru(
        training_samples,
        exact_targets,
        seed
    )

    # --------------------------------------------------------
    # Train logical-target GRU
    # --------------------------------------------------------

    logical_gru = train_gru(
        training_samples,
        logical_targets,
        seed
    )

    # --------------------------------------------------------
    # GRU decoder functions
    # --------------------------------------------------------

    def exact_gru_decoder(
        sample
    ):

        return exact_gru.decode(
            encode_features(
                sample
            )
        )

    def logical_gru_decoder(
        sample
    ):

        return logical_gru.decode(
            encode_features(
                sample
            )
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    traditional_results = (
        evaluate_decoder(
            test_samples,
            traditional_decoder
        )
    )

    empirical_exact_results = (
        evaluate_decoder(
            test_samples,
            empirical_exact
        )
    )

    empirical_logical_results = (
        evaluate_decoder(
            test_samples,
            empirical_logical
        )
    )

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

    gain = (
        logical_gru_results[
            "logical_success"
        ]
        -
        exact_gru_results[
            "logical_success"
        ]
    )

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
        f"{empirical_exact_results['logical_success']:.4f}"
    )

    print(
        f"Empirical logical target       "
        f"{empirical_logical_results['logical_success']:.4f}"
    )

    print(
        f"Exact-error GRU                "
        f"{exact_gru_results['logical_success']:.4f}"
    )

    print(
        f"Logical-target GRU             "
        f"{logical_gru_results['logical_success']:.4f}"
    )

    print()

    print(
        f"Logical-target GRU gain        "
        f"{gain:+.4f}"
    )

    return {

        "seed":
            seed,

        "traditional":
            traditional_results[
                "logical_success"
            ],

        "empirical_exact":
            empirical_exact_results[
                "logical_success"
            ],

        "empirical_logical":
            empirical_logical_results[
                "logical_success"
            ],

        "exact_gru":
            exact_gru_results[
                "logical_success"
            ],

        "logical_gru":
            logical_gru_results[
                "logical_success"
            ],

        "gain":
            gain,

        "test_observations":
            len(test_observations),

        "seen_observations":
            len(seen),

        "unseen_observations":
            len(unseen)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        " LOGICAL-TARGET ROBUSTNESS EXPERIMENT"
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
        f"Training samples          : "
        f"{TRAIN_SAMPLES}"
    )

    print(
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Seeds                     : "
        f"{SEEDS}"
    )

    # --------------------------------------------------------
    # Run every seed
    # --------------------------------------------------------

    results = []

    for seed in SEEDS:

        result = run_seed(
            seed
        )

        results.append(
            result
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print(
        " ROBUSTNESS SUMMARY"
    )
    print("=" * 60)

    print()

    print(
        "Seed       Traditional   "
        "Exact-GRU   Logical-GRU   Gain"
    )

    print(
        "-" * 70
    )

    for result in results:

        print(
            f"{result['seed']:<10}"
            f"{result['traditional']:.4f}       "
            f"{result['exact_gru']:.4f}      "
            f"{result['logical_gru']:.4f}       "
            f"{result['gain']:+.4f}"
        )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    traditional_scores = [
        result[
            "traditional"
        ]
        for result in results
    ]

    exact_scores = [
        result[
            "exact_gru"
        ]
        for result in results
    ]

    logical_scores = [
        result[
            "logical_gru"
        ]
        for result in results
    ]

    gains = [
        result[
            "gain"
        ]
        for result in results
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    traditional_mean = np.mean(
        traditional_scores
    )

    exact_mean = np.mean(
        exact_scores
    )

    logical_mean = np.mean(
        logical_scores
    )

    gain_mean = np.mean(
        gains
    )

    traditional_std = np.std(
        traditional_scores,
        ddof=1
    )

    exact_std = np.std(
        exact_scores,
        ddof=1
    )

    logical_std = np.std(
        logical_scores,
        ddof=1
    )

    gain_std = np.std(
        gains,
        ddof=1
    )

    # ========================================================
    # PRINT STATISTICS
    # ========================================================

    print()

    print(
        "Mean logical success:"
    )

    print(
        f"Traditional lookup      : "
        f"{traditional_mean:.4f}"
    )

    print(
        f"Exact-error GRU         : "
        f"{exact_mean:.4f}"
    )

    print(
        f"Logical-target GRU      : "
        f"{logical_mean:.4f}"
    )

    print()

    print(
        "Standard deviation:"
    )

    print(
        f"Traditional lookup      : "
        f"{traditional_std:.4f}"
    )

    print(
        f"Exact-error GRU         : "
        f"{exact_std:.4f}"
    )

    print(
        f"Logical-target GRU      : "
        f"{logical_std:.4f}"
    )

    print()

    print(
        f"Mean logical-target gain: "
        f"{gain_mean:+.4f}"
    )

    print(
        f"Gain standard deviation : "
        f"{gain_std:.4f}"
    )

    # ========================================================
    # WIN COUNT
    # ========================================================

    logical_wins = sum(
        1
        for gain in gains
        if gain > 0
    )

    exact_wins = sum(
        1
        for gain in gains
        if gain < 0
    )

    ties = sum(
        1
        for gain in gains
        if gain == 0
    )

    print()

    print(
        "Logical-target GRU wins : "
        f"{logical_wins}/{len(SEEDS)}"
    )

    print(
        "Exact-error GRU wins    : "
        f"{exact_wins}/{len(SEEDS)}"
    )

    print(
        "Ties                    : "
        f"{ties}/{len(SEEDS)}"
    )

    # ========================================================
    # DECISION
    # ========================================================

    print()
    print("=" * 60)
    print(
        " ROBUSTNESS DECISION"
    )
    print("=" * 60)

    print()

    if (
        logical_wins == len(SEEDS)
        and gain_mean > 0.01
    ):

        print(
            "PASS"
        )

        print()

        print(
            "The logical-target GRU consistently "
            "outperforms the exact-error GRU "
            "across all tested seeds."
        )

        print()

        print(
            "RECOMMENDATION:"
        )

        print(
            "Promote the logical-target objective "
            "to the primary AI decoder objective."
        )

    elif gain_mean > 0:

        print(
            "POSITIVE BUT NOT FULLY ROBUST"
        )

        print()

        print(
            "The logical-target GRU has a positive "
            "average improvement, but the result "
            "is not consistently dominant across "
            "all seeds."
        )

    else:

        print(
            "NO ROBUST IMPROVEMENT"
        )

        print()

        print(
            "The logical-target GRU does not show "
            "a consistent advantage under the "
            "tested seeds."
        )

    print()

    print(
        "RESULT : DIAGNOSTIC COMPLETE"
    )

    print()
    print("=" * 60)
    print(
        " LOGICAL-TARGET ROBUSTNESS : COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()