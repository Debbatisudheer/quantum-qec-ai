from statistics import mean, stdev

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator,
)

from evaluation.logical_recovery import (
    LogicalRecovery,
)


# ============================================================
# BASELINE CONFIGURATION
# ============================================================

ROUNDS = 5

TEST_SAMPLES = 2000

SEEDS = [
    42,
    43,
    44,
]


# ============================================================
# PHYSICAL NOISE LEVELS
# ============================================================

PHYSICAL_NOISE_LEVELS = [
    0.00,
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
]


# ============================================================
# HELPERS
# ============================================================


def majority_vote(bits):
    """
    Recover a logical bit from a 3-qubit physical state
    using majority voting.

    Examples:

        000 -> 0
        001 -> 0
        010 -> 0
        100 -> 0

        111 -> 1
        110 -> 1
        101 -> 1
        011 -> 1
    """

    if len(bits) != 3:
        raise ValueError(
            "majority_vote expects exactly 3 bits"
        )

    return 1 if sum(bits) >= 2 else 0


def evaluate_baseline(samples):
    """
    Evaluate an uncorrected baseline.

    The baseline receives the noisy physical state and
    performs ONLY majority-vote logical recovery.

    No syndrome information is used.
    No AI decoder is used.
    No correction is applied.
    """

    successes = 0

    for sample in samples:

        logical_state = int(
            sample["logical_state"]
        )

        final_error_state = sample[
            "final_error_state"
        ]

        encoded_state = sample[
            "encoded_state"
        ]

        # ----------------------------------------------------
        # Reconstruct corrupted physical state.
        #
        # corrupted = encoded XOR final_error_state
        # ----------------------------------------------------

        corrupted_state = [
            int(a) ^ int(b)
            for a, b in zip(
                encoded_state,
                final_error_state,
            )
        ]

        recovered_logical = majority_vote(
            corrupted_state
        )

        if recovered_logical == logical_state:
            successes += 1

    total = len(samples)

    if total == 0:
        return 0.0

    return successes / total


def run_one_condition(
    physical_noise,
    seed,
):
    """
    Run one baseline condition.
    """

    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=physical_noise,
        measurement_noise_probability=0.0,
        seed=seed,
    )

    samples = generator.generate_dataset(
        TEST_SAMPLES
    )

    logical_success = evaluate_baseline(
        samples
    )

    return {
        "physical_noise": physical_noise,
        "measurement_noise": 0.0,
        "seed": seed,
        "test_samples": TEST_SAMPLES,
        "logical_success": logical_success,
    }


def aggregate(results):
    """
    Aggregate baseline results across seeds.
    """

    grouped = {}

    for result in results:

        noise = result[
            "physical_noise"
        ]

        grouped.setdefault(
            noise,
            []
        ).append(
            result[
                "logical_success"
            ]
        )

    summaries = []

    for noise in sorted(grouped):

        values = grouped[noise]

        summaries.append(
            {
                "physical_noise": noise,

                "logical_mean": mean(
                    values
                ),

                "logical_std": (
                    stdev(values)
                    if len(values) > 1
                    else 0.0
                ),

                "runs": len(values),
            }
        )

    return summaries


# ============================================================
# OUTPUT
# ============================================================


def print_header():

    print()

    print("=" * 78)
    print(
        " UNCOrRECTED BASELINE - PHYSICAL NOISE SWEEP"
    )
    print("=" * 78)

    print()

    print(
        "QEC code             : bit_flip_3"
    )

    print(
        f"Rounds               : {ROUNDS}"
    )

    print(
        f"Test samples / run   : {TEST_SAMPLES}"
    )

    print(
        f"Measurement noise    : 0.00"
    )

    print(
        f"Seeds                : {SEEDS}"
    )

    print()


def print_single_result(result):

    print(
        f"Physical noise = "
        f"{result['physical_noise']:.2f} | "
        f"Seed = "
        f"{result['seed']} | "
        f"Logical success = "
        f"{result['logical_success']:.4f}"
    )


def print_summary(summaries):

    print()

    print("=" * 78)
    print(
        " BASELINE RESULTS"
    )
    print("=" * 78)

    print()

    print(
        f"{'Physical Noise':<18}"
        f"{'Logical Success':>18}"
        f"{'Std Dev':>15}"
    )

    print("-" * 78)

    for summary in summaries:

        print(
            f"{summary['physical_noise']:<18.2f}"
            f"{summary['logical_mean']:>17.2%}"
            f"{summary['logical_std']:>14.2%}"
        )

    print()


def print_interpretation(summaries):

    print("=" * 78)
    print(
        " BASELINE INTERPRETATION"
    )
    print("=" * 78)

    print()

    print(
        "This is the NO-AI baseline."
    )

    print(
        "The noisy physical state is sent directly "
        "to a majority-vote logical recovery."
    )

    print()

    print(
        "No syndrome information is used."
    )

    print(
        "No machine-learning decoder is used."
    )

    print(
        "No physical correction is applied."
    )

    print()

    print(
        "The purpose is to establish how well the "
        "3-qubit repetition code performs without "
        "the AI decoder."
    )

    print()

    if summaries:

        best = max(
            summaries,
            key=lambda item: item[
                "logical_mean"
            ],
        )

        worst = min(
            summaries,
            key=lambda item: item[
                "logical_mean"
            ],
        )

        print(
            f"Best baseline condition: "
            f"physical noise "
            f"{best['physical_noise']:.2f} "
            f"→ "
            f"{best['logical_mean']:.2%} "
            f"logical success"
        )

        print(
            f"Worst baseline condition: "
            f"physical noise "
            f"{worst['physical_noise']:.2f} "
            f"→ "
            f"{worst['logical_mean']:.2%} "
            f"logical success"
        )

    print()


# ============================================================
# MAIN
# ============================================================


def main():

    print_header()

    results = []

    total = (
        len(PHYSICAL_NOISE_LEVELS)
        * len(SEEDS)
    )

    current = 0

    for physical_noise in (
        PHYSICAL_NOISE_LEVELS
    ):

        for seed in SEEDS:

            current += 1

            print(
                f"[{current}/{total}] "
                f"Running baseline: "
                f"physical noise "
                f"{physical_noise:.2f}, "
                f"seed {seed}"
            )

            result = run_one_condition(
                physical_noise=physical_noise,
                seed=seed,
            )

            results.append(result)

            print_single_result(
                result
            )

    summaries = aggregate(
        results
    )

    print_summary(
        summaries
    )

    print_interpretation(
        summaries
    )

    print(
        "=" * 78
    )

    print(
        " BASELINE VALIDATION COMPLETE"
    )

    print(
        "=" * 78
    )

    print()


if __name__ == "__main__":
    main()