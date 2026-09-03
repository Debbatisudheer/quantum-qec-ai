from statistics import mean, stdev

from experiments.config import ExperimentConfig
from experiments.engine import ExperimentEngine


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ROUNDS = 5

TRAINING_SAMPLES = 2000
TEST_SAMPLES = 500

RANDOM_FOREST_ESTIMATORS = 100

SEEDS = [
    42,
    43,
    44,
]


# ============================================================
# NOISE GRID
# ============================================================

PHYSICAL_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

MEASUREMENT_NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================


def run_one_condition(
    physical_noise,
    measurement_noise,
    seed,
):
    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        logical_state=None,
        rounds=ROUNDS,
        physical_noise_probability=physical_noise,
        measurement_noise_probability=measurement_noise,
        training_samples=TRAINING_SAMPLES,
        test_samples=TEST_SAMPLES,
        decoder_type="logical_target_random_forest",
        random_forest_estimators=(
            RANDOM_FOREST_ESTIMATORS
        ),
        seed=seed,
    )

    engine = ExperimentEngine(
        config=config,
        storage=None,
    )

    result = engine.run()

    return {
        "physical_noise": physical_noise,
        "measurement_noise": measurement_noise,
        "seed": seed,
        "logical_success": result.logical_accuracy,
        "physical_recovery": result.physical_accuracy,
        "exact_accuracy": result.exact_accuracy,
        "bit_accuracy": result.bit_accuracy,
        "training_seconds": result.training_seconds,
        "inference_seconds": result.inference_seconds,
        "samples_per_second": result.samples_per_second,
    }


# ============================================================
# AGGREGATION
# ============================================================


def aggregate(results):
    """
    Aggregate the three seeds for each
    physical-noise / measurement-noise pair.
    """

    grouped = {}

    for result in results:

        key = (
            result["physical_noise"],
            result["measurement_noise"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(result)

    summaries = []

    for key in sorted(grouped):

        physical_noise, measurement_noise = key

        rows = grouped[key]

        logical_values = [
            row["logical_success"]
            for row in rows
        ]

        physical_values = [
            row["physical_recovery"]
            for row in rows
        ]

        exact_values = [
            row["exact_accuracy"]
            for row in rows
        ]

        bit_values = [
            row["bit_accuracy"]
            for row in rows
        ]

        inference_values = [
            row["inference_seconds"]
            for row in rows
        ]

        throughput_values = [
            row["samples_per_second"]
            for row in rows
        ]

        summaries.append(
            {
                "physical_noise": physical_noise,
                "measurement_noise": measurement_noise,

                "logical_mean": mean(
                    logical_values
                ),

                "logical_std": (
                    stdev(logical_values)
                    if len(logical_values) > 1
                    else 0.0
                ),

                "physical_mean": mean(
                    physical_values
                ),

                "exact_mean": mean(
                    exact_values
                ),

                "bit_mean": mean(
                    bit_values
                ),

                "inference_mean": mean(
                    inference_values
                ),

                "throughput_mean": mean(
                    throughput_values
                ),

                "runs": len(rows),
            }
        )

    return summaries


# ============================================================
# PRINT HEADER
# ============================================================


def print_header():

    print()

    print("=" * 90)
    print(
        " AI-QEC PHYSICAL × MEASUREMENT NOISE "
        "ROBUSTNESS MATRIX"
    )
    print("=" * 90)

    print()

    print(
        "QEC code             : bit_flip_3"
    )

    print(
        f"Rounds               : {ROUNDS}"
    )

    print(
        f"Training samples     : {TRAINING_SAMPLES}"
    )

    print(
        f"Test samples         : {TEST_SAMPLES}"
    )

    print(
        f"RF estimators        : "
        f"{RANDOM_FOREST_ESTIMATORS}"
    )

    print(
        f"Seeds                : {SEEDS}"
    )

    print()

    print(
        "Physical noise levels:"
    )

    print(
        PHYSICAL_NOISE_LEVELS
    )

    print()

    print(
        "Measurement noise levels:"
    )

    print(
        MEASUREMENT_NOISE_LEVELS
    )

    print()

    total = (
        len(PHYSICAL_NOISE_LEVELS)
        * len(MEASUREMENT_NOISE_LEVELS)
        * len(SEEDS)
    )

    print(
        f"Total experiments     : {total}"
    )

    print()


# ============================================================
# PRINT SINGLE RESULT
# ============================================================


def print_single_result(result):

    print(
        f"Physical = "
        f"{result['physical_noise']:.2f} | "
        f"Measurement = "
        f"{result['measurement_noise']:.2f} | "
        f"Seed = "
        f"{result['seed']} | "
        f"Logical = "
        f"{result['logical_success']:.4f} | "
        f"Physical recovery = "
        f"{result['physical_recovery']:.4f}"
    )


# ============================================================
# PRINT LONG-FORM RESULTS
# ============================================================


def print_detailed_results(summaries):

    print()

    print("=" * 90)
    print(
        " DETAILED AGGREGATED RESULTS"
    )
    print("=" * 90)

    print()

    print(
        f"{'Physical':<12}"
        f"{'Measurement':<15}"
        f"{'Logical':<15}"
        f"{'Std Dev':<13}"
        f"{'Physical Rec.':<16}"
        f"{'Bit Accuracy':<15}"
    )

    print("-" * 90)

    for summary in summaries:

        print(
            f"{summary['physical_noise']:<12.2f}"
            f"{summary['measurement_noise']:<15.2f}"
            f"{summary['logical_mean']:<15.2%}"
            f"{summary['logical_std']:<13.2%}"
            f"{summary['physical_mean']:<16.2%}"
            f"{summary['bit_mean']:<15.2%}"
        )

    print()


# ============================================================
# PRINT LOGICAL SUCCESS MATRIX
# ============================================================


def print_logical_matrix(summaries):

    lookup = {}

    for summary in summaries:

        key = (
            summary["physical_noise"],
            summary["measurement_noise"],
        )

        lookup[key] = summary[
            "logical_mean"
        ]

    print()

    print("=" * 90)
    print(
        " LOGICAL SUCCESS MATRIX"
    )
    print("=" * 90)

    print()

    print(
        "Rows    = Physical noise"
    )

    print(
        "Columns = Measurement noise"
    )

    print()

    print(
        f"{'Physical':<14}",
        end="",
    )

    for measurement_noise in (
        MEASUREMENT_NOISE_LEVELS
    ):

        print(
            f"{measurement_noise:>12.2f}",
            end="",
        )

    print()

    print("-" * 78)

    for physical_noise in (
        PHYSICAL_NOISE_LEVELS
    ):

        print(
            f"{physical_noise:<14.2f}",
            end="",
        )

        for measurement_noise in (
            MEASUREMENT_NOISE_LEVELS
        ):

            value = lookup[
                (
                    physical_noise,
                    measurement_noise,
                )
            ]

            print(
                f"{value:>11.2%}",
                end="",
            )

        print()

    print()


# ============================================================
# FIND BEST / WORST
# ============================================================


def print_extremes(summaries):

    if not summaries:
        return

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

    print()

    print("=" * 90)
    print(
        " EXTREME CONDITIONS"
    )
    print("=" * 90)

    print()

    print(
        "Best logical recovery:"
    )

    print(
        f"  Physical noise     : "
        f"{best['physical_noise']:.2f}"
    )

    print(
        f"  Measurement noise  : "
        f"{best['measurement_noise']:.2f}"
    )

    print(
        f"  Logical success    : "
        f"{best['logical_mean']:.2%}"
    )

    print()

    print(
        "Worst logical recovery:"
    )

    print(
        f"  Physical noise     : "
        f"{worst['physical_noise']:.2f}"
    )

    print(
        f"  Measurement noise  : "
        f"{worst['measurement_noise']:.2f}"
    )

    print(
        f"  Logical success    : "
        f"{worst['logical_mean']:.2%}"
    )

    print()


# ============================================================
# INTERPRETATION
# ============================================================


def print_interpretation(summaries):

    print("=" * 90)
    print(
        " SCIENTIFIC INTERPRETATION"
    )
    print("=" * 90)

    print()

    print(
        "This experiment measures AI-QEC robustness "
        "when both physical errors and syndrome/"
        "measurement errors are present."
    )

    print()

    print(
        "The primary metric is logical success."
    )

    print()

    print(
        "Higher logical success means the decoder "
        "preserved the logical qubit more often."
    )

    print()

    print(
        "Physical recovery and exact accuracy are "
        "supporting metrics and should not be "
        "confused with logical success."
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
            f"Observed best condition: "
            f"physical={best['physical_noise']:.2f}, "
            f"measurement="
            f"{best['measurement_noise']:.2f}, "
            f"logical success="
            f"{best['logical_mean']:.2%}"
        )

        print(
            f"Observed worst condition: "
            f"physical={worst['physical_noise']:.2f}, "
            f"measurement="
            f"{worst['measurement_noise']:.2f}, "
            f"logical success="
            f"{worst['logical_mean']:.2%}"
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
        * len(MEASUREMENT_NOISE_LEVELS)
        * len(SEEDS)
    )

    current = 0

    for physical_noise in (
        PHYSICAL_NOISE_LEVELS
    ):

        for measurement_noise in (
            MEASUREMENT_NOISE_LEVELS
        ):

            for seed in SEEDS:

                current += 1

                print(
                    f"[{current}/{total}] "
                    f"Running: "
                    f"physical="
                    f"{physical_noise:.2f}, "
                    f"measurement="
                    f"{measurement_noise:.2f}, "
                    f"seed={seed}"
                )

                result = run_one_condition(
                    physical_noise=physical_noise,
                    measurement_noise=(
                        measurement_noise
                    ),
                    seed=seed,
                )

                results.append(result)

                print_single_result(
                    result
                )

    summaries = aggregate(
        results
    )

    print_detailed_results(
        summaries
    )

    print_logical_matrix(
        summaries
    )

    print_extremes(
        summaries
    )

    print_interpretation(
        summaries
    )

    print("=" * 90)
    print(
        " NOISE ROBUSTNESS MATRIX COMPLETE"
    )
    print("=" * 90)

    print()


if __name__ == "__main__":
    main()