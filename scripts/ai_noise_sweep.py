from statistics import mean, stdev

from experiments.config import ExperimentConfig
from experiments.engine import ExperimentEngine


# ============================================================
# AI-QEC NOISE SWEEP
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
# RUN ONE EXPERIMENT
# ============================================================


def run_one_condition(
    physical_noise,
    seed,
):
    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        logical_state=None,
        rounds=ROUNDS,
        physical_noise_probability=physical_noise,
        measurement_noise_probability=0.0,
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
        "seed": seed,
        "logical_success": (
            result.logical_accuracy
        ),
        "physical_recovery": (
            result.physical_accuracy
        ),
        "exact_accuracy": (
            result.exact_accuracy
        ),
        "bit_accuracy": (
            result.bit_accuracy
        ),
        "training_seconds": (
            result.training_seconds
        ),
        "inference_seconds": (
            result.inference_seconds
        ),
        "samples_per_second": (
            result.samples_per_second
        ),
    }


# ============================================================
# AGGREGATE
# ============================================================


def aggregate(results):

    grouped = {}

    for result in results:

        noise = result[
            "physical_noise"
        ]

        grouped.setdefault(
            noise,
            []
        ).append(result)

    summaries = []

    for noise in sorted(grouped):

        rows = grouped[noise]

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

        summaries.append(
            {
                "physical_noise": noise,

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
        " AI-QEC RANDOM FOREST - PHYSICAL NOISE SWEEP"
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
        f"Training samples     : {TRAINING_SAMPLES}"
    )

    print(
        f"Test samples         : {TEST_SAMPLES}"
    )

    print(
        f"Measurement noise    : 0.00"
    )

    print(
        f"RF estimators        : "
        f"{RANDOM_FOREST_ESTIMATORS}"
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
        f"{result['logical_success']:.4f} | "
        f"Physical recovery = "
        f"{result['physical_recovery']:.4f}"
    )


def print_summary(summaries):

    print()

    print("=" * 78)
    print(
        " AI-QEC RESULTS"
    )
    print("=" * 78)

    print()

    print(
        f"{'Physical Noise':<18}"
        f"{'Logical Success':>18}"
        f"{'Std Dev':>15}"
        f"{'Physical':>15}"
    )

    print("-" * 78)

    for summary in summaries:

        print(
            f"{summary['physical_noise']:<18.2f}"
            f"{summary['logical_mean']:>17.2%}"
            f"{summary['logical_std']:>14.2%}"
            f"{summary['physical_mean']:>14.2%}"
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
                f"Running AI-QEC: "
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

    print()

    print("=" * 78)
    print(
        " AI-QEC NOISE SWEEP COMPLETE"
    )
    print("=" * 78)

    print()


if __name__ == "__main__":
    main()