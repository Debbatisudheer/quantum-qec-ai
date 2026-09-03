import statistics
import time

from experiments.config import ExperimentConfig
from experiments.engine import ExperimentEngine


# ============================================================
# SCIENTIFIC VALIDATION CONFIGURATION
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
# CONTROLLED NOISE REGIMES
# ============================================================

NOISE_REGIMES = [
    {
        "name": "Physical Noise Only",
        "physical_noise": 0.10,
        "measurement_noise": 0.00,
    },
    {
        "name": "Measurement Noise Only",
        "physical_noise": 0.00,
        "measurement_noise": 0.10,
    },
    {
        "name": "Combined Noise",
        "physical_noise": 0.10,
        "measurement_noise": 0.10,
    },
]


# ============================================================
# HELPERS
# ============================================================


def create_config(
    physical_noise,
    measurement_noise,
    seed,
):
    """
    Create one controlled ExperimentConfig.
    """

    return ExperimentConfig(
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


def run_single_experiment(
    regime,
    seed,
):
    """
    Run one experiment for one noise regime
    and one random seed.
    """

    config = create_config(
        physical_noise=regime[
            "physical_noise"
        ],
        measurement_noise=regime[
            "measurement_noise"
        ],
        seed=seed,
    )

    engine = ExperimentEngine(
        config=config,
        storage=None,
    )

    start = time.perf_counter()

    result = engine.run()

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "regime": regime["name"],
        "seed": seed,

        "physical_noise": (
            regime["physical_noise"]
        ),

        "measurement_noise": (
            regime["measurement_noise"]
        ),

        "exact_accuracy": (
            result.exact_accuracy
        ),

        "physical_accuracy": (
            result.physical_accuracy
        ),

        "bit_accuracy": (
            result.bit_accuracy
        ),

        "logical_accuracy": (
            result.logical_accuracy
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

        "total_seconds": elapsed,
    }


def mean(values):
    """
    Safe arithmetic mean.
    """

    if not values:
        return 0.0

    return statistics.mean(values)


def standard_deviation(values):
    """
    Sample standard deviation.

    Returns 0 when only one value exists.
    """

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def aggregate_results(results):
    """
    Aggregate repeated-seed results by noise regime.
    """

    grouped = {}

    for result in results:
        regime = result["regime"]

        if regime not in grouped:
            grouped[regime] = []

        grouped[regime].append(
            result
        )

    summaries = []

    for regime_name, rows in grouped.items():

        logical_values = [
            row["logical_accuracy"]
            for row in rows
        ]

        physical_values = [
            row["physical_accuracy"]
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
                "regime": regime_name,

                "physical_noise": rows[0][
                    "physical_noise"
                ],

                "measurement_noise": rows[0][
                    "measurement_noise"
                ],

                "logical_mean": mean(
                    logical_values
                ),

                "logical_std": standard_deviation(
                    logical_values
                ),

                "physical_mean": mean(
                    physical_values
                ),

                "physical_std": standard_deviation(
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
            }
        )

    return summaries


# ============================================================
# PRINT HELPERS
# ============================================================


def print_header():
    print()
    print("=" * 78)
    print(
        " SCIENTIFIC VALIDATION - CONTROLLED NOISE"
    )
    print("=" * 78)

    print()

    print(
        f"QEC code                  : bit_flip_3"
    )

    print(
        f"Qubits                    : 3"
    )

    print(
        f"Rounds                    : {ROUNDS}"
    )

    print(
        f"Training samples          : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Random Forest estimators  : "
        f"{RANDOM_FOREST_ESTIMATORS}"
    )

    print(
        f"Seeds                     : "
        f"{SEEDS}"
    )

    print()


def print_single_result(result):
    print()
    print("-" * 78)

    print(
        f"REGIME: {result['regime']}"
    )

    print(
        f"Seed                    : "
        f"{result['seed']}"
    )

    print(
        f"Physical noise          : "
        f"{result['physical_noise']:.2f}"
    )

    print(
        f"Measurement noise       : "
        f"{result['measurement_noise']:.2f}"
    )

    print()

    print(
        f"Exact error accuracy    : "
        f"{result['exact_accuracy']:.4f}"
    )

    print(
        f"Physical recovery       : "
        f"{result['physical_accuracy']:.4f}"
    )

    print(
        f"Bit accuracy            : "
        f"{result['bit_accuracy']:.4f}"
    )

    print(
        f"Logical success         : "
        f"{result['logical_accuracy']:.4f}"
    )

    print()

    print(
        f"Training time           : "
        f"{result['training_seconds']:.4f}s"
    )

    print(
        f"Inference time          : "
        f"{result['inference_seconds']:.4f}s"
    )

    print(
        f"Throughput              : "
        f"{result['samples_per_second']:.2f} "
        f"samples/s"
    )


def print_summary(summaries):
    print()
    print()
    print("=" * 78)
    print(
        " AGGREGATED SCIENTIFIC RESULTS"
    )
    print("=" * 78)

    print()

    header = (
        f"{'Regime':<25}"
        f"{'Logical':>12}"
        f"{'Physical':>12}"
        f"{'Exact':>12}"
        f"{'Bit':>12}"
    )

    print(header)

    print("-" * 78)

    for summary in summaries:

        logical = (
            summary["logical_mean"]
        )

        physical = (
            summary["physical_mean"]
        )

        exact = (
            summary["exact_mean"]
        )

        bit = (
            summary["bit_mean"]
        )

        print(
            f"{summary['regime']:<25}"
            f"{logical:>11.2%}"
            f"{physical:>11.2%}"
            f"{exact:>11.2%}"
            f"{bit:>11.2%}"
        )

    print()


def print_detailed_summary(summaries):
    print()
    print("=" * 78)
    print(
        " MEAN ± STANDARD DEVIATION"
    )
    print("=" * 78)

    for summary in summaries:

        print()

        print(
            summary["regime"]
        )

        print(
            f"  Physical noise       : "
            f"{summary['physical_noise']:.2f}"
        )

        print(
            f"  Measurement noise    : "
            f"{summary['measurement_noise']:.2f}"
        )

        print()

        print(
            f"  Logical success      : "
            f"{summary['logical_mean']:.4f} "
            f"± {summary['logical_std']:.4f}"
        )

        print(
            f"  Physical recovery    : "
            f"{summary['physical_mean']:.4f} "
            f"± {summary['physical_std']:.4f}"
        )

        print(
            f"  Exact error accuracy : "
            f"{summary['exact_mean']:.4f}"
        )

        print(
            f"  Bit accuracy         : "
            f"{summary['bit_mean']:.4f}"
        )

        print(
            f"  Inference time       : "
            f"{summary['inference_mean']:.4f}s"
        )

        print(
            f"  Throughput           : "
            f"{summary['throughput_mean']:.2f} samples/s"
        )


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================


def print_interpretation(summaries):
    """
    Explain what the controlled experiments mean.

    This does not declare a decoder winner.
    It simply reports the behavior of the current
    decoder under the three controlled regimes.
    """

    print()
    print("=" * 78)
    print(
        " SCIENTIFIC INTERPRETATION"
    )
    print("=" * 78)

    print()

    print(
        "1. PHYSICAL NOISE ONLY"
    )

    print(
        "   Tests whether the decoder can recover "
        "logical information when physical X errors "
        "are present but syndrome measurements are "
        "perfect."
    )

    print()

    print(
        "2. MEASUREMENT NOISE ONLY"
    )

    print(
        "   Tests whether the decoder can tolerate "
        "incorrect/noisy syndrome observations "
        "when physical X noise is absent."
    )

    print()

    print(
        "3. COMBINED NOISE"
    )

    print(
        "   Tests the complete learning problem where "
        "both the physical qubit state and the "
        "syndrome observations are noisy."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Logical success is the primary QEC outcome."
    )

    print(
        "Exact physical-error prediction and physical "
        "state recovery are reported separately."
    )

    print(
        "A decoder can fail to predict the exact "
        "physical error while still preserving the "
        "logical qubit."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print_header()

    all_results = []

    experiment_number = 0

    total_experiments = (
        len(NOISE_REGIMES)
        * len(SEEDS)
    )

    for regime in NOISE_REGIMES:

        for seed in SEEDS:

            experiment_number += 1

            print()
            print(
                f"[{experiment_number}/"
                f"{total_experiments}] "
                f"Running {regime['name']} "
                f"with seed {seed}..."
            )

            result = run_single_experiment(
                regime=regime,
                seed=seed,
            )

            all_results.append(
                result
            )

            print_single_result(
                result
            )

    summaries = aggregate_results(
        all_results
    )

    print_summary(
        summaries
    )

    print_detailed_summary(
        summaries
    )

    print_interpretation(
        summaries
    )

    print()
    print("=" * 78)
    print(
        " SCIENTIFIC VALIDATION COMPLETE"
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()