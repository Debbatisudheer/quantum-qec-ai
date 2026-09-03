from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, median, stdev

from experiments.config import ExperimentConfig
from experiments.engine import ExperimentEngine


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ROUNDS = 5

TRAINING_SAMPLES = 2000
TEST_SAMPLES = 2000

RANDOM_FOREST_ESTIMATORS = 100

SEEDS = [42, 43, 44]

PHYSICAL_NOISE_LEVELS = [
    0.01,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
]

MEASUREMENT_NOISE = 0.00

BOOTSTRAP_ITERATIONS = 10000

OUTPUT_DIRECTORY = (
    Path("experiments")
    / "paired_results"
)


# ============================================================
# BASIC QUANTUM LOGICAL RECOVERY
# ============================================================

def recover_logical(state: str) -> int:
    """
    Majority-vote recovery for the 3-qubit repetition code.

    000 / 001 / 010 / 100 -> logical 0
    111 / 110 / 101 / 011 -> logical 1
    """

    if len(state) != 3:
        raise ValueError(
            f"Expected 3-bit state, got: {state}"
        )

    ones = state.count("1")

    return 1 if ones >= 2 else 0


# ============================================================
# BASELINE SUCCESS
# ============================================================

def baseline_logical_success(sample: dict) -> bool:
    """
    Baseline:
        encoded state
            ↓
        physical noise
            ↓
        corrupted state
            ↓
        majority vote

    No syndrome decoding.
    No AI.
    No correction.
    """

    logical_state = int(
        sample["logical_state"]
    )

    encoded_state = (
        "111"
        if logical_state == 1
        else "000"
    )

    error_bits = sample[
        "final_error_state"
    ]

    corrupted_state = "".join(
        str(
            int(encoded_bit)
            ^ int(error_bit)
        )
        for encoded_bit, error_bit
        in zip(
            encoded_state,
            error_bits,
        )
    )

    recovered = recover_logical(
        corrupted_state
    )

    return recovered == logical_state


# ============================================================
# AI-QEC SUCCESS
# ============================================================

def ai_qec_logical_success(
    sample: dict,
    decoder,
) -> tuple[bool, list[int]]:
    """
    AI-QEC:
        syndrome information
            ↓
        Random Forest decoder
            ↓
        predicted correction
            ↓
        correction applied conceptually
            ↓
        logical recovery
    """

    predicted_correction = decoder.decode(
        sample
    )

    actual_error = [
        int(bit)
        for bit in sample[
            "final_error_state"
        ]
    ]

    corrected_error = [
        actual ^ predicted
        for actual, predicted
        in zip(
            actual_error,
            predicted_correction,
        )
    ]

    residual_error = "".join(
        str(bit)
        for bit in corrected_error
    )

    recovered = recover_logical(
        residual_error
    )

    logical_state = int(
        sample["logical_state"]
    )

    success = recovered == 0

    if logical_state == 1:
        # For logical 1, correction must preserve
        # the logical state rather than force logical 0.
        #
        # The cleanest check is to reconstruct the
        # corrected physical state.
        encoded_state = "111"

        corrected_state = "".join(
            str(
                int(encoded_bit)
                ^ int(residual_bit)
            )
            for encoded_bit, residual_bit
            in zip(
                encoded_state,
                residual_error,
            )
        )

        recovered = recover_logical(
            corrected_state
        )

        success = (
            recovered == logical_state
        )

    return success, predicted_correction


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_mean_difference(
    differences: list[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 123456,
) -> tuple[float, float]:

    if not differences:
        return 0.0, 0.0

    rng = random.Random(seed)

    n = len(differences)

    bootstrap_means = []

    for _ in range(iterations):

        sample = [
            differences[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        bootstrap_means.append(
            mean(sample)
        )

    bootstrap_means.sort()

    lower_index = int(
        0.025 * iterations
    )

    upper_index = int(
        0.975 * iterations
    )

    upper_index = min(
        upper_index,
        iterations - 1,
    )

    return (
        bootstrap_means[lower_index],
        bootstrap_means[upper_index],
    )


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

def paired_permutation_test(
    differences: list[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 987654,
) -> float:

    if not differences:
        return 1.0

    observed = mean(
        differences
    )

    rng = random.Random(seed)

    count = 0

    n = len(differences)

    for _ in range(iterations):

        randomized_sum = 0.0

        for difference in differences:

            if rng.random() < 0.5:
                randomized_sum += difference
            else:
                randomized_sum -= difference

        randomized_mean = (
            randomized_sum / n
        )

        if abs(randomized_mean) >= abs(observed):
            count += 1

    return (
        count + 1
    ) / (
        iterations + 1
    )


# ============================================================
# RUN ONE PAIRED EXPERIMENT
# ============================================================

def run_condition(
    physical_noise: float,
    seed: int,
) -> dict:

    config = ExperimentConfig(
        qec_code="bit_flip_3",
        num_qubits=3,
        logical_state=None,
        rounds=ROUNDS,
        physical_noise_probability=(
            physical_noise
        ),
        measurement_noise_probability=(
            MEASUREMENT_NOISE
        ),
        training_samples=(
            TRAINING_SAMPLES
        ),
        test_samples=(
            TEST_SAMPLES
        ),
        decoder_type=(
            "logical_target_random_forest"
        ),
        random_forest_estimators=(
            RANDOM_FOREST_ESTIMATORS
        ),
        seed=seed,
    )

    config.validate()

    engine = ExperimentEngine(
        config,
        storage=None,
    )

    # --------------------------------------------------------
    # Generate training data
    # --------------------------------------------------------

    training_samples = (
        engine.generate_samples(
            TRAINING_SAMPLES,
            seed,
        )
    )

    # --------------------------------------------------------
    # Generate ONE fixed held-out test set
    # --------------------------------------------------------

    test_samples = (
        engine.generate_samples(
            TEST_SAMPLES,
            seed + 10000,
        )
    )

    # --------------------------------------------------------
    # Train AI decoder
    # --------------------------------------------------------

    decoder = engine.create_decoder()

    decoder.train(
        training_samples
    )

    baseline_results = []
    ai_results = []

    paired_differences = []

    exact_matches = []

    for sample in test_samples:

        # -------------------------------
        # Baseline
        # -------------------------------

        baseline_success = (
            baseline_logical_success(
                sample
            )
        )

        # -------------------------------
        # AI-QEC
        # -------------------------------

        ai_success, predicted = (
            ai_qec_logical_success(
                sample,
                decoder,
            )
        )

        baseline_value = (
            1.0
            if baseline_success
            else 0.0
        )

        ai_value = (
            1.0
            if ai_success
            else 0.0
        )

        baseline_results.append(
            baseline_value
        )

        ai_results.append(
            ai_value
        )

        paired_differences.append(
            ai_value
            - baseline_value
        )

        actual = [
            int(bit)
            for bit in sample[
                "final_error_state"
            ]
        ]

        exact_matches.append(
            predicted == actual
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    baseline_mean = mean(
        baseline_results
    )

    ai_mean = mean(
        ai_results
    )

    difference_mean = mean(
        paired_differences
    )

    difference_std = (
        stdev(paired_differences)
        if len(paired_differences) > 1
        else 0.0
    )

    bootstrap_low, bootstrap_high = (
        bootstrap_mean_difference(
            paired_differences,
            seed=seed,
        )
    )

    permutation_p = (
        paired_permutation_test(
            paired_differences,
            seed=seed + 500000,
        )
    )

    # --------------------------------------------------------
    # Discordant pairs
    # --------------------------------------------------------

    baseline_better = sum(
        1
        for baseline, ai in zip(
            baseline_results,
            ai_results,
        )
        if baseline == 1.0
        and ai == 0.0
    )

    ai_better = sum(
        1
        for baseline, ai in zip(
            baseline_results,
            ai_results,
        )
        if baseline == 0.0
        and ai == 1.0
    )

    both_success = sum(
        1
        for baseline, ai in zip(
            baseline_results,
            ai_results,
        )
        if baseline == 1.0
        and ai == 1.0
    )

    both_failure = sum(
        1
        for baseline, ai in zip(
            baseline_results,
            ai_results,
        )
        if baseline == 0.0
        and ai == 0.0
    )

    exact_accuracy = mean(
        [
            1.0
            if value
            else 0.0
            for value in exact_matches
        ]
    )

    return {
        "physical_noise": physical_noise,
        "measurement_noise": MEASUREMENT_NOISE,
        "seed": seed,
        "training_samples": TRAINING_SAMPLES,
        "test_samples": TEST_SAMPLES,
        "baseline_logical_success": baseline_mean,
        "ai_logical_success": ai_mean,
        "paired_gain": difference_mean,
        "paired_difference_std": difference_std,
        "bootstrap_ci_low": bootstrap_low,
        "bootstrap_ci_high": bootstrap_high,
        "permutation_p_value": permutation_p,
        "baseline_better_count": baseline_better,
        "ai_better_count": ai_better,
        "both_success_count": both_success,
        "both_failure_count": both_failure,
        "exact_accuracy": exact_accuracy,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results: list[dict]):

    print()
    print("=" * 120)
    print(
        " PAIRED BASELINE VS AI-QEC COMPARISON"
    )
    print("=" * 120)
    print()

    print(
        "Both systems use the SAME held-out "
        "test samples for every seed/noise condition."
    )

    print()

    print(
        f"{'Noise':<9}"
        f"{'Seed':<7}"
        f"{'Baseline':>13}"
        f"{'AI-QEC':>13}"
        f"{'Gain':>13}"
        f"{'Bootstrap 95% CI':>28}"
        f"{'p-value':>12}"
    )

    print("-" * 120)

    for result in results:

        ci = (
            f"["
            f"{result['bootstrap_ci_low']:.2%}, "
            f"{result['bootstrap_ci_high']:.2%}"
            f"]"
        )

        print(
            f"{result['physical_noise']:<9.2f}"
            f"{result['seed']:<7}"
            f"{result['baseline_logical_success']:>12.2%}"
            f"{result['ai_logical_success']:>12.2%}"
            f"{result['paired_gain']:>12.2%}"
            f"{ci:>28}"
            f"{result['permutation_p_value']:>12.5f}"
        )

    print()


# ============================================================
# SEED AGGREGATION
# ============================================================

def print_seed_aggregate(
    results: list[dict]
):

    print("=" * 120)
    print(
        " AGGREGATED PAIRED RESULTS"
    )
    print("=" * 120)
    print()

    noises = sorted(
        set(
            result[
                "physical_noise"
            ]
            for result in results
        )
    )

    print(
        f"{'Noise':<9}"
        f"{'Baseline':>14}"
        f"{'AI-QEC':>14}"
        f"{'Gain':>14}"
        f"{'Gain Std':>14}"
        f"{'AI Wins':>12}"
        f"{'Base Wins':>12}"
    )

    print("-" * 100)

    for noise in noises:

        rows = [
            result
            for result in results
            if result[
                "physical_noise"
            ] == noise
        ]

        baseline_values = [
            result[
                "baseline_logical_success"
            ]
            for result in rows
        ]

        ai_values = [
            result[
                "ai_logical_success"
            ]
            for result in rows
        ]

        gains = [
            result[
                "paired_gain"
            ]
            for result in rows
        ]

        ai_wins = sum(
            result[
                "ai_better_count"
            ]
            for result in rows
        )

        baseline_wins = sum(
            result[
                "baseline_better_count"
            ]
            for result in rows
        )

        gain_std = (
            stdev(gains)
            if len(gains) > 1
            else 0.0
        )

        print(
            f"{noise:<9.2f}"
            f"{mean(baseline_values):>13.2%}"
            f"{mean(ai_values):>13.2%}"
            f"{mean(gains):>13.2%}"
            f"{gain_std:>13.2%}"
            f"{ai_wins:>12}"
            f"{baseline_wins:>12}"
        )

    print()


# ============================================================
# SAVE JSON
# ============================================================

def save_results(
    results: list[dict]
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "paired_baseline_ai_results.json"
    )

    payload = {
        "experiment": {
            "rounds": ROUNDS,
            "training_samples": (
                TRAINING_SAMPLES
            ),
            "test_samples": (
                TEST_SAMPLES
            ),
            "random_forest_estimators": (
                RANDOM_FOREST_ESTIMATORS
            ),
            "seeds": SEEDS,
            "physical_noise_levels": (
                PHYSICAL_NOISE_LEVELS
            ),
            "measurement_noise": (
                MEASUREMENT_NOISE
            ),
            "bootstrap_iterations": (
                BOOTSTRAP_ITERATIONS
            ),
        },
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "Starting paired baseline vs AI-QEC experiment..."
    )
    print()

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
                f"physical_noise="
                f"{physical_noise:.2f}, "
                f"seed={seed}"
            )

            result = run_condition(
                physical_noise,
                seed,
            )

            results.append(
                result
            )

            print(
                f"    baseline="
                f"{result['baseline_logical_success']:.2%} | "
                f"AI-QEC="
                f"{result['ai_logical_success']:.2%} | "
                f"gain="
                f"{result['paired_gain']:+.2%}"
            )

    print_results(
        results
    )

    print_seed_aggregate(
        results
    )

    output_path = save_results(
        results
    )

    print("=" * 120)
    print(
        "RESULTS SAVED"
    )
    print("=" * 120)

    print()
    print(output_path)
    print()

    print(
        "Paired experiment complete."
    )
    print()


if __name__ == "__main__":
    main()