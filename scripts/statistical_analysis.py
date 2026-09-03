from math import sqrt


# ============================================================
# VALIDATED BASELINE RESULTS
# ============================================================
#
# Three seeds:
#   42, 43, 44
#
# Physical-noise sweep:
#   measurement noise = 0.00
#
# These values come from the completed baseline experiment.
# ============================================================

BASELINE = {
    0.00: [1.0000, 1.0000, 1.0000],
    0.01: [0.9890, 0.9950, 0.9945],
    0.03: [0.9480, 0.9600, 0.9565],
    0.05: [0.8865, 0.9055, 0.9005],
    0.10: [0.7435, 0.7630, 0.7280],
    0.15: [0.6075, 0.6445, 0.5980],
    0.20: [0.5555, 0.5555, 0.5680],
}


# ============================================================
# VALIDATED AI-QEC RESULTS
# ============================================================
#
# Three seeds:
#   42, 43, 44
#
# Physical-noise sweep:
#   measurement noise = 0.00
# ============================================================

AI_QEC = {
    0.00: [1.0000, 1.0000, 1.0000],
    0.01: [0.9940, 0.9940, 0.9960],
    0.03: [0.9740, 0.9800, 0.9740],
    0.05: [0.9560, 0.9440, 0.9220],
    0.10: [0.8440, 0.8240, 0.8340],
    0.15: [0.7200, 0.7000, 0.7200],
    0.20: [0.5880, 0.5800, 0.6140],
}


# ============================================================
# NOISE MATRIX
# ============================================================

NOISE_MATRIX = {
    0.00: {
        0.00: 1.0000,
        0.05: 1.0000,
        0.10: 1.0000,
        0.15: 1.0000,
        0.20: 1.0000,
    },

    0.05: {
        0.00: 0.9407,
        0.05: 0.9073,
        0.10: 0.8853,
        0.15: 0.8813,
        0.20: 0.8707,
    },

    0.10: {
        0.00: 0.8340,
        0.05: 0.7627,
        0.10: 0.7340,
        0.15: 0.7073,
        0.20: 0.6953,
    },

    0.15: {
        0.00: 0.7133,
        0.05: 0.6720,
        0.10: 0.6427,
        0.15: 0.6173,
        0.20: 0.5947,
    },

    0.20: {
        0.00: 0.5940,
        0.05: 0.5587,
        0.10: 0.5347,
        0.15: 0.5373,
        0.20: 0.5233,
    },
}


# ============================================================
# STATISTICS
# ============================================================


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) <= 1:
        return 0.0

    average = mean(values)

    variance = sum(
        (value - average) ** 2
        for value in values
    ) / (len(values) - 1)

    return sqrt(variance)


def standard_error(values):
    return sample_std(values) / sqrt(len(values))


def confidence_interval_95(values):
    """
    Approximate 95% CI using the normal critical value 1.96.

    This is appropriate here as a simple descriptive
    three-seed interval. With only three independent seeds,
    it should NOT be treated as a definitive inferential test.
    """

    average = mean(values)

    margin = (
        1.96
        * standard_error(values)
    )

    return (
        average - margin,
        average + margin,
    )


def logical_error_rate(logical_success):
    return 1.0 - logical_success


def relative_improvement(
    baseline_success,
    ai_success,
):
    if baseline_success == 0:
        return 0.0

    return (
        (ai_success - baseline_success)
        / baseline_success
    )


def error_reduction(
    baseline_success,
    ai_success,
):
    baseline_error = (
        1.0 - baseline_success
    )

    ai_error = (
        1.0 - ai_success
    )

    if baseline_error == 0:
        return 0.0

    return (
        (baseline_error - ai_error)
        / baseline_error
    )


# ============================================================
# PHYSICAL NOISE COMPARISON
# ============================================================


def build_comparison():

    rows = []

    for noise in sorted(BASELINE):

        baseline_values = BASELINE[
            noise
        ]

        ai_values = AI_QEC[
            noise
        ]

        baseline_mean = mean(
            baseline_values
        )

        ai_mean = mean(
            ai_values
        )

        baseline_std = sample_std(
            baseline_values
        )

        ai_std = sample_std(
            ai_values
        )

        baseline_ci = (
            confidence_interval_95(
                baseline_values
            )
        )

        ai_ci = (
            confidence_interval_95(
                ai_values
            )
        )

        absolute_gain = (
            ai_mean
            - baseline_mean
        )

        relative_gain = (
            relative_improvement(
                baseline_mean,
                ai_mean,
            )
        )

        baseline_error = (
            logical_error_rate(
                baseline_mean
            )
        )

        ai_error = (
            logical_error_rate(
                ai_mean
            )
        )

        reduction = (
            error_reduction(
                baseline_mean,
                ai_mean,
            )
        )

        rows.append(
            {
                "noise": noise,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "baseline_ci_low": baseline_ci[0],
                "baseline_ci_high": baseline_ci[1],
                "ai_mean": ai_mean,
                "ai_std": ai_std,
                "ai_ci_low": ai_ci[0],
                "ai_ci_high": ai_ci[1],
                "absolute_gain": absolute_gain,
                "relative_gain": relative_gain,
                "baseline_error": baseline_error,
                "ai_error": ai_error,
                "error_reduction": reduction,
            }
        )

    return rows


# ============================================================
# PRINT MAIN COMPARISON
# ============================================================


def print_comparison(rows):

    print()

    print("=" * 120)
    print(
        " AI-QEC VS UNCORRECTED BASELINE"
    )
    print("=" * 120)

    print()

    print(
        f"{'Noise':<9}"
        f"{'Baseline':>13}"
        f"{'AI-QEC':>13}"
        f"{'Gain':>13}"
        f"{'Rel. Gain':>13}"
        f"{'Base Error':>13}"
        f"{'AI Error':>13}"
        f"{'Error Red.':>13}"
    )

    print("-" * 120)

    for row in rows:

        print(
            f"{row['noise']:<9.2f}"
            f"{row['baseline_mean']:>12.2%}"
            f"{row['ai_mean']:>12.2%}"
            f"{row['absolute_gain']:>12.2%}"
            f"{row['relative_gain']:>12.2%}"
            f"{row['baseline_error']:>12.2%}"
            f"{row['ai_error']:>12.2%}"
            f"{row['error_reduction']:>12.2%}"
        )

    print()


# ============================================================
# CONFIDENCE INTERVAL TABLE
# ============================================================


def print_confidence_intervals(rows):

    print("=" * 120)
    print(
        " 95% CONFIDENCE INTERVALS"
    )
    print("=" * 120)

    print()

    print(
        f"{'Noise':<9}"
        f"{'Baseline Mean':>18}"
        f"{'Baseline 95% CI':>25}"
        f"{'AI Mean':>15}"
        f"{'AI 95% CI':>25}"
    )

    print("-" * 120)

    for row in rows:

        baseline_ci = (
            f"[{row['baseline_ci_low']:.2%}, "
            f"{row['baseline_ci_high']:.2%}]"
        )

        ai_ci = (
            f"[{row['ai_ci_low']:.2%}, "
            f"{row['ai_ci_high']:.2%}]"
        )

        print(
            f"{row['noise']:<9.2f}"
            f"{row['baseline_mean']:>17.2%}"
            f"{baseline_ci:>25}"
            f"{row['ai_mean']:>14.2%}"
            f"{ai_ci:>25}"
        )

    print()


# ============================================================
# BEST OPERATING REGION
# ============================================================


def print_best_region(rows):

    non_zero = [
        row
        for row in rows
        if row["noise"] > 0
    ]

    best_gain = max(
        non_zero,
        key=lambda row: row[
            "absolute_gain"
        ],
    )

    best_error_reduction = max(
        non_zero,
        key=lambda row: row[
            "error_reduction"
        ],
    )

    print("=" * 120)
    print(
        " AI-QEC BENEFIT"
    )
    print("=" * 120)

    print()

    print(
        "Largest absolute logical-success gain:"
    )

    print(
        f"  Physical noise : "
        f"{best_gain['noise']:.2f}"
    )

    print(
        f"  Baseline       : "
        f"{best_gain['baseline_mean']:.2%}"
    )

    print(
        f"  AI-QEC         : "
        f"{best_gain['ai_mean']:.2%}"
    )

    print(
        f"  Gain           : "
        f"{best_gain['absolute_gain']:.2%}"
    )

    print()

    print(
        "Largest logical-error reduction:"
    )

    print(
        f"  Physical noise : "
        f"{best_error_reduction['noise']:.2f}"
    )

    print(
        f"  Baseline error : "
        f"{best_error_reduction['baseline_error']:.2%}"
    )

    print(
        f"  AI-QEC error   : "
        f"{best_error_reduction['ai_error']:.2%}"
    )

    print(
        f"  Error reduction: "
        f"{best_error_reduction['error_reduction']:.2%}"
    )

    print()


# ============================================================
# NOISE MATRIX
# ============================================================


def print_noise_matrix():

    measurement_levels = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    print("=" * 90)
    print(
        " COMBINED-NOISE LOGICAL SUCCESS"
    )
    print("=" * 90)

    print()

    print(
        "Rows = physical noise"
    )

    print(
        "Columns = measurement noise"
    )

    print()

    print(
        f"{'Physical':<14}",
        end="",
    )

    for measurement in (
        measurement_levels
    ):

        print(
            f"{measurement:>12.2f}",
            end="",
        )

    print()

    print("-" * 74)

    for physical in sorted(
        NOISE_MATRIX
    ):

        print(
            f"{physical:<14.2f}",
            end="",
        )

        for measurement in (
            measurement_levels
        ):

            value = NOISE_MATRIX[
                physical
            ][measurement]

            print(
                f"{value:>11.2%}",
                end="",
            )

        print()

    print()


# ============================================================
# FINAL INTERPRETATION
# ============================================================


def print_interpretation(rows):

    print("=" * 120)
    print(
        " SCIENTIFIC SUMMARY"
    )
    print("=" * 120)

    print()

    positive = [
        row
        for row in rows
        if row["absolute_gain"] > 0
    ]

    print(
        f"Non-zero physical-noise conditions "
        f"where AI-QEC outperformed baseline: "
        f"{len(positive)} / "
        f"{len(rows) - 1}"
    )

    print()

    for row in rows:

        if row["noise"] == 0:
            continue

        direction = (
            "BETTER"
            if row["absolute_gain"] > 0
            else "WORSE"
            if row["absolute_gain"] < 0
            else "EQUAL"
        )

        print(
            f"Noise {row['noise']:.2f}: "
            f"AI-QEC {direction} | "
            f"gain = "
            f"{row['absolute_gain']:+.2%}"
        )

    print()

    print(
        "Important:"
    )

    print(
        "The confidence intervals here are descriptive "
        "seed-level intervals. Only three seeds were used, "
        "so they should not be presented as definitive "
        "statistical significance tests."
    )

    print()

    print(
        "The strongest observed AI-QEC benefit occurs "
        "in the moderate physical-noise regime."
    )

    print(
        "At very high combined physical and measurement "
        "noise, logical recovery approaches the difficult "
        "regime where the current decoder loses robustness."
    )

    print()


# ============================================================
# MAIN
# ============================================================


def main():

    print()

    print(
        "Starting statistical analysis..."
    )

    rows = build_comparison()

    print_comparison(
        rows
    )

    print_confidence_intervals(
        rows
    )

    print_best_region(
        rows
    )

    print_noise_matrix()

    print_interpretation(
        rows
    )

    print("=" * 120)
    print(
        " STATISTICAL ANALYSIS COMPLETE"
    )
    print("=" * 120)

    print()


if __name__ == "__main__":
    main()