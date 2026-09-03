from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULT_PATH = (
    Path("experiments")
    / "paired_results"
    / "paired_baseline_ai_results.json"
)

OUTPUT_DIRECTORY = (
    Path("experiments")
    / "paired_results"
    / "plots"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Paired result file not found: "
            f"{RESULT_PATH}"
        )

    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if "results" not in data:
        raise ValueError(
            "Invalid result file: "
            "'results' field is missing."
        )

    return data["results"]


# ============================================================
# AGGREGATE BY PHYSICAL NOISE
# ============================================================

def aggregate_results(results):

    noise_levels = sorted(
        {
            result["physical_noise"]
            for result in results
        }
    )

    aggregated = []

    for noise in noise_levels:

        rows = [
            result
            for result in results
            if result["physical_noise"] == noise
        ]

        baseline = (
            sum(
                row[
                    "baseline_logical_success"
                ]
                for row in rows
            )
            / len(rows)
        )

        ai = (
            sum(
                row[
                    "ai_logical_success"
                ]
                for row in rows
            )
            / len(rows)
        )

        gain = (
            sum(
                row["paired_gain"]
                for row in rows
            )
            / len(rows)
        )

        bootstrap_low = (
            sum(
                row[
                    "bootstrap_ci_low"
                ]
                for row in rows
            )
            / len(rows)
        )

        bootstrap_high = (
            sum(
                row[
                    "bootstrap_ci_high"
                ]
                for row in rows
            )
            / len(rows)
        )

        aggregated.append(
            {
                "noise": noise,
                "baseline": baseline,
                "ai": ai,
                "gain": gain,
                "bootstrap_low": (
                    bootstrap_low
                ),
                "bootstrap_high": (
                    bootstrap_high
                ),
            }
        )

    return aggregated


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def prepare_output():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# FIGURE 1
# LOGICAL SUCCESS VS PHYSICAL NOISE
# ============================================================

def plot_logical_success(data):

    noise = [
        row["noise"]
        for row in data
    ]

    baseline = [
        row["baseline"]
        for row in data
    ]

    ai = [
        row["ai"]
        for row in data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        noise,
        baseline,
        marker="o",
        linewidth=2,
        label="Uncorrected baseline",
    )

    plt.plot(
        noise,
        ai,
        marker="o",
        linewidth=2,
        label="AI-QEC",
    )

    plt.xlabel(
        "Physical Noise Probability"
    )

    plt.ylabel(
        "Logical Success Rate"
    )

    plt.title(
        "Logical Success vs Physical Noise"
    )

    plt.xticks(noise)

    plt.ylim(
        0.50,
        1.02,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIRECTORY
        / "01_logical_success_vs_noise.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output


# ============================================================
# FIGURE 2
# AI-QEC GAIN
# ============================================================

def plot_gain(data):

    noise = [
        row["noise"]
        for row in data
    ]

    gain = [
        row["gain"]
        for row in data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        noise,
        gain,
        marker="o",
        linewidth=2,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Physical Noise Probability"
    )

    plt.ylabel(
        "AI-QEC Gain"
    )

    plt.title(
        "Paired AI-QEC Improvement Over Baseline"
    )

    plt.xticks(noise)

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIRECTORY
        / "02_paired_ai_gain.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output


# ============================================================
# FIGURE 3
# GAIN + BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def plot_gain_confidence_interval(data):

    noise = [
        row["noise"]
        for row in data
    ]

    gain = [
        row["gain"]
        for row in data
    ]

    lower_error = [
        gain_value
        - row["bootstrap_low"]
        for gain_value, row
        in zip(gain, data)
    ]

    upper_error = [
        row["bootstrap_high"]
        - gain_value
        for gain_value, row
        in zip(gain, data)
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.errorbar(
        noise,
        gain,
        yerr=[
            lower_error,
            upper_error,
        ],
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=5,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Physical Noise Probability"
    )

    plt.ylabel(
        "Paired Logical-Success Difference"
    )

    plt.title(
        "AI-QEC Gain with Bootstrap 95% Intervals"
    )

    plt.xticks(noise)

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    output = (
        OUTPUT_DIRECTORY
        / "03_paired_gain_bootstrap_ci.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output


# ============================================================
# FIGURE 4
# LOGICAL ERROR RATE
# ============================================================

def plot_logical_error(data):

    noise = [
        row["noise"]
        for row in data
    ]

    baseline_error = [
        1.0 - row["baseline"]
        for row in data
    ]

    ai_error = [
        1.0 - row["ai"]
        for row in data
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        noise,
        baseline_error,
        marker="o",
        linewidth=2,
        label="Uncorrected baseline",
    )

    plt.plot(
        noise,
        ai_error,
        marker="o",
        linewidth=2,
        label="AI-QEC",
    )

    plt.xlabel(
        "Physical Noise Probability"
    )

    plt.ylabel(
        "Logical Error Rate"
    )

    plt.title(
        "Logical Error Rate vs Physical Noise"
    )

    plt.xticks(noise)

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    output = (
        OUTPUT_DIRECTORY
        / "04_logical_error_rate.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output


# ============================================================
# FIGURE 5
# COMBINED NOISE HEATMAP
# ============================================================

def plot_noise_matrix():

    physical_levels = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    measurement_levels = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    matrix = [
        [
            1.0000,
            1.0000,
            1.0000,
            1.0000,
            1.0000,
        ],
        [
            0.9407,
            0.9073,
            0.8853,
            0.8813,
            0.8707,
        ],
        [
            0.8340,
            0.7627,
            0.7340,
            0.7073,
            0.6953,
        ],
        [
            0.7133,
            0.6720,
            0.6427,
            0.6173,
            0.5947,
        ],
        [
            0.5940,
            0.5587,
            0.5347,
            0.5373,
            0.5233,
        ],
    ]

    plt.figure(
        figsize=(9, 7)
    )

    image = plt.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        vmin=0.50,
        vmax=1.00,
    )

    plt.colorbar(
        image,
        label="Logical Success Rate",
    )

    plt.xticks(
        range(len(measurement_levels)),
        [
            f"{value:.2f}"
            for value
            in measurement_levels
        ],
    )

    plt.yticks(
        range(len(physical_levels)),
        [
            f"{value:.2f}"
            for value
            in physical_levels
        ],
    )

    plt.xlabel(
        "Measurement Noise Probability"
    )

    plt.ylabel(
        "Physical Noise Probability"
    )

    plt.title(
        "AI-QEC Logical Success Under Combined Noise"
    )

    for row_index in range(
        len(matrix)
    ):

        for column_index in range(
            len(matrix[row_index])
        ):

            value = matrix[
                row_index
            ][
                column_index
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output = (
        OUTPUT_DIRECTORY
        / "05_combined_noise_heatmap.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output


# ============================================================
# WRITE CSV-LIKE SUMMARY
# ============================================================

def save_summary(data):

    output = (
        OUTPUT_DIRECTORY
        / "scientific_plot_summary.csv"
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "physical_noise,"
            "baseline_logical_success,"
            "ai_logical_success,"
            "paired_gain,"
            "bootstrap_ci_low,"
            "bootstrap_ci_high\n"
        )

        for row in data:

            file.write(
                f"{row['noise']},"
                f"{row['baseline']},"
                f"{row['ai']},"
                f"{row['gain']},"
                f"{row['bootstrap_low']},"
                f"{row['bootstrap_high']}\n"
            )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        " SCIENTIFIC VISUALIZATION"
    )
    print("=" * 80)
    print()

    print(
        f"Reading: {RESULT_PATH}"
    )

    results = load_results()

    print(
        f"Loaded {len(results)} paired experiments."
    )

    data = aggregate_results(
        results
    )

    prepare_output()

    outputs = []

    outputs.append(
        plot_logical_success(data)
    )

    outputs.append(
        plot_gain(data)
    )

    outputs.append(
        plot_gain_confidence_interval(data)
    )

    outputs.append(
        plot_logical_error(data)
    )

    outputs.append(
        plot_noise_matrix()
    )

    outputs.append(
        save_summary(data)
    )

    print()
    print(
        "Generated files:"
    )

    for output in outputs:
        print(
            f"  {output}"
        )

    print()
    print("=" * 80)
    print(
        " SCIENTIFIC VISUALIZATION COMPLETE"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()