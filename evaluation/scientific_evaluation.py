from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ScientificResult:
    """Formal scientific result for one noise condition."""

    physical_noise: float
    measurement_noise: float

    baseline_logical_success: float
    ai_logical_success: float

    absolute_gain: float
    relative_gain: float

    baseline_logical_error: float
    ai_logical_error: float
    logical_error_reduction: float

    bootstrap_ci_low: float
    bootstrap_ci_high: float

    permutation_p_value: float

    seed_count: int
    test_samples_per_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScientificEvaluation:
    """
    Formal scientific evaluation layer.

    Reads previously generated paired baseline-vs-AI-QEC
    experiment results.

    This class does NOT rerun experiments.
    """

    def __init__(
        self,
        result_path: str | Path,
    ):
        self.result_path = Path(
            result_path
        )

        if not self.result_path.exists():
            raise FileNotFoundError(
                f"Scientific result file not found: "
                f"{self.result_path}"
            )

        with self.result_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if "results" not in payload:
            raise ValueError(
                "Invalid paired result file: "
                "'results' field is missing."
            )

        self.metadata = payload.get(
            "experiment",
            {},
        )

        self.raw_results = payload[
            "results"
        ]

    # ========================================================
    # CONDITIONS
    # ========================================================

    def conditions(
        self,
    ) -> list[tuple[float, float]]:

        conditions = {
            (
                float(
                    row["physical_noise"]
                ),
                float(
                    row["measurement_noise"]
                ),
            )
            for row in self.raw_results
        }

        return sorted(
            conditions
        )

    # ========================================================
    # EVALUATE ONE CONDITION
    # ========================================================

    def evaluate_condition(
        self,
        physical_noise: float,
        measurement_noise: float = 0.0,
    ) -> ScientificResult:

        rows = [
            row
            for row in self.raw_results
            if (
                float(
                    row["physical_noise"]
                )
                == float(physical_noise)
                and
                float(
                    row["measurement_noise"]
                )
                == float(measurement_noise)
            )
        ]

        if not rows:
            raise ValueError(
                "No paired results found for "
                f"physical_noise={physical_noise}, "
                f"measurement_noise={measurement_noise}"
            )

        baseline_values = [
            float(
                row[
                    "baseline_logical_success"
                ]
            )
            for row in rows
        ]

        ai_values = [
            float(
                row[
                    "ai_logical_success"
                ]
            )
            for row in rows
        ]

        gain_values = [
            float(
                row["paired_gain"]
            )
            for row in rows
        ]

        baseline_mean = (
            sum(baseline_values)
            / len(baseline_values)
        )

        ai_mean = (
            sum(ai_values)
            / len(ai_values)
        )

        gain = (
            sum(gain_values)
            / len(gain_values)
        )

        baseline_error = (
            1.0 - baseline_mean
        )

        ai_error = (
            1.0 - ai_mean
        )

        if baseline_mean == 0.0:
            relative_gain = 0.0
        else:
            relative_gain = (
                gain
                / baseline_mean
            )

        if baseline_error == 0.0:
            error_reduction = 0.0
        else:
            error_reduction = (
                baseline_error
                - ai_error
            ) / baseline_error

        bootstrap_low = (
            sum(
                float(
                    row[
                        "bootstrap_ci_low"
                    ]
                )
                for row in rows
            )
            / len(rows)
        )

        bootstrap_high = (
            sum(
                float(
                    row[
                        "bootstrap_ci_high"
                    ]
                )
                for row in rows
            )
            / len(rows)
        )

        # Conservative aggregation:
        # use the largest p-value across seeds.
        permutation_p = max(
            float(
                row[
                    "permutation_p_value"
                ]
            )
            for row in rows
        )

        test_samples = int(
            self.metadata.get(
                "test_samples",
                rows[0].get(
                    "test_samples",
                    0,
                ),
            )
        )

        return ScientificResult(
            physical_noise=physical_noise,
            measurement_noise=measurement_noise,
            baseline_logical_success=(
                baseline_mean
            ),
            ai_logical_success=(
                ai_mean
            ),
            absolute_gain=gain,
            relative_gain=relative_gain,
            baseline_logical_error=(
                baseline_error
            ),
            ai_logical_error=(
                ai_error
            ),
            logical_error_reduction=(
                error_reduction
            ),
            bootstrap_ci_low=(
                bootstrap_low
            ),
            bootstrap_ci_high=(
                bootstrap_high
            ),
            permutation_p_value=(
                permutation_p
            ),
            seed_count=len(rows),
            test_samples_per_seed=(
                test_samples
            ),
        )

    # ========================================================
    # EVALUATE ALL
    # ========================================================

    def evaluate_all(
        self,
    ) -> list[ScientificResult]:

        return [
            self.evaluate_condition(
                physical_noise=physical,
                measurement_noise=measurement,
            )
            for physical, measurement
            in self.conditions()
        ]

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
        results: list[ScientificResult] | None = None,
    ) -> dict[str, Any]:

        if results is None:
            results = self.evaluate_all()

        if not results:
            return {
                "conditions": 0,
                "positive_gain_conditions": 0,
                "maximum_absolute_gain": 0.0,
                "maximum_error_reduction": 0.0,
            }

        positive_gain = [
            result
            for result in results
            if result.absolute_gain > 0
        ]

        maximum_gain = max(
            results,
            key=lambda result:
            result.absolute_gain,
        )

        maximum_error_reduction = max(
            results,
            key=lambda result:
            result.logical_error_reduction,
        )

        return {
            "conditions": len(
                results
            ),
            "positive_gain_conditions": len(
                positive_gain
            ),
            "maximum_absolute_gain": {
                "physical_noise":
                    maximum_gain.physical_noise,
                "measurement_noise":
                    maximum_gain.measurement_noise,
                "gain":
                    maximum_gain.absolute_gain,
            },
            "maximum_error_reduction": {
                "physical_noise":
                    maximum_error_reduction.physical_noise,
                "measurement_noise":
                    maximum_error_reduction.measurement_noise,
                "reduction":
                    maximum_error_reduction.logical_error_reduction,
            },
        }

    # ========================================================
    # EXPORT JSON
    # ========================================================

    def export_json(
        self,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = self.evaluate_all()

        payload = {
            "source": str(
                self.result_path
            ),
            "metadata": self.metadata,
            "results": [
                result.to_dict()
                for result in results
            ],
            "summary": self.summary(
                results
            ),
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

    # ========================================================
    # EXPORT CSV
    # ========================================================

    def export_csv(
        self,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = self.evaluate_all()

        fields = [
            "physical_noise",
            "measurement_noise",
            "baseline_logical_success",
            "ai_logical_success",
            "absolute_gain",
            "relative_gain",
            "baseline_logical_error",
            "ai_logical_error",
            "logical_error_reduction",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "permutation_p_value",
            "seed_count",
            "test_samples_per_seed",
        ]

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                ",".join(fields)
                + "\n"
            )

            for result in results:

                values = result.to_dict()

                file.write(
                    ",".join(
                        str(
                            values[field]
                        )
                        for field in fields
                    )
                    + "\n"
                )

        return output_path

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    def export_report(
        self,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = self.evaluate_all()

        summary = self.summary(
            results
        )

        lines = []

        lines.append(
            "AI-POWERED QUANTUM ERROR "
            "CORRECTION SYSTEM"
        )

        lines.append(
            "SCIENTIFIC EVALUATION REPORT"
        )

        lines.append("")
        lines.append(
            "=" * 80
        )

        lines.append(
            "EXPERIMENT"
        )

        lines.append(
            f"QEC code: "
            f"{self.metadata.get('qec_code', 'bit_flip_3')}"
        )

        lines.append(
            f"Rounds: "
            f"{self.metadata.get('rounds', 'N/A')}"
        )

        lines.append(
            f"Training samples: "
            f"{self.metadata.get('training_samples', 'N/A')}"
        )

        lines.append(
            f"Test samples per seed: "
            f"{self.metadata.get('test_samples', 'N/A')}"
        )

        lines.append(
            f"Random Forest estimators: "
            f"{self.metadata.get('random_forest_estimators', 'N/A')}"
        )

        lines.append(
            f"Seeds: "
            f"{self.metadata.get('seeds', 'N/A')}"
        )

        lines.append("")
        lines.append(
            "PRIMARY RESULT"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            "Physical Noise | Baseline | AI-QEC | "
            "Gain | Error Reduction"
        )

        lines.append(
            "-" * 80
        )

        for result in results:

            lines.append(
                f"{result.physical_noise:>14.2f} | "
                f"{result.baseline_logical_success:>8.2%} | "
                f"{result.ai_logical_success:>6.2%} | "
                f"{result.absolute_gain:>+6.2%} | "
                f"{result.logical_error_reduction:>14.2%}"
            )

        lines.append("")
        lines.append(
            "STATISTICAL INFORMATION"
        )

        lines.append(
            "-" * 80
        )

        for result in results:

            lines.append(
                f"Noise {result.physical_noise:.2f}: "
                f"bootstrap CI "
                f"[{result.bootstrap_ci_low:.2%}, "
                f"{result.bootstrap_ci_high:.2%}], "
                f"permutation p="
                f"{result.permutation_p_value:.5f}"
            )

        lines.append("")
        lines.append(
            "SUMMARY"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"Conditions evaluated: "
            f"{summary['conditions']}"
        )

        lines.append(
            f"Positive-gain conditions: "
            f"{summary['positive_gain_conditions']}"
        )

        maximum_gain = (
            summary[
                "maximum_absolute_gain"
            ]
        )

        lines.append(
            f"Maximum absolute gain: "
            f"{maximum_gain['gain']:.2%} "
            f"at physical noise "
            f"{maximum_gain['physical_noise']:.2f}"
        )

        maximum_reduction = (
            summary[
                "maximum_error_reduction"
            ]
        )

        lines.append(
            f"Maximum logical-error reduction: "
            f"{maximum_reduction['reduction']:.2%} "
            f"at physical noise "
            f"{maximum_reduction['physical_noise']:.2f}"
        )

        lines.append("")
        lines.append(
            "INTERPRETATION"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            "The paired comparison evaluates baseline "
            "and AI-QEC on identical held-out noisy samples."
        )

        lines.append(
            "Logical success is treated as the primary "
            "system-level QEC metric."
        )

        lines.append(
            "The bootstrap and paired permutation "
            "statistics provide experimental evidence, "
            "but results should be interpreted within "
            "the tested 3-qubit repetition-code configuration."
        )

        lines.append(
            "These results do not establish universal "
            "superiority across other QEC codes, noise "
            "models, circuit depths, or decoder architectures."
        )

        lines.append("")

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "\n".join(lines)
            )

        return output_path


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    source = (
        Path("experiments")
        / "paired_results"
        / "paired_baseline_ai_results.json"
    )

    output_dir = (
        Path("experiments")
        / "scientific_evaluation"
    )

    print()
    print("=" * 80)
    print(
        " SCIENTIFIC EVALUATION ENGINE"
    )
    print("=" * 80)
    print()

    evaluator = ScientificEvaluation(
        source
    )

    results = evaluator.evaluate_all()

    print(
        f"Loaded {len(results)} "
        f"scientific conditions."
    )

    print()

    for result in results:

        print(
            f"Noise {result.physical_noise:.2f}: "
            f"baseline="
            f"{result.baseline_logical_success:.2%}, "
            f"AI-QEC="
            f"{result.ai_logical_success:.2%}, "
            f"gain="
            f"{result.absolute_gain:+.2%}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = evaluator.export_json(
        output_dir
        / "scientific_results.json"
    )

    csv_path = evaluator.export_csv(
        output_dir
        / "scientific_results.csv"
    )

    report_path = evaluator.export_report(
        output_dir
        / "scientific_report.txt"
    )

    print()
    print(
        "Generated:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {report_path}"
    )

    print()
    print("=" * 80)
    print(
        " SCIENTIFIC EVALUATION COMPLETE"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()