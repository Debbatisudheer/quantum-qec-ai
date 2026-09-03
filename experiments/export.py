import csv
import json
from pathlib import Path

from experiments.analysis import ExperimentAnalysis


class ExperimentExporter:
    """
    Export QEC experiment results.

    Supported formats:

        - JSON
        - CSV
        - TXT

    This class does NOT:

        - run experiments
        - train models
        - modify experiment results
        - modify stored experiment data

    It only converts existing results into
    portable output files.
    """

    def __init__(self, analysis):
        if not isinstance(
            analysis,
            ExperimentAnalysis
        ):
            raise TypeError(
                "analysis must be an "
                "ExperimentAnalysis"
            )

        self.analysis = analysis

    # ========================================================
    # RESULT RESOLUTION
    # ========================================================

    def _resolve_results(
        self,
        results=None
    ):
        """
        Resolve results from the provided list or
        load all results through ExperimentAnalysis.
        """

        if results is None:
            results = (
                self.analysis.query.all_results()
            )

        if not isinstance(
            results,
            list
        ):
            raise TypeError(
                "results must be a list"
            )

        return results

    # ========================================================
    # JSON
    # ========================================================

    def export_json(
        self,
        results,
        output_path
    ):
        """
        Export complete experiment results as JSON.
        """

        results = self._resolve_results(
            results
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = []

        for result in results:

            if isinstance(
                result,
                dict
            ):
                data.append(
                    dict(result)
                )

            elif hasattr(
                result,
                "to_dict"
            ):
                data.append(
                    result.to_dict()
                )

            else:
                data.append(
                    {
                        key: value
                        for key, value
                        in vars(result).items()
                    }
                )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

        return output_path

    # ========================================================
    # CSV
    # ========================================================

    def export_csv(
        self,
        results,
        output_path
    ):
        """
        Export compact experiment result rows as CSV.
        """

        results = self._resolve_results(
            results
        )

        rows = (
            self.analysis.result_rows(
                results
            )
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not rows:
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as file:
                file.write("")

            return output_path

        fieldnames = list(
            rows[0].keys()
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

        return output_path

    # ========================================================
    # TEXT
    # ========================================================

    def export_text(
        self,
        results,
        output_path
    ):
        """
        Export a human-readable experiment report.
        """

        results = self._resolve_results(
            results
        )

        text = (
            self.analysis_to_text(
                results
            )
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                text
            )

        return output_path

    # ========================================================
    # TEXT REPORT
    # ========================================================

    def analysis_to_text(
        self,
        results
    ):
        """
        Generate a human-readable text report.

        The report is generated directly from
        ExperimentAnalysis so that the export layer
        does not duplicate analysis logic.
        """

        results = self._resolve_results(
            results
        )

        if not results:
            return (
                "=" * 70
                + "\n"
                + " QEC EXPERIMENT REPORT\n"
                + "=" * 70
                + "\n\n"
                + "No experiments available.\n"
            )

        summary = (
            self.analysis.summary(
                results
            )
        )

        round_analysis = (
            self.analysis.analyze_rounds(
                results
            )
        )

        physical_noise = (
            self.analysis.analyze_physical_noise(
                results
            )
        )

        measurement_noise = (
            self.analysis.analyze_measurement_noise(
                results
            )
        )

        decoder_analysis = (
            self.analysis.analyze_decoders(
                results
            )
        )

        best = (
            self.analysis.best_experiment(
                results
            )
        )

        worst = (
            self.analysis.worst_experiment(
                results
            )
        )

        lines = []

        lines.append(
            "=" * 70
        )

        lines.append(
            " QEC EXPERIMENT REPORT"
        )

        lines.append(
            "=" * 70
        )

        lines.append("")

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        lines.append(
            "OVERVIEW"
        )

        lines.append(
            "-" * 70
        )

        lines.append(
            f"Experiments              : "
            f"{summary['count']}"
        )

        lines.append(
            f"Average logical success  : "
            f"{summary['average_logical_accuracy']:.4f}"
        )

        lines.append(
            f"Best logical success     : "
            f"{summary['best_logical_accuracy']:.4f}"
        )

        lines.append(
            f"Worst logical success    : "
            f"{summary['worst_logical_accuracy']:.4f}"
        )

        if "average_physical_accuracy" in summary:

            lines.append(
                f"Average physical recovery: "
                f"{summary['average_physical_accuracy']:.4f}"
            )

        if "average_bit_accuracy" in summary:

            lines.append(
                f"Average bit accuracy     : "
                f"{summary['average_bit_accuracy']:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # ROUND ANALYSIS
        # ----------------------------------------------------

        lines.append(
            "ROUND ANALYSIS"
        )

        lines.append(
            "-" * 70
        )

        for rounds, accuracy in sorted(
            round_analysis.items()
        ):

            lines.append(
                f"Rounds {rounds:<20}: "
                f"{accuracy:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # PHYSICAL NOISE
        # ----------------------------------------------------

        lines.append(
            "PHYSICAL NOISE ANALYSIS"
        )

        lines.append(
            "-" * 70
        )

        for noise, accuracy in sorted(
            physical_noise.items()
        ):

            lines.append(
                f"Noise {noise:<22}: "
                f"{accuracy:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # MEASUREMENT NOISE
        # ----------------------------------------------------

        lines.append(
            "MEASUREMENT NOISE ANALYSIS"
        )

        lines.append(
            "-" * 70
        )

        for noise, accuracy in sorted(
            measurement_noise.items()
        ):

            lines.append(
                f"Noise {noise:<22}: "
                f"{accuracy:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # DECODER ANALYSIS
        # ----------------------------------------------------

        lines.append(
            "DECODER ANALYSIS"
        )

        lines.append(
            "-" * 70
        )

        for decoder, accuracy in sorted(
            decoder_analysis.items()
        ):

            lines.append(
                f"{decoder:<30}: "
                f"{accuracy:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        lines.append(
            "BEST EXPERIMENT"
        )

        lines.append(
            "-" * 70
        )

        if best is not None:

            if isinstance(
                best,
                dict
            ):
                experiment_id = (
                    best.get(
                        "experiment_id"
                    )
                )

                decoder = (
                    best.get(
                        "decoder_type",
                        best.get(
                            "decoder"
                        )
                    )
                )

                logical_accuracy = (
                    best.get(
                        "logical_accuracy",
                        0.0
                    )
                )

            else:

                experiment_id = (
                    best.experiment_id
                )

                decoder = (
                    best.decoder_type
                )

                logical_accuracy = (
                    best.logical_accuracy
                )

            lines.append(
                f"Experiment ID           : "
                f"{experiment_id}"
            )

            lines.append(
                f"Decoder                 : "
                f"{decoder}"
            )

            lines.append(
                f"Logical success         : "
                f"{logical_accuracy:.4f}"
            )

        else:

            lines.append(
                "No experiment available."
            )

        lines.append("")

        # ----------------------------------------------------
        # WORST
        # ----------------------------------------------------

        lines.append(
            "WORST EXPERIMENT"
        )

        lines.append(
            "-" * 70
        )

        if worst is not None:

            if isinstance(
                worst,
                dict
            ):
                experiment_id = (
                    worst.get(
                        "experiment_id"
                    )
                )

                decoder = (
                    worst.get(
                        "decoder_type",
                        worst.get(
                            "decoder"
                        )
                    )
                )

                logical_accuracy = (
                    worst.get(
                        "logical_accuracy",
                        0.0
                    )
                )

            else:

                experiment_id = (
                    worst.experiment_id
                )

                decoder = (
                    worst.decoder_type
                )

                logical_accuracy = (
                    worst.logical_accuracy
                )

            lines.append(
                f"Experiment ID           : "
                f"{experiment_id}"
            )

            lines.append(
                f"Decoder                 : "
                f"{decoder}"
            )

            lines.append(
                f"Logical success         : "
                f"{logical_accuracy:.4f}"
            )

        else:

            lines.append(
                "No experiment available."
            )

        lines.append("")

        lines.append(
            "=" * 70
        )

        lines.append(
            " END OF REPORT"
        )

        lines.append(
            "=" * 70
        )

        return "\n".join(
            lines
        )

    # ========================================================
    # EXPORT ALL
    # ========================================================

    def export_all(
        self,
        results,
        output_directory
    ):
        """
        Export JSON, CSV, and TXT versions
        into one directory.
        """

        results = self._resolve_results(
            results
        )

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        json_path = (
            self.export_json(
                results,
                output_directory
                / "experiments.json"
            )
        )

        csv_path = (
            self.export_csv(
                results,
                output_directory
                / "experiments.csv"
            )
        )

        text_path = (
            self.export_text(
                results,
                output_directory
                / "experiments.txt"
            )
        )

        return {
            "json": json_path,
            "csv": csv_path,
            "text": text_path,
        }