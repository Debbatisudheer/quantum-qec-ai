from experiments.analysis import (
    ExperimentAnalysis
)


class ExperimentReportGenerator:
    """
    Generate structured reports from QEC experiment results.

    Responsibilities:

        - Build experiment summaries
        - Include configuration information
        - Include performance metrics
        - Include round analysis
        - Include noise analysis
        - Identify best/worst experiments
        - Produce human-readable text reports

    This class does NOT:

        - run experiments
        - train models
        - modify stored results
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
    # BUILD STRUCTURED REPORT
    # ========================================================

    def build(
        self,
        results=None
    ):
        """
        Build a structured report dictionary.
        """

        if results is None:
            results = (
                self.analysis.query.all_results()
            )

        if not results:
            return {
                "experiment_count": 0,
                "summary": {},
                "round_analysis": {},
                "physical_noise_analysis": {},
                "measurement_noise_analysis": {},
                "decoder_analysis": {},
                "best_experiment": None,
                "worst_experiment": None,
                "results": [],
            }

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

        physical_noise_analysis = (
            self.analysis.analyze_physical_noise(
                results
            )
        )

        measurement_noise_analysis = (
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

        return {
            "experiment_count": len(
                results
            ),

            "summary": summary,

            "round_analysis": (
                round_analysis
            ),

            "physical_noise_analysis": (
                physical_noise_analysis
            ),

            "measurement_noise_analysis": (
                measurement_noise_analysis
            ),

            "decoder_analysis": (
                decoder_analysis
            ),

            "best_experiment": (
                self._result_to_dict(
                    best
                )
            ),

            "worst_experiment": (
                self._result_to_dict(
                    worst
                )
            ),

            "results": (
                self.analysis.result_rows(
                    results
                )
            ),
        }

    # ========================================================
    # RESULT CONVERSION
    # ========================================================

    @staticmethod
    def _result_to_dict(
        result
    ):
        """
        Convert an ExperimentResult object or
        dictionary into a report-friendly dictionary.
        """

        if result is None:
            return None

        if isinstance(
            result,
            dict
        ):
            return dict(result)

        if hasattr(
            result,
            "to_dict"
        ):
            return result.to_dict()

        return {
            key: value
            for key, value
            in vars(result).items()
        }

    # ========================================================
    # TEXT REPORT
    # ========================================================

    def to_text(
        self,
        results=None
    ):
        """
        Generate a human-readable text report.
        """

        report = self.build(
            results
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
            f"{report['experiment_count']}"
        )

        summary = report[
            "summary"
        ]

        if summary:

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

            lines.append(
                f"Average physical recovery: "
                f"{summary['average_physical_accuracy']:.4f}"
            )

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
            report[
                "round_analysis"
            ].items()
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
            report[
                "physical_noise_analysis"
            ].items()
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
            report[
                "measurement_noise_analysis"
            ].items()
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
            report[
                "decoder_analysis"
            ].items()
        ):

            lines.append(
                f"{decoder:<30}: "
                f"{accuracy:.4f}"
            )

        lines.append("")

        # ----------------------------------------------------
        # BEST EXPERIMENT
        # ----------------------------------------------------

        lines.append(
            "BEST EXPERIMENT"
        )

        lines.append(
            "-" * 70
        )

        best = report[
            "best_experiment"
        ]

        if best is not None:

            lines.append(
                f"Experiment ID           : "
                f"{best.get('experiment_id')}"
            )

            lines.append(
                f"Decoder                 : "
                f"{best.get('decoder_type', best.get('decoder'))}"
            )

            lines.append(
                f"Logical success         : "
                f"{best.get('logical_accuracy'):.4f}"
            )

        else:

            lines.append(
                "No experiment available."
            )

        lines.append("")

        # ----------------------------------------------------
        # WORST EXPERIMENT
        # ----------------------------------------------------

        lines.append(
            "WORST EXPERIMENT"
        )

        lines.append(
            "-" * 70
        )

        worst = report[
            "worst_experiment"
        ]

        if worst is not None:

            lines.append(
                f"Experiment ID           : "
                f"{worst.get('experiment_id')}"
            )

            lines.append(
                f"Decoder                 : "
                f"{worst.get('decoder_type', worst.get('decoder'))}"
            )

            lines.append(
                f"Logical success         : "
                f"{worst.get('logical_accuracy'):.4f}"
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
    # REPORT ROWS
    # ========================================================

    def result_rows(
        self,
        results=None
    ):
        """
        Return compact result rows for future
        dashboard tables or CSV export.
        """

        if results is None:
            results = (
                self.analysis.query.all_results()
            )

        return self.analysis.result_rows(
            results
        )