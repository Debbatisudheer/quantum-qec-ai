from collections import defaultdict

from experiments.result_query import (
    ExperimentResultQuery
)


class ExperimentAnalysis:
    """
    Analyze QEC experiment results.

    Responsibilities:

        - Calculate aggregate statistics
        - Group results by experiment parameters
        - Find best/worst results
        - Calculate averages
        - Calculate AI improvement over baseline
        - Convert results into dashboard/report rows

    This class does NOT:

        - run experiments
        - train decoders
        - modify stored results
    """

    def __init__(self, query):
        if not isinstance(
            query,
            ExperimentResultQuery
        ):
            raise TypeError(
                "query must be an "
                "ExperimentResultQuery"
            )

        self.query = query

    # ========================================================
    # RESULT ACCESS
    # ========================================================

    @staticmethod
    def get_value(
        result,
        field,
        default=None
    ):
        """
        Read a value from either:

            - ExperimentResult object
            - dictionary

        This keeps the analysis layer compatible with
        both in-memory results and stored JSON results.
        """

        if isinstance(
            result,
            dict
        ):
            return result.get(
                field,
                default
            )

        return getattr(
            result,
            field,
            default
        )

    @classmethod
    def get_config_value(
        cls,
        result,
        field,
        default=None
    ):
        """
        Read a configuration value from either:

            result.config[field]

        or:

            result["config"][field]
        """

        config = cls.get_value(
            result,
            "config",
            None
        )

        if config is None:
            return default

        if isinstance(
            config,
            dict
        ):
            return config.get(
                field,
                default
            )

        return getattr(
            config,
            field,
            default
        )

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    @classmethod
    def average(
        cls,
        results,
        metric
    ):
        """
        Calculate the average of a metric.
        """

        if not results:
            return 0.0

        values = []

        for result in results:

            value = cls.get_value(
                result,
                metric,
                0.0
            )

            values.append(
                float(value)
            )

        return (
            sum(values)
            / len(values)
        )

    @classmethod
    def minimum(
        cls,
        results,
        metric
    ):
        """
        Return the minimum metric value.
        """

        if not results:
            return None

        values = [
            float(
                cls.get_value(
                    result,
                    metric,
                    0.0
                )
            )
            for result in results
        ]

        return min(values)

    @classmethod
    def maximum(
        cls,
        results,
        metric
    ):
        """
        Return the maximum metric value.
        """

        if not results:
            return None

        values = [
            float(
                cls.get_value(
                    result,
                    metric,
                    0.0
                )
            )
            for result in results
        ]

        return max(values)

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
        results=None
    ):
        """
        Generate a summary of experiment results.
        """

        if results is None:
            results = self.query.all_results()

        if not results:
            return {
                "count": 0,
                "average_logical_accuracy": 0.0,
                "best_logical_accuracy": None,
                "worst_logical_accuracy": None,
                "average_physical_accuracy": 0.0,
                "average_bit_accuracy": 0.0,
            }

        return {
            "count": len(results),

            "average_logical_accuracy": (
                self.average(
                    results,
                    "logical_accuracy"
                )
            ),

            "best_logical_accuracy": (
                self.maximum(
                    results,
                    "logical_accuracy"
                )
            ),

            "worst_logical_accuracy": (
                self.minimum(
                    results,
                    "logical_accuracy"
                )
            ),

            "average_physical_accuracy": (
                self.average(
                    results,
                    "physical_accuracy"
                )
            ),

            "average_bit_accuracy": (
                self.average(
                    results,
                    "bit_accuracy"
                )
            ),
        }

    # ========================================================
    # GROUP BY
    # ========================================================

    @classmethod
    def group_by(
        cls,
        results,
        field
    ):
        """
        Group experiment results by a field.

        Examples:

            decoder_type

            config.rounds

            config.physical_noise_probability
        """

        groups = defaultdict(list)

        for result in results:

            if field.startswith(
                "config."
            ):

                config_field = (
                    field[len("config."):]
                )

                key = cls.get_config_value(
                    result,
                    config_field
                )

            else:

                key = cls.get_value(
                    result,
                    field
                )

            groups[key].append(
                result
            )

        return dict(groups)

    # ========================================================
    # GROUPED AVERAGE
    # ========================================================

    def grouped_average(
        self,
        results,
        group_field,
        metric="logical_accuracy"
    ):
        """
        Calculate the average metric for
        every group.
        """

        groups = self.group_by(
            results,
            group_field
        )

        output = {}

        for key, group_results in (
            groups.items()
        ):

            output[key] = (
                self.average(
                    group_results,
                    metric
                )
            )

        return output

    # ========================================================
    # ANALYZE ROUNDS
    # ========================================================

    def analyze_rounds(
        self,
        results=None
    ):
        """
        Analyze logical success as the number
        of QEC rounds changes.
        """

        if results is None:
            results = self.query.all_results()

        return self.grouped_average(
            results,
            "config.rounds",
            "logical_accuracy"
        )

    # ========================================================
    # ANALYZE PHYSICAL NOISE
    # ========================================================

    def analyze_physical_noise(
        self,
        results=None
    ):
        """
        Analyze logical success as physical
        noise changes.
        """

        if results is None:
            results = self.query.all_results()

        return self.grouped_average(
            results,
            "config.physical_noise_probability",
            "logical_accuracy"
        )

    # ========================================================
    # ANALYZE MEASUREMENT NOISE
    # ========================================================

    def analyze_measurement_noise(
        self,
        results=None
    ):
        """
        Analyze logical success as measurement
        noise changes.
        """

        if results is None:
            results = self.query.all_results()

        return self.grouped_average(
            results,
            "config.measurement_noise_probability",
            "logical_accuracy"
        )

    # ========================================================
    # ANALYZE DECODERS
    # ========================================================

    def analyze_decoders(
        self,
        results=None
    ):
        """
        Compare decoder logical accuracy.
        """

        if results is None:
            results = self.query.all_results()

        return self.grouped_average(
            results,
            "decoder_type",
            "logical_accuracy"
        )

    # ========================================================
    # BEST EXPERIMENT
    # ========================================================

    def best_experiment(
        self,
        results=None,
        metric="logical_accuracy"
    ):
        """
        Return the best experiment.
        """

        if results is None:
            results = self.query.all_results()

        if not results:
            return None

        values = [
            (
                float(
                    self.get_value(
                        result,
                        metric,
                        0.0
                    )
                ),
                result
            )
            for result in results
        ]

        return max(
            values,
            key=lambda item: item[0]
        )[1]

    # ========================================================
    # WORST EXPERIMENT
    # ========================================================

    def worst_experiment(
        self,
        results=None,
        metric="logical_accuracy"
    ):
        """
        Return the worst experiment.
        """

        if results is None:
            results = self.query.all_results()

        if not results:
            return None

        values = [
            (
                float(
                    self.get_value(
                        result,
                        metric,
                        0.0
                    )
                ),
                result
            )
            for result in results
        ]

        return min(
            values,
            key=lambda item: item[0]
        )[1]

    # ========================================================
    # AI IMPROVEMENT
    # ========================================================

    @staticmethod
    def calculate_gain(
        baseline_accuracy,
        ai_accuracy
    ):
        """
        Calculate absolute improvement.

        Example:

            baseline = 0.70
            AI       = 0.76

            gain = +0.06
        """

        return (
            float(ai_accuracy)
            - float(baseline_accuracy)
        )

    # ========================================================
    # RESULT TABLE
    # ========================================================

    @classmethod
    def result_rows(
        cls,
        results
    ):
        """
        Convert results into compact rows suitable
        for tables, reports, or dashboard APIs.
        """

        rows = []

        for result in results:

            rows.append(
                {
                    "experiment_id": cls.get_value(
                        result,
                        "experiment_id"
                    ),

                    "qec_code": cls.get_config_value(
                        result,
                        "qec_code"
                    ),

                    "rounds": cls.get_config_value(
                        result,
                        "rounds"
                    ),

                    "physical_noise": cls.get_config_value(
                        result,
                        "physical_noise_probability"
                    ),

                    "measurement_noise": cls.get_config_value(
                        result,
                        "measurement_noise_probability"
                    ),

                    "decoder": cls.get_value(
                        result,
                        "decoder_type"
                    ),

                    "logical_accuracy": cls.get_value(
                        result,
                        "logical_accuracy"
                    ),

                    "physical_accuracy": cls.get_value(
                        result,
                        "physical_accuracy"
                    ),

                    "bit_accuracy": cls.get_value(
                        result,
                        "bit_accuracy"
                    ),

                    "training_seconds": cls.get_value(
                        result,
                        "training_seconds"
                    ),

                    "inference_seconds": cls.get_value(
                        result,
                        "inference_seconds"
                    ),

                    "samples_per_second": cls.get_value(
                        result,
                        "samples_per_second"
                    ),
                }
            )

        return rows