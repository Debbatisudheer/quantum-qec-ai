from experiments.result_storage import (
    ExperimentResultStorage
)


class ExperimentResultQuery:
    """
    Query and compare stored experiment results.

    This class is responsible for reading stored
    experiment results and performing filtering,
    sorting, and comparison.

    It does NOT run experiments.
    It does NOT train models.
    It does NOT modify stored results.
    """

    def __init__(
        self,
        storage
    ):
        if not isinstance(
            storage,
            ExperimentResultStorage
        ):
            raise TypeError(
                "storage must be an "
                "ExperimentResultStorage"
            )

        self.storage = storage

    # ========================================================
    # LOAD ALL
    # ========================================================

    def all_results(self):
        """
        Return all stored experiment results.
        """

        return self.storage.load_all()

    # ========================================================
    # FILTER
    # ========================================================

    def filter(
        self,
        qec_code=None,
        rounds=None,
        physical_noise_probability=None,
        measurement_noise_probability=None,
        decoder_type=None
    ):
        """
        Filter stored experiment results.

        Any argument set to None is ignored.
        """

        results = self.all_results()

        filtered = []

        for result in results:

            config = result.get(
                "config",
                {}
            )

            if (
                qec_code is not None
                and config.get("qec_code")
                != qec_code
            ):
                continue

            if (
                rounds is not None
                and config.get("rounds")
                != rounds
            ):
                continue

            if (
                physical_noise_probability
                is not None
                and config.get(
                    "physical_noise_probability"
                )
                != physical_noise_probability
            ):
                continue

            if (
                measurement_noise_probability
                is not None
                and config.get(
                    "measurement_noise_probability"
                )
                != measurement_noise_probability
            ):
                continue

            if (
                decoder_type is not None
                and result.get(
                    "decoder_type"
                )
                != decoder_type
            ):
                continue

            filtered.append(
                result
            )

        return filtered

    # ========================================================
    # SORT
    # ========================================================

    def sort_by(
        self,
        results,
        metric,
        descending=True
    ):
        """
        Sort results by a metric.

        Example:

            sort_by(
                results,
                "logical_accuracy"
            )
        """

        if not isinstance(
            results,
            list
        ):
            raise TypeError(
                "results must be a list"
            )

        valid_metrics = {
            "exact_accuracy",
            "physical_accuracy",
            "bit_accuracy",
            "logical_accuracy",
            "training_seconds",
            "inference_seconds",
            "samples_per_second",
        }

        if metric not in valid_metrics:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        return sorted(
            results,
            key=lambda result:
                result.get(metric, 0.0),
            reverse=descending
        )

    # ========================================================
    # BEST
    # ========================================================

    def best(
        self,
        results=None,
        metric="logical_accuracy"
    ):
        """
        Return the best result according to
        the selected metric.
        """

        if results is None:
            results = self.all_results()

        if not results:
            return None

        sorted_results = (
            self.sort_by(
                results,
                metric,
                descending=True
            )
        )

        return sorted_results[0]

    # ========================================================
    # WORST
    # ========================================================

    def worst(
        self,
        results=None,
        metric="logical_accuracy"
    ):
        """
        Return the worst result according to
        the selected metric.
        """

        if results is None:
            results = self.all_results()

        if not results:
            return None

        sorted_results = (
            self.sort_by(
                results,
                metric,
                descending=False
            )
        )

        return sorted_results[0]

    # ========================================================
    # COMPARE
    # ========================================================

    def compare(
        self,
        experiment_ids
    ):
        """
        Load and compare specific experiments.

        Returns results sorted by logical success
        from highest to lowest.
        """

        if not experiment_ids:
            raise ValueError(
                "experiment_ids cannot be empty"
            )

        results = []

        for experiment_id in experiment_ids:

            results.append(
                self.storage.load(
                    experiment_id
                )
            )

        return self.sort_by(
            results,
            "logical_accuracy",
            descending=True
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def summary(
        results
    ):
        """
        Create a compact summary of experiment
        results.
        """

        if not results:
            return {
                "count": 0,
                "best_logical_accuracy": None,
                "average_logical_accuracy": None,
            }

        logical_values = [
            result[
                "logical_accuracy"
            ]
            for result in results
        ]

        return {
            "count": len(results),

            "best_logical_accuracy": max(
                logical_values
            ),

            "average_logical_accuracy": (
                sum(logical_values)
                / len(logical_values)
            ),
        }