import json
from pathlib import Path

from experiments.result import ExperimentResult


class ExperimentResultStorage:
    """
    Persistent storage for ExperimentResult objects.

    Results are stored as JSON files.

    Default location:

        experiments/results/
    """

    def __init__(
        self,
        storage_directory=None
    ):
        if storage_directory is None:
            storage_directory = (
                Path(__file__).resolve().parent
                / "results"
            )

        self.storage_directory = Path(
            storage_directory
        )

        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # PATH
    # ========================================================

    def result_path(
        self,
        experiment_id
    ):
        """
        Return the JSON path for an experiment.
        """

        if not experiment_id:
            raise ValueError(
                "experiment_id cannot be empty"
            )

        return (
            self.storage_directory
            / f"{experiment_id}.json"
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        result
    ):
        """
        Save an ExperimentResult as JSON.
        """

        if not isinstance(
            result,
            ExperimentResult
        ):
            raise TypeError(
                "result must be an ExperimentResult"
            )

        path = self.result_path(
            result.experiment_id
        )

        with path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result.to_dict(),
                file,
                indent=4
            )

        return path

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        experiment_id
    ):
        """
        Load one experiment result.

        Returns:
            dict
        """

        path = self.result_path(
            experiment_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Experiment result not found: "
                f"{experiment_id}"
            )

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ========================================================
    # EXISTS
    # ========================================================

    def exists(
        self,
        experiment_id
    ):
        """
        Check whether an experiment result exists.
        """

        return self.result_path(
            experiment_id
        ).exists()

    # ========================================================
    # LIST
    # ========================================================

    def list_results(self):
        """
        List all stored experiment IDs.

        Results are sorted alphabetically.
        """

        paths = sorted(
            self.storage_directory.glob(
                "*.json"
            )
        )

        return [
            path.stem
            for path in paths
        ]

    # ========================================================
    # LOAD ALL
    # ========================================================

    def load_all(self):
        """
        Load all stored experiment results.

        Returns:
            list[dict]
        """

        experiment_ids = (
            self.list_results()
        )

        return [
            self.load(experiment_id)
            for experiment_id
            in experiment_ids
        ]

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        experiment_id
    ):
        """
        Delete one stored experiment result.
        """

        path = self.result_path(
            experiment_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Experiment result not found: "
                f"{experiment_id}"
            )

        path.unlink()

        return True

    # ========================================================
    # COUNT
    # ========================================================

    def count(self):
        """
        Return the number of stored results.
        """

        return len(
            self.list_results()
        )