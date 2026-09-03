from experiments.config import ExperimentConfig
from experiments.engine import ExperimentEngine
from experiments.result_storage import ExperimentResultStorage


class MultiExperimentRunner:
    """
    Run multiple QEC experiments from a configuration grid.

    Responsibilities:

        1. Build ExperimentConfig objects
        2. Run each experiment
        3. Store each result
        4. Return all results

    This class does NOT implement:
        - quantum simulation
        - noise
        - QEC
        - decoding
        - evaluation
        - result querying
    """

    def __init__(
        self,
        storage=None
    ):
        if storage is None:
            storage = ExperimentResultStorage()

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
    # CONFIGURATION GRID
    # ========================================================

    @staticmethod
    def build_grid(
        qec_code,
        num_qubits,
        rounds_list,
        physical_noise_list,
        measurement_noise_list,
        training_samples,
        test_samples,
        decoder_type,
        random_forest_estimators,
        seed
    ):
        """
        Build ExperimentConfig objects from
        parameter lists.
        """

        if not rounds_list:
            raise ValueError(
                "rounds_list cannot be empty"
            )

        if not physical_noise_list:
            raise ValueError(
                "physical_noise_list cannot be empty"
            )

        if not measurement_noise_list:
            raise ValueError(
                "measurement_noise_list cannot be empty"
            )

        configs = []

        seed_offset = 0

        for rounds in rounds_list:

            for physical_noise in (
                physical_noise_list
            ):

                for measurement_noise in (
                    measurement_noise_list
                ):

                    config = ExperimentConfig(
                        qec_code=qec_code,

                        num_qubits=num_qubits,

                        rounds=rounds,

                        physical_noise_probability=(
                            physical_noise
                        ),

                        measurement_noise_probability=(
                            measurement_noise
                        ),

                        training_samples=(
                            training_samples
                        ),

                        test_samples=(
                            test_samples
                        ),

                        decoder_type=(
                            decoder_type
                        ),

                        random_forest_estimators=(
                            random_forest_estimators
                        ),

                        seed=(
                            seed + seed_offset
                        )
                    )

                    config.validate()

                    configs.append(
                        config
                    )

                    seed_offset += 1

        return configs

    # ========================================================
    # RUN ONE
    # ========================================================

    def run_one(
        self,
        config
    ):
        """
        Run one ExperimentConfig.
        """

        if not isinstance(
            config,
            ExperimentConfig
        ):
            raise TypeError(
                "config must be an "
                "ExperimentConfig"
            )

        engine = ExperimentEngine(
            config=config,
            storage=self.storage
        )

        return engine.run()

    # ========================================================
    # RUN MANY
    # ========================================================

    def run(
        self,
        configs
    ):
        """
        Run all supplied experiment configurations.

        Returns:
            list[ExperimentResult]
        """

        if not configs:
            raise ValueError(
                "configs cannot be empty"
            )

        results = []

        total = len(configs)

        for index, config in enumerate(
            configs,
            start=1
        ):

            print()
            print(
                f"[MULTI-RUNNER] "
                f"Experiment {index}/{total}"
            )

            result = self.run_one(
                config
            )

            results.append(
                result
            )

        return results