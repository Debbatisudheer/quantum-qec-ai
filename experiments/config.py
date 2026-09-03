from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration for one complete QEC experiment.
    """

    # --------------------------------------------------------
    # Quantum / QEC
    # --------------------------------------------------------

    qec_code: str = "bit_flip_3"
    num_qubits: int = 3
    logical_state: int | None = None

    # --------------------------------------------------------
    # Repeated QEC
    # --------------------------------------------------------

    rounds: int = 5

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    physical_noise_probability: float = 0.10
    measurement_noise_probability: float = 0.10

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    training_samples: int = 5000
    test_samples: int = 1000

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder_type: str = "logical_target_random_forest"
    random_forest_estimators: int = 100

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed: int = 42

    def validate(self):
        """
        Validate the experiment configuration.
        """

        if self.qec_code != "bit_flip_3":
            raise ValueError(
                "Currently supported qec_code: bit_flip_3"
            )

        if self.num_qubits != 3:
            raise ValueError(
                "Currently supported num_qubits: 3"
            )

        if self.logical_state is not None:
            if self.logical_state not in (0, 1):
                raise ValueError(
                    "logical_state must be 0, 1, or None"
                )

        if self.rounds <= 0:
            raise ValueError(
                "rounds must be greater than 0"
            )

        if not 0.0 <= (
            self.physical_noise_probability
        ) <= 1.0:
            raise ValueError(
                "physical_noise_probability "
                "must be between 0 and 1"
            )

        if not 0.0 <= (
            self.measurement_noise_probability
        ) <= 1.0:
            raise ValueError(
                "measurement_noise_probability "
                "must be between 0 and 1"
            )

        if self.training_samples <= 0:
            raise ValueError(
                "training_samples must be greater than 0"
            )

        if self.test_samples <= 0:
            raise ValueError(
                "test_samples must be greater than 0"
            )

        if self.decoder_type != (
            "logical_target_random_forest"
        ):
            raise ValueError(
                "Currently supported decoder_type: "
                "logical_target_random_forest"
            )

        if self.random_forest_estimators <= 0:
            raise ValueError(
                "random_forest_estimators "
                "must be greater than 0"
            )

        return self