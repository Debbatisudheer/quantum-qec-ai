from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentResult:
    """
    Result produced by one complete experiment.
    """

    experiment_id: str

    config: dict[str, Any]

    training_samples: int
    test_samples: int

    logical_targets_learned: int
    average_target_score: float

    exact_accuracy: float
    physical_accuracy: float
    bit_accuracy: float
    logical_accuracy: float

    training_seconds: float
    inference_seconds: float
    samples_per_second: float

    decoder_type: str

    def to_dict(self):
        """
        Convert the result into a normal dictionary.
        """

        return {
            "experiment_id": self.experiment_id,

            "config": self.config,

            "training_samples": (
                self.training_samples
            ),

            "test_samples": (
                self.test_samples
            ),

            "logical_targets_learned": (
                self.logical_targets_learned
            ),

            "average_target_score": (
                self.average_target_score
            ),

            "exact_accuracy": (
                self.exact_accuracy
            ),

            "physical_accuracy": (
                self.physical_accuracy
            ),

            "bit_accuracy": (
                self.bit_accuracy
            ),

            "logical_accuracy": (
                self.logical_accuracy
            ),

            "training_seconds": (
                self.training_seconds
            ),

            "inference_seconds": (
                self.inference_seconds
            ),

            "samples_per_second": (
                self.samples_per_second
            ),

            "decoder_type": (
                self.decoder_type
            ),
        }