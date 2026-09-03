from collections import Counter, defaultdict

from evaluation.logical_recovery import LogicalRecovery


CORRECTION_PATTERNS = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]


class LogicalTargetBuilder:
    """
    Build correction targets whose objective is
    logical-state preservation.

    The target is learned from training observations
    and their corresponding final physical error states.

    Important:

        Observed syndrome history
                    ↓
            observation group
                    ↓
          possible error states
                    ↓
        evaluate every correction
                    ↓
          choose correction
          with highest logical
             success rate

    Ground-truth error information is used only
    while constructing training targets.

    It must NOT be provided to the decoder as
    an input feature.
    """

    def __init__(self):
        self.recovery = LogicalRecovery()

    @staticmethod
    def observation_key(
        observed_syndrome_history
    ):
        if not observed_syndrome_history:
            raise ValueError(
                "observed_syndrome_history "
                "cannot be empty"
            )

        return "|".join(
            observed_syndrome_history
        )

    @staticmethod
    def state_to_tuple(state):
        return tuple(
            int(bit)
            for bit in state
        )

    @staticmethod
    def xor_states(a, b):
        return [
            int(x) ^ int(y)
            for x, y in zip(a, b)
        ]

    def logical_success_for_correction(
        self,
        actual_error,
        correction
    ):
        """
        Determine whether a correction preserves
        the logical state.

        For the 3-qubit bit-flip code, the logical
        effect depends on the residual error:

            actual_error XOR correction

        Logical success occurs when the residual
        error corresponds to logical identity.
        """

        residual_error = self.xor_states(
            actual_error,
            correction
        )

        recovered_logical = (
            self.recovery.recover(
                residual_error
            )
        )

        return recovered_logical == 0

    def build(
        self,
        training_samples
    ):
        """
        Build one logical-optimal correction
        for each observed syndrome history.
        """

        if not training_samples:
            raise ValueError(
                "training_samples cannot be empty"
            )

        groups = defaultdict(Counter)

        for sample in training_samples:

            observation = self.observation_key(
                sample[
                    "observed_syndrome_history"
                ]
            )

            error_state = self.state_to_tuple(
                sample[
                    "final_error_state"
                ]
            )

            groups[
                observation
            ][
                error_state
            ] += 1

        targets = {}

        scores = {}

        for observation, error_counts in (
            groups.items()
        ):

            total = sum(
                error_counts.values()
            )

            best_correction = None
            best_score = -1.0

            for correction in (
                CORRECTION_PATTERNS
            ):

                success_count = 0

                for (
                    actual_error,
                    count
                ) in error_counts.items():

                    if self.logical_success_for_correction(
                        actual_error,
                        correction
                    ):
                        success_count += count

                score = (
                    success_count
                    / total
                )

                if score > best_score:

                    best_score = score
                    best_correction = correction

            targets[
                observation
            ] = list(
                best_correction
            )

            scores[
                observation
            ] = best_score

        return targets, scores

    def get_target(
        self,
        observation,
        targets
    ):
        """
        Retrieve a learned correction.

        Unknown observations fall back to
        no correction.
        """

        if observation in targets:
            return list(
                targets[observation]
            )

        return [
            0,
            0,
            0
        ]