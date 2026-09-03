from collections import Counter, defaultdict

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)


ROUNDS = 5
PHYSICAL_ERROR_PROBABILITY = 0.10
MEASUREMENT_NOISE_PROBABILITY = 0.10
SAMPLES = 10000
SEED = 42


def encode_observation(sample):
    """
    Encode only information that would be observable
    by the AI decoder.

    Input:

        observed_syndrome_history
        detection_events

    For 5 rounds:

        5 × 2 syndrome bits
        5 × 2 detection bits

        total = 20 features
    """

    features = []

    for syndrome in sample[
        "observed_syndrome_history"
    ]:
        features.extend(
            int(bit)
            for bit in syndrome
        )

    for event in sample[
        "detection_events"
    ]:
        features.extend(
            int(bit)
            for bit in event
        )

    return tuple(features)


def final_error_target(sample):
    """
    Current target:

        final accumulated physical error.
    """

    return tuple(
        sample["final_error_state"]
    )


def transition_target(sample):
    """
    Temporal target.

    Instead of predicting the accumulated
    error state directly, predict the error
    transition that occurred during each round.

    Transition:

        previous_state XOR current_state

    For example:

        previous = [1,0,0]
        current  = [1,1,0]

        transition = [0,1,0]
    """

    history = sample[
        "physical_error_history"
    ]

    previous = [0, 0, 0]

    transitions = []

    for current in history:

        transition = tuple(
            previous[qubit] ^ current[qubit]
            for qubit in range(3)
        )

        transitions.append(
            transition
        )

        previous = current.copy()

    return tuple(
        bit
        for transition in transitions
        for bit in transition
    )


def error_history_target(sample):
    """
    Full accumulated physical error history.

    This is the ground-truth sequence:

        Round 1 -> [q0,q1,q2]
        Round 2 -> [q0,q1,q2]
        ...
        Round N -> [q0,q1,q2]
    """

    return tuple(
        bit
        for state in sample[
            "physical_error_history"
        ]
        for bit in state
    )


def majority_accuracy(observations, targets):
    """
    Calculate the best deterministic prediction
    possible from the exact observation.

    For every identical observation:

        choose the most frequent target.

    This tells us how much information the
    observation representation contains about
    the target.
    """

    mapping = defaultdict(Counter)

    for observation, target in zip(
        observations,
        targets
    ):
        mapping[observation][target] += 1

    correct = 0
    total = len(observations)

    for observation, target_counts in mapping.items():

        best_target, count = (
            target_counts.most_common(1)[0]
        )

        correct += count

    accuracy = correct / total

    return accuracy, mapping


def ambiguity_statistics(mapping):
    """
    Calculate ambiguity statistics.
    """

    ambiguous_observations = 0
    ambiguous_samples = 0
    maximum_targets = 0

    for observation, target_counts in mapping.items():

        number_of_targets = len(
            target_counts
        )

        maximum_targets = max(
            maximum_targets,
            number_of_targets
        )

        if number_of_targets > 1:

            ambiguous_observations += 1

            ambiguous_samples += sum(
                target_counts.values()
            )

    return (
        ambiguous_observations,
        ambiguous_samples,
        maximum_targets
    )


def print_target_distribution(
    name,
    targets,
    max_items=15
):
    """
    Print the most common target patterns.
    """

    counter = Counter(targets)

    print()
    print("-----------------------------------")
    print(name)
    print("-----------------------------------")

    print(
        f"Unique target patterns : "
        f"{len(counter)}"
    )

    for target, count in (
        counter.most_common(max_items)
    ):

        percentage = (
            count / len(targets)
        ) * 100

        print(
            f"{target} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


def analyze_target(
    name,
    observations,
    targets
):
    """
    Analyze how predictable a target is
    from the observable features.
    """

    accuracy, mapping = (
        majority_accuracy(
            observations,
            targets
        )
    )

    (
        ambiguous_observations,
        ambiguous_samples,
        maximum_targets
    ) = ambiguity_statistics(mapping)

    print()
    print(
        f"TARGET: {name}"
    )
    print("-----------------------------------")

    print(
        f"Exact-observation majority accuracy "
        f": {accuracy:.4f}"
    )

    print(
        f"Unique observations              "
        f": {len(mapping)}"
    )

    print(
        f"Ambiguous observations            "
        f": {ambiguous_observations}"
    )

    print(
        f"Samples in ambiguous groups      "
        f": {ambiguous_samples}"
    )

    print(
        f"Maximum targets for one "
        f"observation                      "
        f": {maximum_targets}"
    )

    return accuracy


def compare_targets(
    final_accuracy,
    transition_accuracy,
    history_accuracy
):
    print()
    print("===================================")
    print(" TARGET COMPARISON")
    print("===================================")

    print(
        f"Final error target       : "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Transition target        : "
        f"{transition_accuracy:.4f}"
    )

    print(
        f"Full error history       : "
        f"{history_accuracy:.4f}"
    )

    print()

    scores = {
        "Final error": final_accuracy,
        "Error transition": transition_accuracy,
        "Full error history": history_accuracy
    }

    best_target = max(
        scores,
        key=scores.get
    )

    print(
        f"Most predictable target : "
        f"{best_target}"
    )

    print(
        f"Best information score  : "
        f"{scores[best_target]:.4f}"
    )

    print()
    print("-----------------------------------")
    print(" INTERPRETATION")
    print("-----------------------------------")

    if (
        transition_accuracy
        > final_accuracy
    ):
        print(
            "The temporal transition target "
            "is more predictable than the "
            "current final-error target."
        )

        print()
        print(
            "This suggests that repeated QEC "
            "may be better modeled as a "
            "temporal error-transition "
            "prediction problem."
        )

    elif (
        transition_accuracy
        < final_accuracy
    ):
        print(
            "The current final-error target "
            "is more predictable than the "
            "transition target."
        )

        print()
        print(
            "The current target formulation "
            "may therefore remain useful."
        )

    else:
        print(
            "Both target formulations contain "
            "approximately the same amount "
            "of observable information."
        )

    if history_accuracy > 0.90:

        print()
        print(
            "The full physical error history "
            "is highly predictable from the "
            "current observations."
        )

    elif history_accuracy > 0.70:

        print()
        print(
            "The full physical error history "
            "contains useful temporal information "
            "but remains ambiguous."
        )

    else:

        print()
        print(
            "The full physical error history "
            "has substantial ambiguity."
        )


def main():

    print()
    print("===================================")
    print(" TEMPORAL TARGET DIAGNOSTIC")
    print("===================================")

    print()
    print(
        f"Rounds                    : {ROUNDS}"
    )

    print(
        f"Physical X noise          : "
        f"{PHYSICAL_ERROR_PROBABILITY}"
    )

    print(
        f"Measurement noise         : "
        f"{MEASUREMENT_NOISE_PROBABILITY}"
    )

    print(
        f"Samples                   : {SAMPLES}"
    )

    print(
        f"Random seed               : {SEED}"
    )

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=SEED
        )
    )

    samples = generator.generate_dataset(
        SAMPLES
    )

    print()
    print(
        f"Samples generated         : "
        f"{len(samples)}"
    )

    # ------------------------------------------------
    # Observable features
    # ------------------------------------------------

    observations = [
        encode_observation(sample)
        for sample in samples
    ]

    feature_lengths = sorted(
        set(
            len(observation)
            for observation in observations
        )
    )

    print()
    print("-----------------------------------")
    print("OBSERVABLE REPRESENTATION")
    print("-----------------------------------")

    print(
        f"Feature lengths found     : "
        f"{feature_lengths}"
    )

    expected_features = (
        ROUNDS * 2
        +
        ROUNDS * 2
    )

    print(
        f"Expected feature count    : "
        f"{expected_features}"
    )

    if feature_lengths == [
        expected_features
    ]:

        print(
            "Observable representation : PASS"
        )

    else:

        print(
            "Observable representation : FAIL"
        )

    # ------------------------------------------------
    # Targets
    # ------------------------------------------------

    final_targets = [
        final_error_target(sample)
        for sample in samples
    ]

    transition_targets = [
        transition_target(sample)
        for sample in samples
    ]

    history_targets = [
        error_history_target(sample)
        for sample in samples
    ]

    print_target_distribution(
        "FINAL ERROR TARGET",
        final_targets
    )

    print_target_distribution(
        "ERROR TRANSITION TARGET",
        transition_targets
    )

    print_target_distribution(
        "FULL ERROR HISTORY TARGET",
        history_targets
    )

    # ------------------------------------------------
    # Information analysis
    # ------------------------------------------------

    print()
    print("===================================")
    print("OBSERVATION → TARGET INFORMATION")
    print("===================================")

    final_accuracy = analyze_target(
        "FINAL ERROR",
        observations,
        final_targets
    )

    transition_accuracy = analyze_target(
        "ERROR TRANSITIONS",
        observations,
        transition_targets
    )

    history_accuracy = analyze_target(
        "FULL ERROR HISTORY",
        observations,
        history_targets
    )

    # ------------------------------------------------
    # Compare
    # ------------------------------------------------

    compare_targets(
        final_accuracy,
        transition_accuracy,
        history_accuracy
    )

    # ------------------------------------------------
    # Example
    # ------------------------------------------------

    print()
    print("===================================")
    print("EXAMPLE TEMPORAL TARGET")
    print("===================================")

    example = samples[0]

    print()
    print(
        "Observed syndrome history:"
    )

    print(
        example[
            "observed_syndrome_history"
        ]
    )

    print()
    print(
        "Detection events:"
    )

    print(
        example[
            "detection_events"
        ]
    )

    print()
    print(
        "Physical error history:"
    )

    for round_index, state in enumerate(
        example[
            "physical_error_history"
        ],
        start=1
    ):

        print(
            f"Round {round_index}: {state}"
        )

    print()
    print(
        "Derived error transitions:"
    )

    previous = [0, 0, 0]

    for round_index, current in enumerate(
        example[
            "physical_error_history"
        ],
        start=1
    ):

        transition = [
            previous[q] ^ current[q]
            for q in range(3)
        ]

        print(
            f"Round {round_index}: "
            f"{transition}"
        )

        previous = current.copy()

    print()
    print(
        "Final error state:"
    )

    print(
        example[
            "final_error_state"
        ]
    )

    print()
    print("===================================")
    print(
        "TEMPORAL TARGET DIAGNOSTIC : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()