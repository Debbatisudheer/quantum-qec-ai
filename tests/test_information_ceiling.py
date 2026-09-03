from collections import Counter, defaultdict

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)


ROUNDS = 5
SAMPLES = 10000
SEED = 42

# Controlled noise configurations.
NOISE_CONFIGURATIONS = [
    {
        "physical": 0.10,
        "measurement": 0.00,
    },
    {
        "physical": 0.10,
        "measurement": 0.05,
    },
    {
        "physical": 0.10,
        "measurement": 0.10,
    },
    {
        "physical": 0.05,
        "measurement": 0.10,
    },
    {
        "physical": 0.01,
        "measurement": 0.10,
    },
]


def encode_observation(sample):
    """
    Encode only information available to the decoder.

    Features:

        observed syndrome history
        +
        detection-event history

    For 5 rounds:

        5 × 2 syndrome bits = 10
        5 × 2 event bits    = 10

        Total = 20 features.
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


def encode_target(sample):
    """
    Target = final accumulated physical
    X-error state.
    """

    return tuple(
        sample["final_error_state"]
    )


def calculate_information_ceiling(
    observations,
    targets
):
    """
    Calculate the best deterministic prediction
    possible using the exact observable feature
    vector.

    For each identical observation:

        choose its most common target.

    This is the empirical information ceiling
    for deterministic prediction on this dataset.
    """

    mapping = defaultdict(Counter)

    for observation, target in zip(
        observations,
        targets
    ):

        mapping[observation][target] += 1

    correct = 0

    for target_counts in mapping.values():

        _, best_count = (
            target_counts.most_common(1)[0]
        )

        correct += best_count

    total = len(observations)

    accuracy = correct / total

    ambiguous_observations = 0
    ambiguous_samples = 0
    maximum_targets = 0

    for target_counts in mapping.values():

        target_count = len(
            target_counts
        )

        maximum_targets = max(
            maximum_targets,
            target_count
        )

        if target_count > 1:

            ambiguous_observations += 1

            ambiguous_samples += sum(
                target_counts.values()
            )

    return {
        "accuracy": accuracy,
        "unique_observations": len(mapping),
        "ambiguous_observations": (
            ambiguous_observations
        ),
        "ambiguous_samples": (
            ambiguous_samples
        ),
        "maximum_targets": (
            maximum_targets
        ),
    }


def run_configuration(
    physical_noise,
    measurement_noise
):
    """
    Generate a dataset under one controlled
    noise configuration and calculate the
    information ceiling.
    """

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                physical_noise
            ),
            measurement_noise_probability=(
                measurement_noise
            ),
            seed=SEED
        )
    )

    samples = generator.generate_dataset(
        SAMPLES
    )

    observations = [
        encode_observation(sample)
        for sample in samples
    ]

    targets = [
        encode_target(sample)
        for sample in samples
    ]

    result = calculate_information_ceiling(
        observations,
        targets
    )

    result["physical_noise"] = (
        physical_noise
    )

    result["measurement_noise"] = (
        measurement_noise
    )

    return result


def print_configuration_result(result):
    print()
    print(
        f"Physical noise     : "
        f"{result['physical_noise']:.2f}"
    )

    print(
        f"Measurement noise  : "
        f"{result['measurement_noise']:.2f}"
    )

    print(
        f"Information ceiling: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Unique observations: "
        f"{result['unique_observations']}"
    )

    print(
        f"Ambiguous groups   : "
        f"{result['ambiguous_observations']}"
    )

    print(
        f"Ambiguous samples  : "
        f"{result['ambiguous_samples']}"
    )

    print(
        f"Maximum targets    : "
        f"{result['maximum_targets']}"
    )


def print_summary(results):
    print()
    print("===================================")
    print(" INFORMATION CEILING SUMMARY")
    print("===================================")

    print()
    print(
        "Physical   Measurement   Ceiling"
    )

    print(
        "Noise      Noise         Accuracy"
    )

    print(
        "-----------------------------------"
    )

    for result in results:

        print(
            f"{result['physical_noise']:<10.2f}"
            f"{result['measurement_noise']:<14.2f}"
            f"{result['accuracy']:.4f}"
        )


def interpret_results(results):
    print()
    print("===================================")
    print(" INTERPRETATION")
    print("===================================")

    # -----------------------------------------
    # Measurement-noise effect at physical=0.10
    # -----------------------------------------

    physical_010 = [
        result
        for result in results
        if result["physical_noise"] == 0.10
    ]

    no_measurement_noise = next(
        (
            result
            for result in physical_010
            if result["measurement_noise"] == 0.00
        ),
        None
    )

    measurement_010 = next(
        (
            result
            for result in physical_010
            if result["measurement_noise"] == 0.10
        ),
        None
    )

    if (
        no_measurement_noise is not None
        and measurement_010 is not None
    ):

        difference = (
            no_measurement_noise["accuracy"]
            -
            measurement_010["accuracy"]
        )

        print()
        print(
            "Measurement-noise effect "
            f"(physical noise = 0.10): "
            f"{difference:+.4f}"
        )

        if difference > 0.05:

            print(
                "Measurement noise has a "
                "strong effect on the "
                "information ceiling."
            )

        elif difference > 0.01:

            print(
                "Measurement noise has a "
                "moderate effect on the "
                "information ceiling."
            )

        else:

            print(
                "Measurement noise has only "
                "a small effect on the "
                "information ceiling."
            )

    # -----------------------------------------
    # Physical-noise effect
    # -----------------------------------------

    physical_001 = next(
        (
            result
            for result in results
            if (
                result["physical_noise"] == 0.01
                and
                result["measurement_noise"] == 0.10
            )
        ),
        None
    )

    physical_005 = next(
        (
            result
            for result in results
            if (
                result["physical_noise"] == 0.05
                and
                result["measurement_noise"] == 0.10
            )
        ),
        None
    )

    physical_010_measurement = next(
        (
            result
            for result in results
            if (
                result["physical_noise"] == 0.10
                and
                result["measurement_noise"] == 0.10
            )
        ),
        None
    )

    if (
        physical_001 is not None
        and physical_005 is not None
        and physical_010_measurement is not None
    ):

        print()
        print(
            "Physical-noise sweep:"
        )

        print(
            f"0.01 → "
            f"{physical_001['accuracy']:.4f}"
        )

        print(
            f"0.05 → "
            f"{physical_005['accuracy']:.4f}"
        )

        print(
            f"0.10 → "
            f"{physical_010_measurement['accuracy']:.4f}"
        )

        print()

        if (
            physical_001["accuracy"]
            >
            physical_005["accuracy"]
            >
            physical_010_measurement["accuracy"]
        ):

            print(
                "Higher physical noise produces "
                "greater target ambiguity."
            )

        else:

            print(
                "The physical-noise relationship "
                "is not strictly monotonic in "
                "this experiment."
            )

    # -----------------------------------------
    # Final conclusion
    # -----------------------------------------

    print()
    print("-----------------------------------")
    print(" RESEARCH CONCLUSION")
    print("-----------------------------------")

    best_result = max(
        results,
        key=lambda result: result["accuracy"]
    )

    worst_result = min(
        results,
        key=lambda result: result["accuracy"]
    )

    print(
        "Best information ceiling:"
    )

    print(
        f"Physical={best_result['physical_noise']:.2f}, "
        f"Measurement="
        f"{best_result['measurement_noise']:.2f}, "
        f"Accuracy="
        f"{best_result['accuracy']:.4f}"
    )

    print()

    print(
        "Worst information ceiling:"
    )

    print(
        f"Physical={worst_result['physical_noise']:.2f}, "
        f"Measurement="
        f"{worst_result['measurement_noise']:.2f}, "
        f"Accuracy="
        f"{worst_result['accuracy']:.4f}"
    )

    print()
    print(
        "This experiment measures the information "
        "available to the decoder before any AI "
        "model is trained."
    )


def main():

    print()
    print("===================================")
    print(" INFORMATION CEILING EXPERIMENT")
    print("===================================")

    print()
    print(
        f"Rounds             : {ROUNDS}"
    )

    print(
        f"Samples            : {SAMPLES}"
    )

    print(
        f"Random seed        : {SEED}"
    )

    print()
    print(
        "The experiment changes noise conditions "
        "while keeping the observable representation "
        "and target definition unchanged."
    )

    results = []

    for configuration in NOISE_CONFIGURATIONS:

        print()
        print("===================================")
        print("NOISE CONFIGURATION")
        print("===================================")

        result = run_configuration(
            physical_noise=(
                configuration["physical"]
            ),
            measurement_noise=(
                configuration["measurement"]
            )
        )

        print_configuration_result(
            result
        )

        results.append(result)

    print_summary(results)

    interpret_results(results)

    print()
    print("===================================")
    print(
        "INFORMATION CEILING EXPERIMENT : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()