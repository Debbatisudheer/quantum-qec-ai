from collections import Counter, defaultdict

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITY = 0.10

SAMPLES = 10000

SEED = 42


# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_features(sample):
    """
    Encode exactly the same observable information
    used by the AI decoders.

    5 rounds:

        observed syndrome history = 10 bits
        detection events          = 10 bits

        total                     = 20 features
    """

    features = []

    for syndrome in sample[
        "observed_syndrome_history"
    ]:

        for bit in syndrome:
            features.append(int(bit))

    for event in sample[
        "detection_events"
    ]:

        for bit in event:
            features.append(int(bit))

    return tuple(features)


def encode_target(sample):
    """
    Final physical error state.
    """

    return tuple(
        sample["final_error_state"]
    )


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_samples():
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

    return generator.generate_dataset(
        num_samples=SAMPLES
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def analyze_target_distribution(samples):

    counts = Counter(
        encode_target(sample)
        for sample in samples
    )

    print()
    print("-----------------------------------")
    print("FINAL ERROR DISTRIBUTION")
    print("-----------------------------------")

    total = len(samples)

    for target, count in sorted(
        counts.items()
    ):

        percentage = (
            count / total * 100
        )

        print(
            f"{list(target)}"
            f" -> {count:5d}"
            f" ({percentage:6.2f}%)"
        )

    print(
        f"Unique error patterns : "
        f"{len(counts)}"
    )

    return counts


# ============================================================
# OBSERVED SYNDROME DISTRIBUTION
# ============================================================

def analyze_syndrome_distribution(samples):

    counts = Counter()

    for sample in samples:

        final_syndrome = sample[
            "observed_syndrome_history"
        ][-1]

        counts[final_syndrome] += 1

    print()
    print("-----------------------------------")
    print("FINAL OBSERVED SYNDROME DISTRIBUTION")
    print("-----------------------------------")

    total = len(samples)

    for syndrome, count in sorted(
        counts.items()
    ):

        percentage = (
            count / total * 100
        )

        print(
            f"{syndrome}"
            f" -> {count:5d}"
            f" ({percentage:6.2f}%)"
        )

    return counts


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

def analyze_feature_distribution(samples):

    feature_lengths = set()

    for sample in samples:

        features = encode_features(
            sample
        )

        feature_lengths.add(
            len(features)
        )

    print()
    print("-----------------------------------")
    print("FEATURE REPRESENTATION")
    print("-----------------------------------")

    print(
        f"Feature lengths found : "
        f"{sorted(feature_lengths)}"
    )

    expected = ROUNDS * 4

    print(
        f"Expected feature count : "
        f"{expected}"
    )

    if feature_lengths == {expected}:

        print(
            "Feature representation : PASS"
        )

    else:

        print(
            "Feature representation : FAIL"
        )


# ============================================================
# FEATURE → TARGET AMBIGUITY
# ============================================================

def analyze_feature_target_relationship(
    samples
):
    """
    Determine whether identical observable
    feature vectors correspond to multiple
    ground-truth targets.

    This is the critical diagnostic.

    If:

        same features
             ↓
        multiple targets

    then the decoder cannot perfectly recover
    the target from those features alone.
    """

    mapping = defaultdict(Counter)

    for sample in samples:

        features = encode_features(
            sample
        )

        target = encode_target(
            sample
        )

        mapping[features][target] += 1

    ambiguous_features = 0

    total_feature_groups = len(
        mapping
    )

    samples_in_ambiguous_groups = 0

    maximum_target_options = 0

    for features, target_counts in (
        mapping.items()
    ):

        number_of_targets = len(
            target_counts
        )

        maximum_target_options = max(
            maximum_target_options,
            number_of_targets
        )

        if number_of_targets > 1:

            ambiguous_features += 1

            samples_in_ambiguous_groups += (
                sum(
                    target_counts.values()
                )
            )

    print()
    print("-----------------------------------")
    print("FEATURE → TARGET RELATIONSHIP")
    print("-----------------------------------")

    print(
        f"Unique feature vectors : "
        f"{total_feature_groups}"
    )

    print(
        f"Ambiguous feature vectors : "
        f"{ambiguous_features}"
    )

    print(
        f"Samples in ambiguous groups : "
        f"{samples_in_ambiguous_groups}"
    )

    print(
        f"Maximum target patterns "
        f"for one feature vector : "
        f"{maximum_target_options}"
    )

    if ambiguous_features == 0:

        print()
        print(
            "Information ambiguity : "
            "NONE DETECTED"
        )

    else:

        print()
        print(
            "Information ambiguity : "
            "DETECTED"
        )

    return mapping


# ============================================================
# BAYES / MAJORITY BASELINE
# ============================================================

def calculate_feature_majority_accuracy(
    mapping
):
    """
    Calculate the best possible training-set
    accuracy obtainable by predicting the most
    common target for each exact feature vector.

    This is NOT a learned AI model.

    It is an information diagnostic.

    If a feature vector has:

        target A = 70%
        target B = 30%

    then even a perfect deterministic decoder
    using only those features cannot exceed 70%
    on those samples.
    """

    correct = 0

    total = 0

    for target_counts in mapping.values():

        best_count = max(
            target_counts.values()
        )

        correct += best_count

        total += sum(
            target_counts.values()
        )

    if total == 0:

        return 0.0

    return correct / total


# ============================================================
# FINAL SYNDROME → TARGET RELATIONSHIP
# ============================================================

def analyze_final_syndrome_target_relationship(
    samples
):
    """
    Measure ambiguity when using only the
    final observed syndrome.

    This approximates what the traditional
    lookup decoder sees.
    """

    mapping = defaultdict(Counter)

    for sample in samples:

        syndrome = sample[
            "observed_syndrome_history"
        ][-1]

        target = encode_target(
            sample
        )

        mapping[
            syndrome
        ][target] += 1

    print()
    print("-----------------------------------")
    print(
        "FINAL SYNDROME → ERROR RELATIONSHIP"
    )
    print("-----------------------------------")

    for syndrome in sorted(
        mapping.keys()
    ):

        counts = mapping[
            syndrome
        ]

        total = sum(
            counts.values()
        )

        print()

        print(
            f"Syndrome {syndrome}"
            f" ({total} samples)"
        )

        for target, count in (
            counts.most_common()
        ):

            percentage = (
                count / total * 100
            )

            print(
                f"    {list(target)}"
                f" -> {percentage:6.2f}%"
            )

    # Majority prediction accuracy

    correct = 0
    total_samples = len(samples)

    for counts in mapping.values():

        correct += max(
            counts.values()
        )

    accuracy = (
        correct / total_samples
    )

    print()

    print(
        "Final-syndrome majority "
        f"accuracy : {accuracy:.4f}"
    )

    return mapping, accuracy


# ============================================================
# SAMPLE INSPECTION
# ============================================================

def show_examples(samples):

    print()
    print("-----------------------------------")
    print("EXAMPLE SAMPLES")
    print("-----------------------------------")

    for index, sample in enumerate(
        samples[:5],
        start=1
    ):

        print()
        print(
            f"Sample {index}"
        )

        print(
            "Observed syndrome history : "
            f"{sample['observed_syndrome_history']}"
        )

        print(
            "Detection events           : "
            f"{sample['detection_events']}"
        )

        print(
            "Final error state          : "
            f"{sample['final_error_state']}"
        )

        print(
            "Logical state              : "
            f"{sample['logical_state']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("===================================")
    print(
        " INFORMATION / TARGET DIAGNOSTIC"
    )
    print("===================================")

    print()

    print(
        f"Rounds                    : "
        f"{ROUNDS}"
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
        f"Samples                   : "
        f"{SAMPLES}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    print()

    # --------------------------------------------------------
    # Generate dataset
    # --------------------------------------------------------

    samples = generate_samples()

    print(
        f"Samples generated         : "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Basic representation checks
    # --------------------------------------------------------

    analyze_feature_distribution(
        samples
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    analyze_target_distribution(
        samples
    )

    # --------------------------------------------------------
    # Syndrome distribution
    # --------------------------------------------------------

    analyze_syndrome_distribution(
        samples
    )

    # --------------------------------------------------------
    # Feature → target relationship
    # --------------------------------------------------------

    mapping = (
        analyze_feature_target_relationship(
            samples
        )
    )

    majority_accuracy = (
        calculate_feature_majority_accuracy(
            mapping
        )
    )

    print()
    print(
        "Exact-feature majority accuracy : "
        f"{majority_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Final syndrome relationship
    # --------------------------------------------------------

    (
        syndrome_mapping,
        syndrome_accuracy
    ) = (
        analyze_final_syndrome_target_relationship(
            samples
        )
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    show_examples(
        samples
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("===================================")
    print(" DIAGNOSTIC INTERPRETATION")
    print("===================================")

    print()

    if majority_accuracy < 0.70:

        print(
            "The current observable features "
            "have substantial target ambiguity."
        )

        print(
            "A more powerful AI model alone "
            "is unlikely to solve the problem."
        )

    elif majority_accuracy < 0.90:

        print(
            "The current observable features "
            "contain useful information, "
            "but significant ambiguity remains."
        )

    else:

        print(
            "The current observable features "
            "strongly constrain the target."
        )

    print()

    if syndrome_accuracy < majority_accuracy:

        print(
            "The full temporal feature set "
            "contains more information than "
            "the final syndrome alone."
        )

    else:

        print(
            "The temporal features are not "
            "showing a clear advantage over "
            "the final syndrome in this dataset."
        )

    print()

    print("===================================")
    print(
        "INFORMATION DIAGNOSTIC : SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()