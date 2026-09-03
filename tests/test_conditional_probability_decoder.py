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

TOTAL_SAMPLES = 25000
TRAINING_SIZE = 20000
TEST_SAMPLES = 5000

SEED = 42


# ============================================================
# FEATURE REPRESENTATION
# ============================================================

def calculate_detection_events(
    observed_syndrome_history
):

    detection_events = []

    previous = "00"

    for syndrome in observed_syndrome_history:

        event = (
            str(
                int(previous[0])
                ^ int(syndrome[0])
            )
            +
            str(
                int(previous[1])
                ^ int(syndrome[1])
            )
        )

        detection_events.append(event)

        previous = syndrome

    return detection_events


def encode_observation(sample):

    observed_history = (
        sample["observed_syndrome_history"]
    )

    detection_events = (
        calculate_detection_events(
            observed_history
        )
    )

    parts = []

    for syndrome, detection in zip(
        observed_history,
        detection_events
    ):

        parts.append(syndrome)
        parts.append(detection)

    return "|".join(parts)


# ============================================================
# TARGET
# ============================================================

def encode_target(sample):

    return tuple(
        int(bit)
        for bit in sample["final_error_state"]
    )


# ============================================================
# CONDITIONAL-PROBABILITY DECODER
# ============================================================

class ConditionalProbabilityDecoder:

    def __init__(self):

        self.observation_counts = (
            defaultdict(Counter)
        )

        self.global_counts = Counter()

    def train(
        self,
        observations,
        targets
    ):

        if len(observations) != len(targets):

            raise ValueError(
                "observations and targets "
                "must have the same length"
            )

        for observation, target in zip(
            observations,
            targets
        ):

            self.observation_counts[
                observation
            ][target] += 1

            self.global_counts[
                target
            ] += 1

        return self

    def predict_one(
        self,
        observation
    ):

        counts = (
            self.observation_counts.get(
                observation
            )
        )

        if counts:

            return counts.most_common(1)[0][0]

        # Fallback for an observation that
        # never appeared during training.
        return self.global_counts.most_common(
            1
        )[0][0]

    def predict(
        self,
        observations
    ):

        return [
            self.predict_one(observation)
            for observation in observations
        ]

    def probability_distribution(
        self,
        observation
    ):

        counts = (
            self.observation_counts.get(
                observation
            )
        )

        if not counts:

            total = sum(
                self.global_counts.values()
            )

            if total == 0:
                return {}

            return {
                target: count / total
                for target, count
                in self.global_counts.items()
            }

        total = sum(counts.values())

        return {
            target: count / total
            for target, count
            in counts.items()
        }


# ============================================================
# METRICS
# ============================================================

def exact_pattern_accuracy(
    predictions,
    targets
):

    correct = 0

    for prediction, target in zip(
        predictions,
        targets
    ):

        if tuple(prediction) == tuple(target):

            correct += 1

    return correct / len(targets)


def bit_accuracy(
    predictions,
    targets
):

    correct = 0
    total = 0

    for prediction, target in zip(
        predictions,
        targets
    ):

        for predicted_bit, target_bit in zip(
            prediction,
            target
        ):

            if predicted_bit == target_bit:
                correct += 1

            total += 1

    return correct / total


# ============================================================
# EMPIRICAL INFORMATION CEILING
# ============================================================

def calculate_information_ceiling(
    observations,
    targets
):

    groups = defaultdict(Counter)

    for observation, target in zip(
        observations,
        targets
    ):

        groups[observation][target] += 1

    correct_if_perfect = 0

    for counts in groups.values():

        correct_if_perfect += (
            counts.most_common(1)[0][1]
        )

    return (
        correct_if_perfect
        / len(targets)
    )


# ============================================================
# OBSERVATION AMBIGUITY
# ============================================================

def analyze_ambiguity(
    observations,
    targets
):

    groups = defaultdict(Counter)

    for observation, target in zip(
        observations,
        targets
    ):

        groups[observation][target] += 1

    total_groups = len(groups)

    ambiguous_groups = 0
    ambiguous_samples = 0
    maximum_targets = 0

    for counts in groups.values():

        number_of_targets = len(counts)

        maximum_targets = max(
            maximum_targets,
            number_of_targets
        )

        if number_of_targets > 1:

            ambiguous_groups += 1

            ambiguous_samples += sum(
                counts.values()
            )

    return {
        "unique_observations": total_groups,
        "ambiguous_groups": ambiguous_groups,
        "ambiguous_samples": ambiguous_samples,
        "maximum_targets": maximum_targets
    }


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def analyze_confidence(
    decoder,
    observations,
    targets
):

    buckets = {
        "0.50-0.59": [],
        "0.60-0.69": [],
        "0.70-0.79": [],
        "0.80-0.89": [],
        "0.90-0.99": [],
        "1.00": []
    }

    for observation, target in zip(
        observations,
        targets
    ):

        probabilities = (
            decoder.probability_distribution(
                observation
            )
        )

        if not probabilities:
            continue

        predicted = max(
            probabilities,
            key=probabilities.get
        )

        confidence = probabilities[
            predicted
        ]

        correct = (
            predicted == target
        )

        if confidence >= 1.0:
            bucket = "1.00"
        elif confidence >= 0.90:
            bucket = "0.90-0.99"
        elif confidence >= 0.80:
            bucket = "0.80-0.89"
        elif confidence >= 0.70:
            bucket = "0.70-0.79"
        elif confidence >= 0.60:
            bucket = "0.60-0.69"
        else:
            bucket = "0.50-0.59"

        buckets[bucket].append(
            correct
        )

    return buckets


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================================")
    print(" CONDITIONAL-PROBABILITY DECODER")
    print("==============================================")

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
        f"Training samples          : "
        f"{TRAINING_SIZE}"
    )

    print(
        f"Test samples              : "
        f"{TEST_SAMPLES}"
    )

    print(
        f"Random seed               : "
        f"{SEED}"
    )

    # ========================================================
    # GENERATE DATASET
    # ========================================================

    print()
    print("----------------------------------------------")
    print("GENERATING FIXED DATASET")
    print("----------------------------------------------")

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
        TOTAL_SAMPLES
    )

    print(
        f"Generated samples        : "
        f"{len(samples)}"
    )

    # ========================================================
    # FIXED SPLIT
    # ========================================================

    train_samples = samples[
        :TRAINING_SIZE
    ]

    test_samples = samples[
        TRAINING_SIZE:
    ]

    train_observations = [
        encode_observation(sample)
        for sample in train_samples
    ]

    train_targets = [
        encode_target(sample)
        for sample in train_samples
    ]

    test_observations = [
        encode_observation(sample)
        for sample in test_samples
    ]

    test_targets = [
        encode_target(sample)
        for sample in test_samples
    ]

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TRAINING CONDITIONAL-PROBABILITY DECODER")
    print("----------------------------------------------")

    decoder = (
        ConditionalProbabilityDecoder()
    )

    decoder.train(
        train_observations,
        train_targets
    )

    # ========================================================
    # INFORMATION STATISTICS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("OBSERVATION STATISTICS")
    print("----------------------------------------------")

    ambiguity = analyze_ambiguity(
        train_observations,
        train_targets
    )

    print(
        f"Unique observations       : "
        f"{ambiguity['unique_observations']}"
    )

    print(
        f"Ambiguous observations    : "
        f"{ambiguity['ambiguous_groups']}"
    )

    print(
        f"Samples in ambiguous groups: "
        f"{ambiguity['ambiguous_samples']}"
    )

    print(
        f"Maximum target patterns   : "
        f"{ambiguity['maximum_targets']}"
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    train_predictions = (
        decoder.predict(
            train_observations
        )
    )

    test_predictions = (
        decoder.predict(
            test_observations
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    train_exact = (
        exact_pattern_accuracy(
            train_predictions,
            train_targets
        )
    )

    train_bit = (
        bit_accuracy(
            train_predictions,
            train_targets
        )
    )

    test_exact = (
        exact_pattern_accuracy(
            test_predictions,
            test_targets
        )
    )

    test_bit = (
        bit_accuracy(
            test_predictions,
            test_targets
        )
    )

    # ========================================================
    # EMPIRICAL CEILING
    # ========================================================

    ceiling = (
        calculate_information_ceiling(
            test_observations,
            test_targets
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("==============================================")
    print(" MODEL PERFORMANCE")
    print("==============================================")

    print()
    print(
        f"Train exact              : "
        f"{train_exact:.4f}"
    )

    print(
        f"Train bit                : "
        f"{train_bit:.4f}"
    )

    print(
        f"Test exact               : "
        f"{test_exact:.4f}"
    )

    print(
        f"Test bit                 : "
        f"{test_bit:.4f}"
    )

    print()
    print(
        f"Empirical test ceiling   : "
        f"{ceiling:.4f}"
    )

    print(
        f"Gap to ceiling           : "
        f"{ceiling - test_exact:.4f}"
    )

    # ========================================================
    # CONFIDENCE ANALYSIS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("CONFIDENCE ANALYSIS")
    print("----------------------------------------------")

    confidence_buckets = (
        analyze_confidence(
            decoder,
            test_observations,
            test_targets
        )
    )

    for bucket, values in (
        confidence_buckets.items()
    ):

        if not values:
            print(
                f"{bucket}: no samples"
            )
            continue

        accuracy = (
            sum(values)
            / len(values)
        )

        print(
            f"{bucket}: "
            f"samples={len(values):4d} "
            f"accuracy={accuracy:.4f}"
        )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("----------------------------------------------")
    print("TEST PREDICTION DISTRIBUTION")
    print("----------------------------------------------")

    prediction_counter = Counter(
        test_predictions
    )

    for pattern, count in (
        prediction_counter.most_common()
    ):

        percentage = (
            count
            / len(test_predictions)
            * 100
        )

        print(
            f"{pattern} -> "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # SAMPLE PROBABILITIES
    # ========================================================

    print()
    print("----------------------------------------------")
    print("SAMPLE CONDITIONAL PROBABILITIES")
    print("----------------------------------------------")

    shown = 0

    seen = set()

    for observation, target in zip(
        test_observations,
        test_targets
    ):

        if observation in seen:
            continue

        seen.add(observation)

        probabilities = (
            decoder.probability_distribution(
                observation
            )
        )

        ordered = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )

        print()
        print(
            f"Actual target: {target}"
        )

        print(
            "P(error | observation):"
        )

        for pattern, probability in (
            ordered
        ):

            print(
                f"  {pattern}: "
                f"{probability:.4f}"
            )

        shown += 1

        if shown >= 5:
            break

    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print("==============================================")
    print(" DECODER COMPARISON")
    print("==============================================")

    print()
    print(
        "Conditional-probability decoder"
    )

    print(
        f"Exact = {test_exact:.4f}"
    )

    print(
        f"Bit   = {test_bit:.4f}"
    )

    print()
    print(
        "Previous best GRU"
    )

    print(
        "Exact = 0.6222"
    )

    print(
        "Bit   = 0.7357"
    )

    print()
    print(
        "Empirical information ceiling"
    )

    print(
        f"Exact = {ceiling:.4f}"
    )

    print()
    print("----------------------------------------------")
    print("COMPARISON WITH GRU")
    print("----------------------------------------------")

    print(
        f"Exact difference vs GRU : "
        f"{test_exact - 0.6222:+.4f}"
    )

    print(
        f"Bit difference vs GRU   : "
        f"{test_bit - 0.7357:+.4f}"
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    print()
    print("----------------------------------------------")
    print("DIAGNOSIS")
    print("----------------------------------------------")

    if test_exact >= 0.65:

        print()
        print(
            "STRONG CONDITIONAL-PROBABILITY RESULT"
        )

        print(
            "The observation contains substantial "
            "predictive information."
        )

        print(
            "The GRU still has meaningful room "
            "for improvement."
        )

    elif test_exact >= 0.62:

        print()
        print(
            "GRU IS CLOSE TO THE PROBABILISTIC BASELINE"
        )

        print(
            "The current GRU is already learning "
            "a large portion of the available "
            "predictive information."
        )

    else:

        print()
        print(
            "GRU OUTPERFORMS SIMPLE CONDITIONAL "
            "PROBABILITY"
        )

        print(
            "The temporal neural model is extracting "
            "patterns that the exact-observation "
            "lookup cannot capture reliably."
        )

    print()
    print("==============================================")
    print(
        " CONDITIONAL-PROBABILITY DECODER : COMPLETE"
    )
    print("==============================================")


if __name__ == "__main__":
    main()