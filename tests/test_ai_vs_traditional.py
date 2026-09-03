from sklearn.model_selection import train_test_split

from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)

from decoders.repeated_lookup import (
    RepeatedLookupDecoder
)

from experiments.stochastic_quantum_ai_experiment import (
    StochasticQuantumAIExperiment
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITY = 0.10

TRAINING_SAMPLES = 5000

QUANTUM_TEST_TRIALS = 1000

SEEDS = [
    42,
    123,
    456
]


# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_sample_features(sample):
    """
    Convert observable syndrome information
    into the 20-feature representation.

    For 5 rounds:

        observed syndrome history
            = 5 × 2 = 10 bits

        detection events
            = 5 × 2 = 10 bits

        total
            = 20 features
    """

    features = []

    observed_syndrome_history = (
        sample[
            "observed_syndrome_history"
        ]
    )

    detection_events = (
        sample[
            "detection_events"
        ]
    )

    for syndrome in (
        observed_syndrome_history
    ):

        for bit in syndrome:

            features.append(
                int(bit)
            )

    for event in detection_events:

        for bit in event:

            features.append(
                int(bit)
            )

    return features


def encode_sample_target(sample):
    """
    Ground-truth final physical error pattern.

    This is the supervised-learning target.

    It is NOT provided to the decoder.
    """

    return list(
        sample[
            "final_error_state"
        ]
    )


# ============================================================
# DATASET CREATION
# ============================================================

def create_training_dataset(seed):
    """
    Generate a fresh supervised training dataset.
    """

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=ROUNDS,
            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),
            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),
            seed=seed
        )
    )

    samples = (
        generator.generate_dataset(
            num_samples=TRAINING_SAMPLES
        )
    )

    X = [
        encode_sample_features(
            sample
        )
        for sample in samples
    ]

    y = [
        encode_sample_target(
            sample
        )
        for sample in samples
    ]

    return X, y


def train_ai_decoder(
    decoder_class,
    seed
):
    """
    Generate training data and train
    one AI decoder.
    """

    X, y = create_training_dataset(
        seed
    )

    decoder = decoder_class()

    decoder.train(
        X,
        y
    )

    return decoder


# ============================================================
# TRADITIONAL DECODER ADAPTER
# ============================================================

class TraditionalDecoderAdapter:
    """
    Adapter allowing the traditional repeated
    lookup decoder to use the same interface
    as the AI decoders.

    The stochastic experiment supplies a
    20-feature vector:

        10 observed syndrome bits
        +
        10 detection-event bits

    The traditional decoder intentionally
    ignores the detection-event features and
    uses only the final observed syndrome.

    Feature layout for 5 rounds:

        Round 1 syndrome -> features 0,1
        Round 2 syndrome -> features 2,3
        Round 3 syndrome -> features 4,5
        Round 4 syndrome -> features 6,7
        Round 5 syndrome -> features 8,9

    Therefore the final observed syndrome is:

        features[8:10]
    """

    def __init__(self):

        self.decoder = (
            RepeatedLookupDecoder()
        )

    def decode(self, features):

        if len(features) != (
            ROUNDS * 4
        ):
            raise ValueError(
                "Unexpected feature count"
            )

        final_syndrome = (
            str(features[8])
            + str(features[9])
        )

        return (
            self.decoder.decode(
                final_syndrome
            )
        )


# ============================================================
# SINGLE DECODER / SINGLE SEED
# ============================================================

def run_single_seed(
    decoder_name,
    decoder,
    seed
):
    """
    Run one decoder against an independent
    stochastic quantum test set.
    """

    experiment = (
        StochasticQuantumAIExperiment(
            rounds=ROUNDS,

            physical_error_probability=(
                PHYSICAL_ERROR_PROBABILITY
            ),

            measurement_noise_probability=(
                MEASUREMENT_NOISE_PROBABILITY
            ),

            seed=seed + 1000
        )
    )

    result = (
        experiment.run_experiment(
            decoder=decoder,
            num_trials=QUANTUM_TEST_TRIALS
        )
    )

    print(
        f"Seed {seed:<6} "
        f"Physical: "
        f"{result['physical_success']:.4f}    "
        f"Logical: "
        f"{result['logical_success']:.4f}    "
        f"Logical Error: "
        f"{result['logical_error_rate']:.4f}"
    )

    return result


# ============================================================
# MULTI-SEED DECODER EXPERIMENT
# ============================================================

def run_decoder_experiment(
    decoder_name,
    decoder_factory
):
    """
    Run one decoder across all random seeds.
    """

    print()
    print("-----------------------------------")
    print(
        f"Decoder: {decoder_name}"
    )
    print("-----------------------------------")

    physical_results = []

    logical_results = []

    logical_error_results = []

    for seed in SEEDS:

        print(
            f"Running seed {seed}..."
        )

        decoder = decoder_factory(
            seed
        )

        result = run_single_seed(
            decoder_name,
            decoder,
            seed
        )

        physical_results.append(
            result[
                "physical_success"
            ]
        )

        logical_results.append(
            result[
                "logical_success"
            ]
        )

        logical_error_results.append(
            result[
                "logical_error_rate"
            ]
        )

    average_physical = (
        sum(physical_results)
        / len(physical_results)
    )

    average_logical = (
        sum(logical_results)
        / len(logical_results)
    )

    average_logical_error = (
        sum(logical_error_results)
        / len(logical_error_results)
    )

    print()
    print(
        f"Average Physical Success : "
        f"{average_physical:.4f}"
    )

    print(
        f"Average Logical Success  : "
        f"{average_logical:.4f}"
    )

    print(
        f"Average Logical Error    : "
        f"{average_logical_error:.4f}"
    )

    return {
        "decoder": decoder_name,

        "physical_success":
            average_physical,

        "logical_success":
            average_logical,

        "logical_error_rate":
            average_logical_error
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("===================================")
    print(
        " AI vs TRADITIONAL DECODER "
        "EXPERIMENT"
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
        f"Training samples          : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Quantum test trials/seed  : "
        f"{QUANTUM_TEST_TRIALS}"
    )

    print(
        f"Seeds                     : "
        f"{SEEDS}"
    )

    print(
        f"Total quantum trials      : "
        f"{QUANTUM_TEST_TRIALS * len(SEEDS)}"
    )

    print()

    print(
        "Feature count             : 20"
    )

    print(
        "Target size               : 3"
    )

    print()

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    # Traditional lookup
    traditional_result = (
        run_decoder_experiment(
            "Traditional Lookup",
            lambda seed:
                TraditionalDecoderAdapter()
        )
    )

    results.append(
        traditional_result
    )

    # Logistic Regression
    logistic_result = (
        run_decoder_experiment(
            "Logistic Regression",
            lambda seed:
                train_ai_decoder(
                    TimeVaryingLogisticDecoder,
                    seed
                )
        )
    )

    results.append(
        logistic_result
    )

    # Random Forest
    random_forest_result = (
        run_decoder_experiment(
            "Random Forest",
            lambda seed:
                train_ai_decoder(
                    TimeVaryingRandomForestDecoder,
                    seed
                )
        )
    )

    results.append(
        random_forest_result
    )

    # MLP
    mlp_result = (
        run_decoder_experiment(
            "MLP",
            lambda seed:
                train_ai_decoder(
                    TimeVaryingMLPDecoder,
                    seed
                )
        )
    )

    results.append(
        mlp_result
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()

    print("===================================")
    print(
        " FINAL DECODER COMPARISON"
    )
    print("===================================")

    print()

    print(
        f"{'Decoder':<24}"
        f"{'Physical':>12}"
        f"{'Logical':>12}"
        f"{'Logical Error':>16}"
    )

    print(
        "-" * 64
    )

    for result in results:

        print(
            f"{result['decoder']:<24}"
            f"{result['physical_success']:>12.4f}"
            f"{result['logical_success']:>12.4f}"
            f"{result['logical_error_rate']:>16.4f}"
        )

    # --------------------------------------------------------
    # Rank by logical success
    # --------------------------------------------------------

    ranked_results = sorted(
        results,
        key=lambda result:
            result["logical_success"],
        reverse=True
    )

    print()

    print(
        "==================================="
    )

    print(
        " RANKING BY LOGICAL SUCCESS"
    )

    print(
        "==================================="
    )

    print()

    for index, result in enumerate(
        ranked_results,
        start=1
    ):

        print(
            f"{index}. "
            f"{result['decoder']:<24}"
            f"{result['logical_success']:.4f}"
        )

    # --------------------------------------------------------
    # Best decoder
    # --------------------------------------------------------

    best = ranked_results[0]

    print()

    print(
        "Best decoder                : "
        f"{best['decoder']}"
    )

    print(
        "Best logical success        : "
        f"{best['logical_success']:.4f}"
    )

    print()

    print("===================================")
    print(
        "AI vs TRADITIONAL EXPERIMENT : "
        "SUCCESS"
    )
    print("===================================")


if __name__ == "__main__":
    main()