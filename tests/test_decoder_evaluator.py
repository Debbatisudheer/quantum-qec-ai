from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder
)

from evaluation.decoder_evaluator import (
    DecoderEvaluator
)


ROUNDS = 5

PHYSICAL_NOISE = 0.10
MEASUREMENT_NOISE = 0.10

TRAINING_SAMPLES = 5000
TEST_SAMPLES = 1000

TRAIN_SEED = 42
TEST_SEED = 10042


def generate_samples(
    count,
    seed
):
    generator = TimeVaryingQECDatasetGenerator(
        rounds=ROUNDS,
        physical_error_probability=PHYSICAL_NOISE,
        measurement_noise_probability=MEASUREMENT_NOISE,
        seed=seed
    )

    return [
        generator.generate_sample(i)
        for i in range(count)
    ]


def main():

    print()
    print("=" * 70)
    print(
        " REUSABLE DECODER EVALUATOR TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # GENERATE DATA
    # --------------------------------------------------------

    print()
    print(
        "Generating training data..."
    )

    training_samples = generate_samples(
        TRAINING_SAMPLES,
        TRAIN_SEED
    )

    print(
        "Generating test data..."
    )

    test_samples = generate_samples(
        TEST_SAMPLES,
        TEST_SEED
    )

    print()
    print(
        f"Training samples : "
        f"{len(training_samples)}"
    )

    print(
        f"Test samples     : "
        f"{len(test_samples)}"
    )

    # --------------------------------------------------------
    # TRAIN DECODER
    # --------------------------------------------------------

    decoder = (
        LogicalTargetRandomForestDecoder(
            rounds=ROUNDS,
            n_estimators=100,
            random_seed=TRAIN_SEED
        )
    )

    decoder.train(
        training_samples
    )

    print()
    print(
        "Random Forest training : PASS"
    )

    # --------------------------------------------------------
    # CREATE EVALUATOR
    # --------------------------------------------------------

    evaluator = (
        DecoderEvaluator()
    )

    print(
        "DecoderEvaluator creation : PASS"
    )

    # --------------------------------------------------------
    # FULL EVALUATION
    # --------------------------------------------------------

    results = evaluator.evaluate(
        decoder,
        test_samples
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " EVALUATION RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Exact error accuracy : "
        f"{results['exact']:.4f}"
    )

    print(
        f"Physical recovery    : "
        f"{results['physical']:.4f}"
    )

    print(
        f"Bit accuracy         : "
        f"{results['bit']:.4f}"
    )

    print(
        f"Logical success      : "
        f"{results['logical']:.4f}"
    )

    print(
        f"Inference time       : "
        f"{results['inference_seconds']:.4f} sec"
    )

    print(
        f"Throughput           : "
        f"{results['samples_per_second']:.2f} samples/sec"
    )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    assert 0.0 <= results["exact"] <= 1.0

    assert 0.0 <= results["physical"] <= 1.0

    assert 0.0 <= results["bit"] <= 1.0

    assert 0.0 <= results["logical"] <= 1.0

    assert results["sample_count"] == (
        TEST_SAMPLES
    )

    assert results[
        "inference_seconds"
    ] >= 0.0

    assert results[
        "samples_per_second"
    ] > 0.0

    print()
    print(
        "Metric validation       : PASS"
    )

    print(
        "Logical evaluation      : PASS"
    )

    print(
        "Inference measurement   : PASS"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " DECODER EVALUATOR TEST : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()