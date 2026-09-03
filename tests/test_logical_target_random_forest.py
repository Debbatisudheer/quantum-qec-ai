from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target_random_forest import (
    LogicalTargetRandomForestDecoder
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


ROUNDS = 5

PHYSICAL_NOISE = 0.10
MEASUREMENT_NOISE = 0.10

TRAINING_SAMPLES = 5000
TEST_SAMPLES = 1000

SEED = 42


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


def logical_success(
    sample,
    correction,
    recovery
):
    encoded_state = [
        int(bit)
        for bit in sample["encoded_state"]
    ]

    actual_error = [
        int(bit)
        for bit in sample["final_error_state"]
    ]

    corrupted_state = [
        encoded_state[i] ^ actual_error[i]
        for i in range(3)
    ]

    corrected_state = [
        corrupted_state[i] ^ correction[i]
        for i in range(3)
    ]

    recovered_logical = recovery.recover(
        corrected_state
    )

    return (
        recovered_logical
        == sample["logical_state"]
    )


def main():

    print()
    print("=" * 70)
    print(
        " LOGICAL-TARGET RANDOM FOREST DECODER TEST"
    )
    print("=" * 70)

    print()
    print(
        f"Rounds              : {ROUNDS}"
    )

    print(
        f"Physical noise      : "
        f"{PHYSICAL_NOISE:.2f}"
    )

    print(
        f"Measurement noise   : "
        f"{MEASUREMENT_NOISE:.2f}"
    )

    print(
        f"Training samples    : "
        f"{TRAINING_SAMPLES}"
    )

    print(
        f"Test samples        : "
        f"{TEST_SAMPLES}"
    )

    # --------------------------------------------------------
    # GENERATE TRAINING DATA
    # --------------------------------------------------------

    training_samples = generate_samples(
        TRAINING_SAMPLES,
        SEED
    )

    # --------------------------------------------------------
    # GENERATE INDEPENDENT TEST DATA
    # --------------------------------------------------------

    test_samples = generate_samples(
        TEST_SAMPLES,
        SEED + 10000
    )

    print()
    print(
        "Training samples generated : "
        f"{len(training_samples)}"
    )

    print(
        "Test samples generated     : "
        f"{len(test_samples)}"
    )

    # --------------------------------------------------------
    # CREATE DECODER
    # --------------------------------------------------------

    decoder = (
        LogicalTargetRandomForestDecoder(
            rounds=ROUNDS,
            n_estimators=100,
            random_seed=SEED
        )
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    decoder.train(
        training_samples
    )

    print()
    print(
        "Random Forest training : PASS"
    )

    # --------------------------------------------------------
    # SINGLE SAMPLE TEST
    # --------------------------------------------------------

    sample = test_samples[0]

    prediction = decoder.decode(
        sample
    )

    print()
    print(
        "Sample prediction     : "
        f"{prediction}"
    )

    assert len(prediction) == 3

    assert all(
        bit in (0, 1)
        for bit in prediction
    )

    print(
        "Single decode         : PASS"
    )

    # --------------------------------------------------------
    # BATCH TEST
    # --------------------------------------------------------

    predictions = decoder.decode_batch(
        test_samples
    )

    assert len(predictions) == (
        TEST_SAMPLES
    )

    print(
        "Batch decode          : PASS"
    )

    # --------------------------------------------------------
    # LOGICAL EVALUATION
    # --------------------------------------------------------

    recovery = LogicalRecovery()

    successes = 0

    for sample, prediction in zip(
        test_samples,
        predictions
    ):

        if logical_success(
            sample,
            prediction,
            recovery
        ):
            successes += 1

    logical_accuracy = (
        successes / TEST_SAMPLES
    )

    print()
    print(
        "Logical success       : "
        f"{logical_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # TARGET INFORMATION
    # --------------------------------------------------------

    print()
    print(
        "Logical targets learned : "
        f"{len(decoder.targets)}"
    )

    if decoder.target_scores:

        average_score = (
            sum(
                decoder.target_scores.values()
            )
            / len(decoder.target_scores)
        )

        print(
            "Average target score    : "
            f"{average_score:.4f}"
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        " RANDOM FOREST DECODER TEST : SUCCESS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()