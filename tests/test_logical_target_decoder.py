from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from decoders.logical_target import (
    LogicalTargetBuilder
)

from decoders.logical_target_gru import (
    LogicalTargetGRUDecoder
)


ROUNDS = 5

PHYSICAL_ERROR_PROBABILITY = 0.10

MEASUREMENT_NOISE_PROBABILITY = 0.10

SAMPLES = 1000

SEED = 42


def main():

    print()
    print("=" * 60)
    print(
        " LOGICAL-TARGET DECODER TEST"
    )
    print("=" * 60)

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
        f"Generated samples : {len(samples)}"
    )

    # --------------------------------------------------------
    # Target builder
    # --------------------------------------------------------

    builder = LogicalTargetBuilder()

    targets, scores = builder.build(
        samples
    )

    print(
        f"Logical targets   : {len(targets)}"
    )

    average_score = (
        sum(scores.values())
        / len(scores)
    )

    print(
        f"Average target score : "
        f"{average_score:.4f}"
    )

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    decoder = LogicalTargetGRUDecoder(
        rounds=ROUNDS,
        hidden_size=32,
        learning_rate=0.003,
        epochs=10,
        random_seed=SEED
    )

    decoder.train(
        samples,
        verbose=False
    )

    # --------------------------------------------------------
    # Single prediction
    # --------------------------------------------------------

    prediction = decoder.decode(
        samples[0]
    )

    print()
    print(
        "Sample prediction:"
    )

    print(
        f"Observed syndrome history: "
        f"{samples[0]['observed_syndrome_history']}"
    )

    print(
        f"Predicted correction: "
        f"{prediction}"
    )

    assert len(prediction) == 3

    assert all(
        bit in (0, 1)
        for bit in prediction
    )

    # --------------------------------------------------------
    # Batch prediction
    # --------------------------------------------------------

    predictions = decoder.predict_batch(
        samples[:10]
    )

    assert len(predictions) == 10

    assert all(
        len(prediction) == 3
        for prediction in predictions
    )

    print()
    print(
        "Single prediction : PASS"
    )

    print(
        "Batch prediction   : PASS"
    )

    print()
    print(
        "RESULT : SUCCESS"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()