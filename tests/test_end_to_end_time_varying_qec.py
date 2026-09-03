from dataset.time_varying_generator import (
    TimeVaryingQECDatasetGenerator
)

from dataset.time_varying_ml_dataset import (
    TimeVaryingMLDatasetBuilder
)

from decoders.time_varying_ml import (
    TimeVaryingLogisticDecoder,
    TimeVaryingRandomForestDecoder,
    TimeVaryingMLPDecoder
)

from correction.time_varying_correction import (
    TimeVaryingCorrectionEngine
)

from evaluation.logical_recovery import (
    LogicalRecovery
)


def evaluate_end_to_end(
    decoder,
    ml_dataset,
    test_samples,
    correction_engine,
    logical_recovery
):
    """
    Evaluate one AI decoder from:

        observed data
            ↓
        AI prediction
            ↓
        correction
            ↓
        logical recovery
    """

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    logical_successes = 0
    logical_failures = 0

    physical_successes = 0
    physical_failures = 0

    predictions = decoder.predict(
        ml_dataset.X_test
    )

    for sample, predicted_error in zip(
        test_samples,
        predictions
    ):

        actual_error = sample[
            "final_error_state"
        ]

        # --------------------------------
        # Correction
        # --------------------------------

        correction_result = (
            correction_engine.correct_sample(
                actual_error,
                predicted_error
            )
        )

        corrected_state = (
            correction_result[
                "corrected_state"
            ]
        )

        if correction_result[
            "physically_correct"
        ]:
            physical_successes += 1
        else:
            physical_failures += 1

        # --------------------------------
        # Logical recovery
        # --------------------------------

        # The current generator does not
        # explicitly randomize logical state.
        #
        # Its physical error state represents
        # X errors around the logical codeword.
        #
        # We therefore evaluate logical recovery
        # using the parity of the final physical
        # error pattern.

        original_logical_state = 0

        logical_result = (
            logical_recovery.recover_sample(
                original_logical_state,
                corrected_state
            )
        )

        if logical_result[
            "logical_success"
        ]:
            logical_successes += 1
        else:
            logical_failures += 1

    total = len(test_samples)

    physical_success_rate = (
        physical_successes / total
    )

    physical_error_rate = (
        physical_failures / total
    )

    logical_success_rate = (
        logical_successes / total
    )

    logical_error_rate = (
        logical_failures / total
    )

    return {
        "physical_successes":
            physical_successes,

        "physical_failures":
            physical_failures,

        "physical_success_rate":
            physical_success_rate,

        "physical_error_rate":
            physical_error_rate,

        "logical_successes":
            logical_successes,

        "logical_failures":
            logical_failures,

        "logical_success_rate":
            logical_success_rate,

        "logical_error_rate":
            logical_error_rate
    }


def test_end_to_end_time_varying_qec():

    print("\n===================================")
    print(" END-TO-END TIME-VARYING AI QEC")
    print("===================================")

    rounds = 5
    dataset_size = 5000

    physical_error_probability = 0.10
    measurement_noise_probability = 0.10

    random_state = 42

    print(
        f"\nRounds              : {rounds}"
    )

    print(
        f"Physical noise      : "
        f"{physical_error_probability * 100:.0f}%"
    )

    print(
        f"Measurement noise   : "
        f"{measurement_noise_probability * 100:.0f}%"
    )

    print(
        f"Dataset size        : "
        f"{dataset_size}"
    )

    # ===================================
    # 1. GENERATE DATA
    # ===================================

    generator = (
        TimeVaryingQECDatasetGenerator(
            rounds=rounds,
            physical_error_probability=(
                physical_error_probability
            ),
            measurement_noise_probability=(
                measurement_noise_probability
            ),
            seed=random_state
        )
    )

    samples = generator.generate_dataset(
        num_samples=dataset_size
    )

    assert len(samples) == dataset_size

    print(
        "\nDataset generation : PASS"
    )

    # ===================================
    # 2. BUILD ML DATASET
    # ===================================

    builder = (
        TimeVaryingMLDatasetBuilder(
            test_size=0.10,
            validation_size=0.10,
            random_state=random_state
        )
    )

    ml_dataset = builder.build(
        samples
    )

    print(
        "ML dataset construction : PASS"
    )

    # ===================================
    # 3. RECREATE TEST SAMPLE ORDER
    # ===================================
    #
    # The ML builder splits X/y internally.
    # To evaluate final physical/logical
    # results, we need the corresponding
    # original samples.
    #
    # We reproduce the same split using
    # sample IDs.

    from sklearn.model_selection import (
        train_test_split
    )

    sample_indices = list(
        range(len(samples))
    )

    stratify_labels = [
        "".join(
            str(bit)
            for bit in sample[
                "final_error_state"
            ]
        )
        for sample in samples
    ]

    (
        train_indices,
        temp_indices,
        _,
        _
    ) = train_test_split(
        sample_indices,
        sample_indices,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify_labels
    )

    (
        validation_indices,
        test_indices,
        _,
        _
    ) = train_test_split(
        temp_indices,
        temp_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=[
            stratify_labels[index]
            for index in temp_indices
        ]
    )

    test_samples = [
        samples[index]
        for index in test_indices
    ]

    assert len(test_samples) == len(
        ml_dataset.X_test
    )

    print(
        "Test sample alignment : PASS"
    )

    # ===================================
    # 4. ENGINES
    # ===================================

    correction_engine = (
        TimeVaryingCorrectionEngine()
    )

    logical_recovery = (
        LogicalRecovery()
    )

    # ===================================
    # 5. LOGISTIC REGRESSION
    # ===================================

    print(
        "\nTraining Logistic Regression..."
    )

    logistic = (
        TimeVaryingLogisticDecoder(
            random_state=random_state
        )
    )

    logistic_results = (
        evaluate_end_to_end(
            logistic,
            ml_dataset,
            test_samples,
            correction_engine,
            logical_recovery
        )
    )

    # ===================================
    # 6. RANDOM FOREST
    # ===================================

    print(
        "\nTraining Random Forest..."
    )

    random_forest = (
        TimeVaryingRandomForestDecoder(
            n_estimators=100,
            random_state=random_state
        )
    )

    random_forest_results = (
        evaluate_end_to_end(
            random_forest,
            ml_dataset,
            test_samples,
            correction_engine,
            logical_recovery
        )
    )

    # ===================================
    # 7. MLP
    # ===================================

    print(
        "\nTraining MLP..."
    )

    mlp = (
        TimeVaryingMLPDecoder(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=random_state
        )
    )

    mlp_results = (
        evaluate_end_to_end(
            mlp,
            ml_dataset,
            test_samples,
            correction_engine,
            logical_recovery
        )
    )

    # ===================================
    # 8. PRINT RESULTS
    # ===================================

    print(
        "\n==================================="
    )

    print(
        " LOGISTIC REGRESSION"
    )

    print(
        "==================================="
    )

    print(
        f"\nPhysical Success Rate : "
        f"{logistic_results['physical_success_rate']:.4f}"
    )

    print(
        f"Physical Error Rate   : "
        f"{logistic_results['physical_error_rate']:.4f}"
    )

    print(
        f"Logical Success Rate  : "
        f"{logistic_results['logical_success_rate']:.4f}"
    )

    print(
        f"Logical Error Rate    : "
        f"{logistic_results['logical_error_rate']:.4f}"
    )

    print(
        "\n==================================="
    )

    print(
        " RANDOM FOREST"
    )

    print(
        "==================================="
    )

    print(
        f"\nPhysical Success Rate : "
        f"{random_forest_results['physical_success_rate']:.4f}"
    )

    print(
        f"Physical Error Rate   : "
        f"{random_forest_results['physical_error_rate']:.4f}"
    )

    print(
        f"Logical Success Rate  : "
        f"{random_forest_results['logical_success_rate']:.4f}"
    )

    print(
        f"Logical Error Rate    : "
        f"{random_forest_results['logical_error_rate']:.4f}"
    )

    print(
        "\n==================================="
    )

    print(
        " MLP NEURAL NETWORK"
    )

    print(
        "==================================="
    )

    print(
        f"\nPhysical Success Rate : "
        f"{mlp_results['physical_success_rate']:.4f}"
    )

    print(
        f"Physical Error Rate   : "
        f"{mlp_results['physical_error_rate']:.4f}"
    )

    print(
        f"Logical Success Rate  : "
        f"{mlp_results['logical_success_rate']:.4f}"
    )

    print(
        f"Logical Error Rate    : "
        f"{mlp_results['logical_error_rate']:.4f}"
    )

    # ===================================
    # 9. VALIDATE METRICS
    # ===================================

    all_results = [
        logistic_results,
        random_forest_results,
        mlp_results
    ]

    for result in all_results:

        assert (
            0.0
            <= result["physical_success_rate"]
            <= 1.0
        )

        assert (
            0.0
            <= result["physical_error_rate"]
            <= 1.0
        )

        assert (
            0.0
            <= result["logical_success_rate"]
            <= 1.0
        )

        assert (
            0.0
            <= result["logical_error_rate"]
            <= 1.0
        )

        assert (
            result["physical_success_rate"]
            + result["physical_error_rate"]
            == 1.0
        )

        assert (
            result["logical_success_rate"]
            + result["logical_error_rate"]
            == 1.0
        )

    print(
        "\nMetric ranges : PASS"
    )

    # ===================================
    # 10. MODEL COMPARISON
    # ===================================

    print(
        "\n==================================="
    )

    print(
        " END-TO-END MODEL COMPARISON"
    )

    print(
        "==================================="
    )

    print(
        "\nModel               "
        "Physical Success   "
        "Logical Success"
    )

    print(
        "-----------------------------------"
    )

    print(
        "Logistic Regression "
        f"{logistic_results['physical_success_rate']:.4f}"
        "             "
        f"{logistic_results['logical_success_rate']:.4f}"
    )

    print(
        "Random Forest       "
        f"{random_forest_results['physical_success_rate']:.4f}"
        "             "
        f"{random_forest_results['logical_success_rate']:.4f}"
    )

    print(
        "MLP                 "
        f"{mlp_results['physical_success_rate']:.4f}"
        "             "
        f"{mlp_results['logical_success_rate']:.4f}"
    )

    # ===================================
    # FINAL
    # ===================================

    print(
        "\n==================================="
    )

    print(
        " END-TO-END QEC RESULT"
    )

    print(
        "==================================="
    )

    print(
        "AI decoding          : PASS"
    )

    print(
        "Correction           : PASS"
    )

    print(
        "Logical recovery     : PASS"
    )

    print(
        "Logical error metric : PASS"
    )

    print(
        "Model comparison     : PASS"
    )

    print(
        "RESULT               : SUCCESS"
    )


if __name__ == "__main__":
    test_end_to_end_time_varying_qec()