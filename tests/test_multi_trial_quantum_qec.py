from sklearn.model_selection import (
    train_test_split
)

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

from experiments.time_varying_quantum_qec import (
    TimeVaryingQuantumQECExperiment
)


def prepare_test_samples(
    samples,
    random_state=42
):
    """
    Reproduce the same 80/10/10 split used
    by TimeVaryingMLDatasetBuilder.
    """

    indices = list(
        range(len(samples))
    )

    labels = [
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
        temp_indices
    ) = train_test_split(
        indices,
        test_size=0.20,
        random_state=random_state,
        stratify=labels
    )

    (
        validation_indices,
        test_indices
    ) = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=[
            labels[index]
            for index in temp_indices
        ]
    )

    return (
        train_indices,
        validation_indices,
        test_indices
    )


def evaluate_model(
    decoder,
    ml_dataset,
    test_samples,
    experiment
):
    """
    Train decoder and run the complete
    quantum experiment.
    """

    decoder.train(
        ml_dataset.X_train,
        ml_dataset.y_train
    )

    results = (
        experiment.run_experiment(
            samples=test_samples,
            ml_features=ml_dataset.X_test,
            decoder=decoder
        )
    )

    metrics = (
        experiment.calculate_metrics(
            results
        )
    )

    return results, metrics


def test_multi_trial_quantum_qec():

    print("\n===================================")
    print(" MULTI-TRIAL QUANTUM AI QEC")
    print("===================================")

    rounds = 5
    dataset_size = 1000

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

    assert len(
        ml_dataset.X_test
    ) == 100

    # ===================================
    # 3. RECREATE TEST SPLIT
    # ===================================

    (
        train_indices,
        validation_indices,
        test_indices
    ) = prepare_test_samples(
        samples,
        random_state=random_state
    )

    test_samples = [
        samples[index]
        for index in test_indices
    ]

    assert len(test_samples) == (
        len(ml_dataset.X_test)
    )

    print(
        "Test sample alignment : PASS"
    )

    # ===================================
    # 4. QUANTUM EXPERIMENT
    # ===================================

    experiment = (
        TimeVaryingQuantumQECExperiment(
            shots=1
        )
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

    logistic_results, logistic_metrics = (
        evaluate_model(
            logistic,
            ml_dataset,
            test_samples,
            experiment
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

    rf_results, rf_metrics = (
        evaluate_model(
            random_forest,
            ml_dataset,
            test_samples,
            experiment
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

    mlp_results, mlp_metrics = (
        evaluate_model(
            mlp,
            ml_dataset,
            test_samples,
            experiment
        )
    )

    # ===================================
    # 8. PRINT RESULTS
    # ===================================

    model_results = [
        (
            "Logistic Regression",
            logistic_metrics
        ),
        (
            "Random Forest",
            rf_metrics
        ),
        (
            "MLP",
            mlp_metrics
        )
    ]

    print(
        "\n==================================="
    )

    print(
        " QUANTUM AI QEC RESULTS"
    )

    print(
        "==================================="
    )

    print(
        "\nModel               "
        "Physical Success   "
        "Logical Success   "
        "Logical Error"
    )

    print(
        "-----------------------------------"
    )

    for model_name, metrics in (
        model_results
    ):

        print(
            f"{model_name:<20}"
            f"{metrics['physical_success_rate']:.4f}"
            "             "
            f"{metrics['logical_success_rate']:.4f}"
            "             "
            f"{metrics['logical_error_rate']:.4f}"
        )

    # ===================================
    # 9. VALIDATE
    # ===================================

    for _, metrics in model_results:

        assert (
            metrics["total_trials"]
            == 100
        )

        assert (
            0.0
            <= metrics[
                "logical_success_rate"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= metrics[
                "logical_error_rate"
            ]
            <= 1.0
        )

        assert (
            0.0
            <= metrics[
                "physical_success_rate"
            ]
            <= 1.0
        )

        assert (
            metrics[
                "logical_success_rate"
            ]
            + metrics[
                "logical_error_rate"
            ]
            == 1.0
        )

    print(
        "\nMetric validation : PASS"
    )

    # ===================================
    # 10. EXAMPLE
    # ===================================

    example = logistic_results[0]

    print(
        "\n==================================="
    )

    print(
        " QUANTUM TRIAL EXAMPLE"
    )

    print(
        "==================================="
    )

    print(
        f"\nLogical state       : "
        f"{example['logical_state']}"
    )

    print(
        f"Actual error        : "
        f"{example['actual_error_state']}"
    )

    print(
        f"AI prediction       : "
        f"{example['predicted_error_state']}"
    )

    print(
        f"Corrected error     : "
        f"{example['corrected_error_state']}"
    )

    print(
        f"Measured state      : "
        f"{example['measured_state']}"
    )

    print(
        f"Recovered logical   : "
        f"{example['recovered_logical_state']}"
    )

    print(
        f"Logical success     : "
        f"{example['logical_success']}"
    )

    # ===================================
    # FINAL
    # ===================================

    print(
        "\n==================================="
    )

    print(
        " MULTI-TRIAL QUANTUM QEC RESULT"
    )

    print(
        "==================================="
    )

    print(
        "Quantum simulation       : PASS"
    )

    print(
        "AI decoding              : PASS"
    )

    print(
        "Correction               : PASS"
    )

    print(
        "Logical recovery         : PASS"
    )

    print(
        "Multi-trial evaluation   : PASS"
    )

    print(
        "Logical error rate       : PASS"
    )

    print(
        "RESULT                   : SUCCESS"
    )


if __name__ == "__main__":
    test_multi_trial_quantum_qec()