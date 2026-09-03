from fastapi import APIRouter, HTTPException, Query


from backend.api.schemas import (
    ExperimentAnalysisResponse,
    ExperimentResultResponse,
    ExperimentSummary,
    ExperimentSummaryStatistics,
    SimulationRequest,
    SimulationResponse,
    SimulationTraceRequest,
    SimulationTraceResponse,
)


from backend.services.trace_service import (
    SimulationTraceService,
)


from experiments.analysis import ExperimentAnalysis
from experiments.engine import ExperimentEngine
from experiments.result_storage import ExperimentResultStorage
from experiments.result_query import ExperimentResultQuery
from experiments.visualization import ExperimentVisualization


from evaluation.scientific_evaluation import (
    ScientificEvaluation,
)


router = APIRouter()


# ============================================================
# SIMULATION
# ============================================================

@router.post(
    "/simulate",
    response_model=SimulationResponse,
)
def simulate(
    request: SimulationRequest,
):
    """
    Run a complete QEC + AI experiment.
    """

    # --------------------------------------------------------
    # Validate currently supported QEC configuration
    # --------------------------------------------------------

    if request.qec_code != "bit_flip_3":
        raise HTTPException(
            status_code=400,
            detail=(
                "Currently supported qec_code: "
                "bit_flip_3"
            ),
        )

    if request.num_qubits != 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "Currently supported num_qubits: 3"
            ),
        )

    # --------------------------------------------------------
    # Create experiment configuration
    # --------------------------------------------------------

    from experiments.config import ExperimentConfig

    config = ExperimentConfig(
        qec_code=request.qec_code,
        num_qubits=request.num_qubits,
        logical_state=request.logical_state,
        rounds=request.rounds,
        physical_noise_probability=(
            request.physical_noise_probability
        ),
        measurement_noise_probability=(
            request.measurement_noise_probability
        ),
        training_samples=request.training_samples,
        test_samples=request.test_samples,
        decoder_type=(
            "logical_target_random_forest"
        ),
        random_forest_estimators=(
            request.random_forest_estimators
        ),
        seed=request.seed,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    storage = ExperimentResultStorage()

    # --------------------------------------------------------
    # Experiment engine
    # --------------------------------------------------------

    engine = ExperimentEngine(
        config=config,
        storage=storage,
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    result = engine.run()

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return SimulationResponse(
        experiment_id=result.experiment_id,

        qec_code=config.qec_code,
        num_qubits=config.num_qubits,
        rounds=config.rounds,

        logical_state=(
            config.logical_state
            if config.logical_state is not None
            else 0
        ),

        physical_noise_probability=(
            config.physical_noise_probability
        ),

        measurement_noise_probability=(
            config.measurement_noise_probability
        ),

        training_samples=result.training_samples,
        test_samples=result.test_samples,

        decoder_type=result.decoder_type,

        logical_targets_learned=(
            result.logical_targets_learned
        ),

        average_target_score=(
            result.average_target_score
        ),

        exact_accuracy=result.exact_accuracy,
        physical_accuracy=result.physical_accuracy,
        bit_accuracy=result.bit_accuracy,
        logical_accuracy=result.logical_accuracy,

        training_seconds=result.training_seconds,
        inference_seconds=result.inference_seconds,
        samples_per_second=result.samples_per_second,

        status="completed",
    )


# ============================================================
# SINGLE SIMULATION TRACE
# ============================================================

@router.post(
    "/simulate/trace",
    response_model=SimulationTraceResponse,
)
def simulate_trace(
    request: SimulationTraceRequest,
):
    """
    Generate one complete QEC simulation trace.

    The trace shows:

        logical state
              ↓
        encoded state
              ↓
        physical noise
              ↓
        corrupted state
              ↓
        syndrome history
              ↓
        AI decoder
              ↓
        predicted correction
              ↓
        corrected state
              ↓
        logical recovery
    """

    # --------------------------------------------------------
    # Validate currently supported QEC configuration
    # --------------------------------------------------------

    if request.qec_code != "bit_flip_3":
        raise HTTPException(
            status_code=400,
            detail=(
                "Currently supported qec_code: "
                "bit_flip_3"
            ),
        )

    if request.num_qubits != 3:
        raise HTTPException(
            status_code=400,
            detail=(
                "Currently supported num_qubits: 3"
            ),
        )

    # --------------------------------------------------------
    # Create trace service
    # --------------------------------------------------------

    try:

        service = SimulationTraceService(
            rounds=request.rounds,

            physical_noise_probability=(
                request.physical_noise_probability
            ),

            measurement_noise_probability=(
                request.measurement_noise_probability
            ),

            training_samples=(
                request.training_samples
            ),

            random_forest_estimators=(
                request.random_forest_estimators
            ),

            seed=(
                request.seed
                if request.seed is not None
                else 42
            ),
        )

        # ----------------------------------------------------
        # Generate trace
        # ----------------------------------------------------

        trace = service.generate_trace()

        # ----------------------------------------------------
        # Return trace
        # ----------------------------------------------------

        return trace

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Trace generation failed: {exc}"
            ),
        ) from exc


# ============================================================
# GET ALL EXPERIMENTS
# ============================================================

@router.get(
    "/experiments",
    response_model=list[ExperimentSummary],
)
def get_experiments(
    rounds: int | None = Query(
        default=None,
        gt=0,
    ),

    physical_noise_probability: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),

    measurement_noise_probability: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),

    decoder_type: str | None = Query(
        default=None,
    ),

    sort_by: str | None = Query(
        default=None,
    ),

    descending: bool = Query(
        default=True,
    ),
):
    """
    Return stored experiments with optional
    filtering and sorting.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    try:

        results = query.filter(
            rounds=rounds,

            physical_noise_probability=(
                physical_noise_probability
            ),

            measurement_noise_probability=(
                measurement_noise_probability
            ),

            decoder_type=decoder_type,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if sort_by is not None:

        try:

            results = query.sort_by(
                results,
                metric=sort_by,
                descending=descending,
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    summaries = []

    for result in results:

        config = result.get(
            "config",
            {},
        )

        summaries.append(
            ExperimentSummary(

                experiment_id=result[
                    "experiment_id"
                ],

                qec_code=config.get(
                    "qec_code",
                    "bit_flip_3",
                ),

                num_qubits=config.get(
                    "num_qubits",
                    3,
                ),

                rounds=config.get(
                    "rounds",
                    5,
                ),

                physical_noise_probability=(
                    config.get(
                        "physical_noise_probability"
                    )
                ),

                measurement_noise_probability=(
                    config.get(
                        "measurement_noise_probability"
                    )
                ),

                training_samples=result[
                    "training_samples"
                ],

                test_samples=result[
                    "test_samples"
                ],

                decoder_type=result[
                    "decoder_type"
                ],

                exact_accuracy=result[
                    "exact_accuracy"
                ],

                physical_accuracy=result[
                    "physical_accuracy"
                ],

                bit_accuracy=result[
                    "bit_accuracy"
                ],

                logical_accuracy=result[
                    "logical_accuracy"
                ],

                training_seconds=result[
                    "training_seconds"
                ],

                inference_seconds=result[
                    "inference_seconds"
                ],

                samples_per_second=result[
                    "samples_per_second"
                ],
            )
        )

    return summaries


# ============================================================
# BEST EXPERIMENT
# ============================================================

@router.get(
    "/experiments/best",
)
def get_best_experiment(
    metric: str = Query(
        default="logical_accuracy",
    ),
):
    """
    Return the best experiment according
    to the selected metric.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    try:

        return query.best(
            metric=metric
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ============================================================
# WORST EXPERIMENT
# ============================================================

@router.get(
    "/experiments/worst",
)
def get_worst_experiment(
    metric: str = Query(
        default="logical_accuracy",
    ),
):
    """
    Return the worst experiment according
    to the selected metric.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    try:

        return query.worst(
            metric=metric
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


# ============================================================
# COMPARE EXPERIMENTS
# ============================================================

@router.get(
    "/experiments/compare",
)
def compare_experiments(
    experiment_ids: str = Query(
        ...,
    ),
):
    """
    Compare multiple experiments.

    Example:

        /experiments/compare?
        experiment_ids=id1,id2,id3
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    ids = [
        experiment_id.strip()
        for experiment_id
        in experiment_ids.split(",")
        if experiment_id.strip()
    ]

    if not ids:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one experiment_id "
                "is required"
            ),
        )

    try:

        return query.compare(
            ids
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


# ============================================================
# GET ONE RESULT
# ============================================================

@router.get(
    "/results/{experiment_id}",
    response_model=ExperimentResultResponse,
)
def get_result(
    experiment_id: str,
):
    """
    Return one stored experiment result.
    """

    storage = ExperimentResultStorage()

    try:

        result = storage.load(
            experiment_id
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return result


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

@router.get(
    "/experiments/summary",
    response_model=ExperimentSummaryStatistics,
)
def get_experiment_summary():
    """
    Return aggregate statistics for all experiments.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    results = query.all_results()

    summary = query.summary(
        results
    )

    return summary


# ============================================================
# EXPERIMENT ANALYSIS
# ============================================================

@router.get(
    "/experiments/analysis",
    response_model=ExperimentAnalysisResponse,
)
def get_experiment_analysis():
    """
    Return detailed experiment analysis.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    analyzer = ExperimentAnalysis(
        query
    )

    results = query.all_results()

    analysis = {

        "rounds": analyzer.analyze_rounds(
            results
        ),

        "physical_noise": (
            analyzer.analyze_physical_noise(
                results
            )
        ),

        "measurement_noise": (
            analyzer.analyze_measurement_noise(
                results
            )
        ),

        "decoders": analyzer.analyze_decoders(
            results
        ),
    }

    return {
        "analysis": analysis
    }


# ============================================================
# EXPERIMENT VISUALIZATION
# ============================================================

@router.get(
    "/experiments/visualization",
)
def get_experiment_visualization():
    """
    Return frontend-ready visualization data
    for all stored experiments.
    """

    storage = ExperimentResultStorage()

    query = ExperimentResultQuery(
        storage
    )

    analyzer = ExperimentAnalysis(
        query
    )

    visualization = ExperimentVisualization(
        analyzer
    )

    return visualization.frontend_data()


# ============================================================
# SCIENTIFIC EVALUATION
# ============================================================

def _create_scientific_evaluator():

    source = (
        "experiments"
        "/paired_results"
        "/paired_baseline_ai_results.json"
    )

    return ScientificEvaluation(
        source
    )


@router.get(
    "/scientific/evaluation",
)
def get_scientific_evaluation():
    """
    Return the complete formal scientific evaluation.

    Includes:

        source
        metadata
        per-condition scientific results
        summary
    """

    try:

        evaluator = (
            _create_scientific_evaluator()
        )

        results = (
            evaluator.evaluate_all()
        )

        return {
            "source": (
                str(
                    evaluator.result_path
                )
            ),

            "metadata": (
                evaluator.metadata
            ),

            "results": [
                result.to_dict()
                for result in results
            ],

            "summary": (
                evaluator.summary(
                    results
                )
            ),
        }

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.get(
    "/scientific/results",
)
def get_scientific_results():
    """
    Return only the scientific result rows.
    """

    try:

        evaluator = (
            _create_scientific_evaluator()
        )

        results = (
            evaluator.evaluate_all()
        )

        return [
            result.to_dict()
            for result in results
        ]

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@router.get(
    "/scientific/summary",
)
def get_scientific_summary():
    """
    Return high-level scientific evaluation summary.
    """

    try:

        evaluator = (
            _create_scientific_evaluator()
        )

        results = (
            evaluator.evaluate_all()
        )

        return evaluator.summary(
            results
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error