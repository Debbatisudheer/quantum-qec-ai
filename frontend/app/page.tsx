"use client";

import { useEffect, useMemo, useState } from "react";
import SiteHeader from "../component/SiteHeader";
import SiteFooter from "../component/SiteFooter";

/* =========================================================
   TYPES
========================================================= */

type ExperimentConfig = {
  qec_code: string;
  num_qubits: number;
  logical_state: number;
  rounds: number;
  physical_noise_probability: number;
  measurement_noise_probability: number;
  training_samples: number;
  test_samples: number;
  random_forest_estimators: number;
  seed: number;
};

type ExperimentResult = {
  experiment_id: string;
  qec_code: string;
  num_qubits: number;
  rounds: number;
  logical_state: number;
  physical_noise_probability: number;
  measurement_noise_probability: number;
  training_samples: number;
  test_samples: number;
  decoder_type: string;
  logical_targets_learned: number;
  average_target_score: number;
  exact_accuracy: number;
  physical_accuracy: number;
  bit_accuracy: number;
  logical_accuracy: number;
  training_seconds: number;
  inference_seconds: number;
  samples_per_second: number;
  status: string;
};

type ExperimentSummary = {
  experiment_id: string;
  qec_code: string;
  num_qubits: number;
  rounds: number;
  physical_noise_probability: number;
  measurement_noise_probability: number;
  training_samples: number;
  test_samples: number;
  decoder_type: string;
  exact_accuracy: number;
  physical_accuracy: number;
  bit_accuracy: number;
  logical_accuracy: number;
  training_seconds: number;
  inference_seconds: number;
  samples_per_second: number;
};

type VisualizationData = {
  charts: {
    rounds: {
      rounds: number;
      logical_success: number;
    }[];

    physical_noise: {
      physical_noise: number | null;
      logical_success: number;
    }[];

    measurement_noise: {
      measurement_noise: number | null;
      logical_success: number;
    }[];

    decoders: {
      decoder: string;
      logical_success: number;
    }[];
  };

  performance: {
    experiment_id: string;
    decoder: string;
    rounds: number;
    physical_noise: number | null;
    measurement_noise: number | null;
    logical_success: number;
    physical_recovery: number;
    bit_accuracy: number;
    training_seconds: number;
    inference_seconds: number;
    samples_per_second: number;
  }[];
};

type ScientificResult = {
  physical_noise: number;
  measurement_noise: number;
  baseline_logical_success: number;
  ai_logical_success: number;
  absolute_gain: number;
  relative_gain: number;
  baseline_logical_error: number;
  ai_logical_error: number;
  logical_error_reduction: number;
  bootstrap_ci_low: number;
  bootstrap_ci_high: number;
  permutation_p_value: number;
  seed_count: number;
  test_samples_per_seed: number;
};

type ScientificEvaluation = {
  source?: string;
  metadata?: Record<string, unknown>;
  results: ScientificResult[];
  summary: {
    conditions: number;
    positive_gain_conditions: number;
    maximum_absolute_gain: {
      physical_noise: number;
      measurement_noise: number;
      absolute_gain?: number;
      gain?: number;
    };
    maximum_error_reduction: {
      physical_noise: number;
      measurement_noise: number;
      logical_error_reduction?: number;
      reduction?: number;
    };
  };
};

/* =========================================================
   SINGLE TRACE TYPES
========================================================= */

type TraceRound = {
  round: number;
  physical_error_state: string;
  perfect_syndrome: string;
  observed_syndrome: string;
  detection_event: string;
};

type SimulationTrace = {
  sample_id: number;
  qec_code: string;
  num_qubits: number;
  rounds: number;
  logical_state: number;
  encoded_state: string;
  noise: {
    physical_error_probability: number;
    measurement_noise_probability: number;
    physical_error_history: string[];
    final_error_state: string;
    final_error_description: string;
  };
  quantum_state: {
    encoded: string;
    corrupted: string;
  };
  syndrome: {
    perfect_history: string[];
    observed_history: string[];
    detection_events: string[];
    final_perfect: string;
    final_observed: string;
    rounds: TraceRound[];
  };
  decoder: {
    type: string;
    training_samples: number;
    random_forest_estimators: number;
    predicted_correction: string;
    predicted_correction_bits: number[];
    confidence: number | null;
  };
  correction: {
    actual_error: string;
    predicted_correction: string;
    corrected_state: string;
  };
  recovery: {
    original_logical_state: number;
    recovered_logical_state: number;
    logical_success: boolean;
    logical_failure: boolean;
    physical_recovery: boolean;
    exact_error_match: boolean;
  };
};

/* =========================================================
   DEFAULT CONFIG
========================================================= */

const defaultConfig: ExperimentConfig = {
  qec_code: "bit_flip_3",
  num_qubits: 3,
  logical_state: 0,
  rounds: 5,
  physical_noise_probability: 0.1,
  measurement_noise_probability: 0.1,
  training_samples: 5000,
  test_samples: 1000,
  random_forest_estimators: 100,
  seed: 42,
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ||
  "http://127.0.0.1:8001";

/* =========================================================
   MAIN PAGE
========================================================= */

function ScientificMetric({
  label,
  value,
  detail,
  highlight = false,
}: {
  label: string;
  value: string;
  detail: string;
  highlight?: boolean;
}) {
  return (
    <div className={`rounded-xl border p-5 ${highlight ? "border-cyan-400/20 bg-cyan-400/5" : "border-white/10 bg-[#07111f]"}`}>
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={`mt-2 text-2xl font-bold ${highlight ? "text-cyan-300" : "text-white"}`}>{value}</p>
      <p className="mt-1 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

function ScientificBar({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: number;
  highlight?: boolean;
}) {
  return (
    <div className="grid grid-cols-[58px_1fr_58px] items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <div className="h-2 overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${highlight ? "bg-cyan-400" : "bg-slate-500"}`}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
      <span className={`text-right text-xs font-semibold ${highlight ? "text-cyan-300" : "text-slate-400"}`}>
        {(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

function ScientificLineChart({
  data,
  title,
  baselineKey,
  aiKey,
}: {
  data: ScientificResult[];
  title: string;
  baselineKey: "baseline_logical_success" | "baseline_logical_error";
  aiKey: "ai_logical_success" | "ai_logical_error";
}) {
  const width = 760;
  const height = 330;
  const left = 58;
  const right = 24;
  const top = 30;
  const bottom = 58;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  if (!data.length) return null;

  const x = (index: number) =>
    left + (data.length === 1 ? plotWidth / 2 : (index / (data.length - 1)) * plotWidth);
  const y = (value: number) => top + (1 - Math.max(0, Math.min(1, value))) * plotHeight;

  const baselinePoints = data.map((row, i) => `${x(i)},${y(row[baselineKey])}`).join(" ");
  const aiPoints = data.map((row, i) => `${x(i)},${y(row[aiKey])}`).join(" ");

  return (
    <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
      <div className="mb-4 flex flex-col justify-between gap-2 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">{title}</p>
          <p className="mt-1 text-xs text-slate-500">Paired held-out evaluation across physical-noise levels.</p>
        </div>
        <div className="flex gap-4 text-xs">
          <span className="text-slate-400">● Baseline</span>
          <span className="text-cyan-300">● AI-QEC</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-auto min-w-[680px] w-full" role="img" aria-label={title}>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const yy = y(tick);
            return (
              <g key={tick}>
                <line x1={left} y1={yy} x2={width - right} y2={yy} stroke="currentColor" className="text-white/10" />
                <text x={left - 9} y={yy + 4} textAnchor="end" className="fill-slate-500 text-[11px]">{Math.round(tick * 100)}%</text>
              </g>
            );
          })}
          <line x1={left} y1={top + plotHeight} x2={width - right} y2={top + plotHeight} stroke="currentColor" className="text-white/20" />
          <polyline points={baselinePoints} fill="none" stroke="currentColor" className="text-slate-500" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          <polyline points={aiPoints} fill="none" stroke="currentColor" className="text-cyan-400" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          {data.map((row, i) => (
            <g key={`${row.physical_noise}-${row.measurement_noise}`}>
              <circle cx={x(i)} cy={y(row[baselineKey])} r="4" fill="currentColor" className="text-slate-500" />
              <circle cx={x(i)} cy={y(row[aiKey])} r="4" fill="currentColor" className="text-cyan-400" />
              <text x={x(i)} y={top + plotHeight + 22} textAnchor="middle" className="fill-slate-500 text-[10px]">{(row.physical_noise * 100).toFixed(0)}%</text>
            </g>
          ))}
          <text x={width / 2} y={height - 10} textAnchor="middle" className="fill-slate-500 text-[11px]">Physical Noise</text>
        </svg>
      </div>
    </div>
  );
}

function ScientificGainChart({ data }: { data: ScientificResult[] }) {
  const width = 760;
  const height = 300;
  const left = 58;
  const right = 24;
  const top = 30;
  const bottom = 58;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const maxValue = Math.max(...data.map((row) => row.absolute_gain), 0.01);
  const xStep = data.length ? plotWidth / data.length : plotWidth;

  return (
    <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
      <div className="mb-4">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">AI-QEC Absolute Gain</p>
        <p className="mt-1 text-xs text-slate-500">Absolute improvement in logical success over the baseline.</p>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="h-auto min-w-[680px] w-full" role="img" aria-label="AI-QEC absolute gain by physical noise">
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const yy = top + (1 - tick) * plotHeight;
            return (
              <g key={tick}>
                <line x1={left} y1={yy} x2={width - right} y2={yy} stroke="currentColor" className="text-white/10" />
                <text x={left - 9} y={yy + 4} textAnchor="end" className="fill-slate-500 text-[11px]">{(maxValue * tick * 100).toFixed(1)} pp</text>
              </g>
            );
          })}
          {data.map((row, i) => {
            const barWidth = Math.max(28, xStep * 0.58);
            const barHeight = (row.absolute_gain / maxValue) * plotHeight;
            const bx = left + i * xStep + (xStep - barWidth) / 2;
            const by = top + plotHeight - barHeight;
            return (
              <g key={`${row.physical_noise}-${row.measurement_noise}`}>
                <rect x={bx} y={by} width={barWidth} height={barHeight} rx="6" fill="currentColor" className="text-cyan-400" opacity="0.85" />
                <text x={bx + barWidth / 2} y={by - 9} textAnchor="middle" className="fill-cyan-300 text-[11px] font-semibold">+{(row.absolute_gain * 100).toFixed(2)} pp</text>
                <text x={bx + barWidth / 2} y={top + plotHeight + 22} textAnchor="middle" className="fill-slate-500 text-[10px]">{(row.physical_noise * 100).toFixed(0)}%</text>
              </g>
            );
          })}
          <text x={width / 2} y={height - 10} textAnchor="middle" className="fill-slate-500 text-[11px]">Physical Noise</text>
        </svg>
      </div>
    </div>
  );
}

export default function Home() {
  const [config, setConfig] =
    useState<ExperimentConfig>(defaultConfig);

  const [result, setResult] =
    useState<ExperimentResult | null>(null);

  const [experiments, setExperiments] =
    useState<ExperimentSummary[]>([]);

  const [visualization, setVisualization] =
    useState<VisualizationData | null>(null);

  const [scientificEvaluation, setScientificEvaluation] =
    useState<ScientificEvaluation | null>(null);

  const [scientificLoading, setScientificLoading] =
    useState(false);

  const [scientificError, setScientificError] =
    useState<string | null>(null);

  const [status, setStatus] =
    useState("Ready");

  const [loading, setLoading] =
    useState(false);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  const [visualizationLoading, setVisualizationLoading] =
    useState(false);

  const [selectedExperimentId, setSelectedExperimentId] =
    useState<string | null>(null);

  const [selectedExperiments, setSelectedExperiments] =
    useState<string[]>([]);

  const [error, setError] =
    useState<string | null>(null);


  /* =======================================================
     SINGLE TRACE STATE
  ======================================================= */

  const [trace, setTrace] =
    useState<SimulationTrace | null>(null);

  const [traceLoading, setTraceLoading] =
    useState(false);

  const [traceError, setTraceError] =
    useState<string | null>(null);

  /* =======================================================
     ANALYSIS STATE
  ======================================================= */

  const [roundFilter, setRoundFilter] =
    useState("all");

  const [physicalNoiseFilter, setPhysicalNoiseFilter] =
    useState("all");

  const [measurementNoiseFilter, setMeasurementNoiseFilter] =
    useState("all");

  const [decoderFilter, setDecoderFilter] =
    useState("all");

  const [sortMetric, setSortMetric] =
    useState("logical_accuracy");

  const [sortDescending, setSortDescending] =
    useState(true);

  /* =======================================================
     CONFIGURATION
  ======================================================= */

  const updateConfig = (
    field: keyof ExperimentConfig,
    value: string | number
  ) => {
    setConfig((current) => ({
      ...current,
      [field]: value,
    }));
  };

  const resetExperiment = () => {
    setConfig({ ...defaultConfig });
    setResult(null);
    setTrace(null);
    setTraceError(null);
    setSelectedExperimentId(null);
    setStatus("Ready");
    setError(null);
  };

  /* =======================================================
     LOAD EXPERIMENT HISTORY
  ======================================================= */

  const loadExperiments = async () => {
    setHistoryLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/experiments`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to load experiments (${response.status})`
        );
      }

      const data: ExperimentSummary[] =
        await response.json();

      setExperiments(data);
    } catch (err) {
      console.error(
        "Experiment history error:",
        err
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  /* =======================================================
     LOAD VISUALIZATION DATA
  ======================================================= */

  const loadVisualization = async () => {
    setVisualizationLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/experiments/visualization`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to load visualization data (${response.status})`
        );
      }

      const data: VisualizationData =
        await response.json();

      setVisualization(data);
    } catch (err) {
      console.error(
        "Visualization error:",
        err
      );
    } finally {
      setVisualizationLoading(false);
    }
  };

  /* =======================================================
     LOAD SCIENTIFIC EVALUATION
  ======================================================= */

  const loadScientificEvaluation = async () => {
    setScientificLoading(true);
    setScientificError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/scientific/evaluation`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to load scientific evaluation (${response.status})`
        );
      }

      const data: ScientificEvaluation =
        await response.json();

      setScientificEvaluation(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to load scientific evaluation.";

      setScientificError(message);
    } finally {
      setScientificLoading(false);
    }
  };

  /* =======================================================
     LOAD ONE EXPERIMENT
  ======================================================= */

  const loadExperiment = async (
    experimentId: string
  ) => {
    setSelectedExperimentId(experimentId);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/results/${experimentId}`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to load experiment (${response.status})`
        );
      }

      const data: ExperimentResult =
        await response.json();

      setResult(data);
      setStatus("Stored experiment loaded");
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to load experiment.";

      setError(message);
      setStatus("Failed to load experiment");
    }
  };

  /* =======================================================
     RUN EXPERIMENT
  ======================================================= */

  const runExperiment = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedExperimentId(null);
    setStatus("Running experiment...");

    try {
      const response = await fetch(
        `${API_BASE_URL}/simulate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(config),
        }
      );

      if (!response.ok) {
        let message =
          `Backend request failed (${response.status})`;

        try {
          const errorData =
            await response.json();

          if (errorData?.detail) {
            message = String(
              errorData.detail
            );
          }
        } catch {
          // Keep default message.
        }

        throw new Error(message);
      }

      const data: ExperimentResult =
        await response.json();

      setResult(data);
      setSelectedExperimentId(
        data.experiment_id
      );
      setStatus("Experiment completed");

      await loadExperiments();
      await loadVisualization();
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to connect to backend.";

      setError(message);
      setStatus("Experiment failed");
    } finally {
      setLoading(false);
    }
  };

  /* =======================================================
     RUN SINGLE SIMULATION TRACE
  ======================================================= */

  const runTrace = async () => {
    setTraceLoading(true);
    setTraceError(null);
    setTrace(null);
    setStatus("Generating single simulation trace...");

    try {
      const traceConfig = {
        qec_code: config.qec_code,
        num_qubits: config.num_qubits,
        logical_state: config.logical_state,
        rounds: config.rounds,
        physical_noise_probability:
          config.physical_noise_probability,
        measurement_noise_probability:
          config.measurement_noise_probability,
        training_samples: config.training_samples,
        random_forest_estimators:
          config.random_forest_estimators,
        seed: config.seed,
      };

      const response = await fetch(
        `${API_BASE_URL}/simulate/trace`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(traceConfig),
        }
      );

      if (!response.ok) {
        let message =
          `Trace request failed (${response.status})`;

        try {
          const errorData = await response.json();

          if (errorData?.detail) {
            message = String(errorData.detail);
          }
        } catch {
          // Keep default message.
        }

        throw new Error(message);
      }

      const data: SimulationTrace =
        await response.json();

      setTrace(data);
      setStatus(
        data.recovery.logical_success
          ? "Trace completed — logical success"
          : "Trace completed — logical failure"
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to generate trace.";

      setTraceError(message);
      setStatus("Trace failed");
    } finally {
      setTraceLoading(false);
    }
  };

  /* =======================================================
     FILTER OPTIONS
  ======================================================= */

  const roundOptions = useMemo(() => {
    return Array.from(
      new Set(
        experiments.map(
          (experiment) =>
            experiment.rounds
        )
      )
    ).sort((a, b) => a - b);
  }, [experiments]);

  const physicalNoiseOptions = useMemo(() => {
    return Array.from(
      new Set(
        experiments.map(
          (experiment) =>
            experiment.physical_noise_probability
        )
      )
    ).sort((a, b) => a - b);
  }, [experiments]);

  const measurementNoiseOptions =
    useMemo(() => {
      return Array.from(
        new Set(
          experiments.map(
            (experiment) =>
              experiment.measurement_noise_probability
          )
        )
      ).sort((a, b) => a - b);
    }, [experiments]);

  const decoderOptions = useMemo(() => {
    return Array.from(
      new Set(
        experiments.map(
          (experiment) =>
            experiment.decoder_type
        )
      )
    ).sort();
  }, [experiments]);

  /* =======================================================
     FILTER + SORT
  ======================================================= */

  const filteredExperiments = useMemo(() => {
    const filtered =
      experiments.filter((experiment) => {
        const roundsMatch =
          roundFilter === "all" ||
          experiment.rounds ===
            Number(roundFilter);

        const physicalNoiseMatch =
          physicalNoiseFilter === "all" ||
          experiment.physical_noise_probability ===
            Number(
              physicalNoiseFilter
            );

        const measurementNoiseMatch =
          measurementNoiseFilter === "all" ||
          experiment.measurement_noise_probability ===
            Number(
              measurementNoiseFilter
            );

        const decoderMatch =
          decoderFilter === "all" ||
          experiment.decoder_type ===
            decoderFilter;

        return (
          roundsMatch &&
          physicalNoiseMatch &&
          measurementNoiseMatch &&
          decoderMatch
        );
      });

    return [...filtered].sort(
      (a, b) => {
        const aValue =
          a[
            sortMetric as keyof ExperimentSummary
          ] as number;

        const bValue =
          b[
            sortMetric as keyof ExperimentSummary
          ] as number;

        if (sortDescending) {
          return bValue - aValue;
        }

        return aValue - bValue;
      }
    );
  }, [
    experiments,
    roundFilter,
    physicalNoiseFilter,
    measurementNoiseFilter,
    decoderFilter,
    sortMetric,
    sortDescending,
  ]);

  /* =======================================================
     BEST / WORST
  ======================================================= */

  const bestExperiment =
    filteredExperiments.length > 0
      ? filteredExperiments[0]
      : null;

  const worstExperiment =
    filteredExperiments.length > 0
      ? filteredExperiments[
          filteredExperiments.length - 1
        ]
      : null;

  /* =======================================================
     RESET FILTERS
  ======================================================= */

  const resetFilters = () => {
    setRoundFilter("all");
    setPhysicalNoiseFilter("all");
    setMeasurementNoiseFilter("all");
    setDecoderFilter("all");
    setSortMetric("logical_accuracy");
    setSortDescending(true);
  };

  /* =======================================================
     EXPERIMENT SELECTION
  ======================================================= */

  const toggleExperimentSelection = (
    experimentId: string
  ) => {
    setSelectedExperiments(
      (current) => {
        if (
          current.includes(experimentId)
        ) {
          return current.filter(
            (id) =>
              id !== experimentId
          );
        }

        if (current.length >= 3) {
          return current;
        }

        return [
          ...current,
          experimentId,
        ];
      }
    );
  };

  const clearExperimentSelection = () => {
    setSelectedExperiments([]);
  };

  const selectedComparisonExperiments =
    experiments.filter((experiment) =>
      selectedExperiments.includes(
        experiment.experiment_id
      )
    );

  /* =======================================================
     INITIAL LOAD
  ======================================================= */

  useEffect(() => {
    loadExperiments();
    loadVisualization();
    loadScientificEvaluation();
  }, []);

  /* =======================================================
     PAGE
  ======================================================= */

  return (
    <main className="min-h-screen bg-[#07111f] text-white">

      <SiteHeader />

      <div className="mx-auto max-w-7xl px-6 py-8">

        {/* MAIN GRID */}

        <div className="grid gap-6 lg:grid-cols-[420px_1fr]">

          {/* CONFIGURATION */}

          <section className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

            <div className="mb-6">

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Experiment
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Configuration
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Configure the quantum error-correction
                experiment.
              </p>

            </div>

            <div className="space-y-5">

              <InputSelect
                label="QEC Code"
                value={config.qec_code}
                onChange={(value) =>
                  updateConfig(
                    "qec_code",
                    value
                  )
                }
                options={[
                  {
                    value: "bit_flip_3",
                    label: "bit_flip_3",
                  },
                ]}
              />

              <NumberInput
                label="Number of Qubits"
                value={config.num_qubits}
                min={3}
                onChange={(value) =>
                  updateConfig(
                    "num_qubits",
                    value
                  )
                }
              />

              <InputSelect
                label="Logical State"
                value={String(
                  config.logical_state
                )}
                onChange={(value) =>
                  updateConfig(
                    "logical_state",
                    Number(value)
                  )
                }
                options={[
                  {
                    value: "0",
                    label: "0",
                  },
                  {
                    value: "1",
                    label: "1",
                  },
                ]}
              />

              <NumberInput
                label="Syndrome Rounds"
                value={config.rounds}
                min={1}
                onChange={(value) =>
                  updateConfig(
                    "rounds",
                    value
                  )
                }
              />

              <NumberInput
                label="Physical Noise Probability"
                value={
                  config.physical_noise_probability
                }
                min={0}
                max={1}
                step={0.01}
                onChange={(value) =>
                  updateConfig(
                    "physical_noise_probability",
                    value
                  )
                }
              />

              <NumberInput
                label="Measurement Noise Probability"
                value={
                  config.measurement_noise_probability
                }
                min={0}
                max={1}
                step={0.01}
                onChange={(value) =>
                  updateConfig(
                    "measurement_noise_probability",
                    value
                  )
                }
              />

              <NumberInput
                label="Training Samples"
                value={
                  config.training_samples
                }
                min={1}
                onChange={(value) =>
                  updateConfig(
                    "training_samples",
                    value
                  )
                }
              />

              <NumberInput
                label="Test Samples"
                value={
                  config.test_samples
                }
                min={1}
                onChange={(value) =>
                  updateConfig(
                    "test_samples",
                    value
                  )
                }
              />

              <NumberInput
                label="Random Forest Estimators"
                value={
                  config.random_forest_estimators
                }
                min={1}
                onChange={(value) =>
                  updateConfig(
                    "random_forest_estimators",
                    value
                  )
                }
              />

              <NumberInput
                label="Random Seed"
                value={config.seed}
                min={0}
                onChange={(value) =>
                  updateConfig(
                    "seed",
                    value
                  )
                }
              />

              <div className="flex gap-3 pt-2">

                <button
                  type="button"
                  onClick={() => {
                    setError(null);
                    setStatus(
                      "Configuration ready"
                    );
                  }}
                  className="flex-1 rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5"
                >
                  Validate
                </button>

                <button
                  type="button"
                  onClick={resetExperiment}
                  className="flex-1 rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5"
                >
                  Reset
                </button>

              </div>

              <button
                type="button"
                onClick={runExperiment}
                disabled={loading}
                className="w-full rounded-xl bg-cyan-500 px-4 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading
                  ? "Running Experiment..."
                  : "Run Experiment"}
              </button>

              <button
                type="button"
                onClick={runTrace}
                disabled={traceLoading || loading}
                className="w-full rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-3 font-semibold text-cyan-300 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {traceLoading
                  ? "Generating Trace..."
                  : "Run Single Simulation Trace"}
              </button>

              {error && (
                <div className="rounded-xl border border-red-400/20 bg-red-400/10 p-4">

                  <p className="text-xs font-semibold uppercase tracking-wide text-red-300">
                    Error
                  </p>

                  <p className="mt-2 break-words text-sm text-red-200">
                    {error}
                  </p>

                </div>
              )}

              {traceError && (
                <div className="rounded-xl border border-amber-400/20 bg-amber-400/10 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-amber-300">
                    Trace Error
                  </p>
                  <p className="mt-2 break-words text-sm text-amber-200">
                    {traceError}
                  </p>
                </div>
              )}

            </div>

          </section>

          {/* RIGHT SIDE */}

          <section className="space-y-6">

            {/* STATUS */}

            <div className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

              <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

                <div>

                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                    System Status
                  </p>

                  <h2 className="mt-2 text-xl font-semibold">
                    Experiment Control
                  </h2>

                </div>

                <div className="flex items-center gap-3 rounded-full border border-white/10 bg-[#07111f] px-4 py-2">

                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      loading
                        ? "animate-pulse bg-yellow-400"
                        : error
                        ? "bg-red-400"
                        : result
                        ? "bg-green-400"
                        : "bg-cyan-400"
                    }`}
                  />

                  <span className="text-sm text-slate-300">
                    {status}
                  </span>

                </div>

              </div>

            </div>

            {/* METRICS */}

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

              <MetricCard
                title="Logical Accuracy"
                value={
                  result
                    ? `${(
                        result.logical_accuracy *
                        100
                      ).toFixed(2)}%`
                    : "--"
                }
              />

              <MetricCard
                title="Physical Accuracy"
                value={
                  result
                    ? `${(
                        result.physical_accuracy *
                        100
                      ).toFixed(2)}%`
                    : "--"
                }
              />

              <MetricCard
                title="Bit Accuracy"
                value={
                  result
                    ? `${(
                        result.bit_accuracy *
                        100
                      ).toFixed(2)}%`
                    : "--"
                }
              />

              <MetricCard
                title="Exact Accuracy"
                value={
                  result
                    ? `${(
                        result.exact_accuracy *
                        100
                      ).toFixed(2)}%`
                    : "--"
                }
              />

            </div>

            {/* CURRENT RESULT */}

            <div className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Experiment Details
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Current Run
              </h2>

              <div className="mt-6 grid gap-4 sm:grid-cols-2">

                <InfoRow
                  label="Experiment ID"
                  value={
                    result?.experiment_id ??
                    "--"
                  }
                />

                <InfoRow
                  label="Decoder"
                  value={
                    result?.decoder_type ??
                    "--"
                  }
                />

                <InfoRow
                  label="Logical Targets"
                  value={
                    result
                      ? String(
                          result.logical_targets_learned
                        )
                      : "--"
                  }
                />

                <InfoRow
                  label="Target Score"
                  value={
                    result
                      ? result.average_target_score.toFixed(
                          4
                        )
                      : "--"
                  }
                />

                <InfoRow
                  label="Training Time"
                  value={
                    result
                      ? `${result.training_seconds.toFixed(
                          4
                        )} s`
                      : "--"
                  }
                />

                <InfoRow
                  label="Inference Time"
                  value={
                    result
                      ? `${result.inference_seconds.toFixed(
                          4
                        )} s`
                      : "--"
                  }
                />

                <InfoRow
                  label="Throughput"
                  value={
                    result
                      ? `${result.samples_per_second.toFixed(
                          2
                        )} samples/s`
                      : "--"
                  }
                />

                <InfoRow
                  label="Status"
                  value={
                    result?.status ??
                    "Not executed"
                  }
                />

              </div>

            </div>

            {/* ACTIVE CONFIG */}

            <div className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Configuration
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Active Experiment
              </h2>

              <div className="mt-6 grid gap-4 sm:grid-cols-2">

                <InfoRow
                  label="QEC Code"
                  value={config.qec_code}
                />

                <InfoRow
                  label="Qubits"
                  value={String(
                    config.num_qubits
                  )}
                />

                <InfoRow
                  label="Logical State"
                  value={String(
                    config.logical_state
                  )}
                />

                <InfoRow
                  label="Rounds"
                  value={String(
                    config.rounds
                  )}
                />

                <InfoRow
                  label="Physical Noise"
                  value={config.physical_noise_probability.toFixed(
                    2
                  )}
                />

                <InfoRow
                  label="Measurement Noise"
                  value={config.measurement_noise_probability.toFixed(
                    2
                  )}
                />

              </div>

            </div>

            {/* PIPELINE */}

            <div className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Processing Pipeline
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Quantum → AI → Recovery
              </h2>

              <div className="mt-6 grid gap-3 md:grid-cols-4">

                {[
                  "Quantum Simulation",
                  "Noise + Syndrome",
                  "AI Decoder",
                  "Logical Recovery",
                ].map(
                  (step, index) => (
                    <div
                      key={step}
                      className="rounded-xl border border-white/10 bg-[#07111f] p-4"
                    >

                      <p className="text-xs text-cyan-400">
                        STEP {index + 1}
                      </p>

                      <p className="mt-2 text-sm font-semibold text-slate-200">
                        {step}
                      </p>

                    </div>
                  )
                )}

              </div>



            {/* =================================================
                SINGLE SIMULATION TRACE
            ================================================= */}

            <section className="rounded-2xl border border-cyan-400/20 bg-[#0b1b2d] p-6 shadow-xl">

              <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                    Live QEC Trace
                  </p>
                  <h2 className="mt-2 text-xl font-semibold">
                    Single Simulation Trace
                  </h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Follow one noisy quantum sample from encoding through AI decoding and logical recovery.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={runTrace}
                  disabled={traceLoading || loading}
                  className="rounded-xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {traceLoading ? "Running Trace..." : "Run Trace"}
                </button>
              </div>

              {!trace && !traceLoading && (
                <div className="mt-6 rounded-xl border border-dashed border-white/10 bg-[#07111f] p-8 text-center">
                  <p className="text-sm font-semibold text-slate-300">
                    No trace generated yet.
                  </p>
                  <p className="mt-2 text-sm text-slate-500">
                    Use the configuration above and run a single trace to inspect the complete QEC path.
                  </p>
                </div>
              )}

              {traceLoading && (
                <div className="mt-6 rounded-xl border border-cyan-400/20 bg-cyan-400/5 p-8 text-center">
                  <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-cyan-400/20 border-t-cyan-400" />
                  <p className="mt-4 text-sm font-semibold text-cyan-300">
                    Training decoder and generating trace...
                  </p>
                  <p className="mt-2 text-xs text-slate-500">
                    This uses the same backend decoder as the full experiment.
                  </p>
                </div>
              )}

              {trace && (
                <div className="mt-8 space-y-6">

                  {/* TOP RESULT */}
                  <div className={`rounded-2xl border p-5 ${
                    trace.recovery.logical_success
                      ? "border-green-400/30 bg-green-400/5"
                      : "border-red-400/30 bg-red-400/5"
                  }`}>
                    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                      <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                          Final Logical Result
                        </p>
                        <p className={`mt-2 text-2xl font-bold ${
                          trace.recovery.logical_success
                            ? "text-green-300"
                            : "text-red-300"
                        }`}>
                          {trace.recovery.logical_success
                            ? "✓ LOGICAL SUCCESS"
                            : "✕ LOGICAL FAILURE"}
                        </p>
                      </div>

                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <TraceMetric label="Logical" value={String(trace.recovery.recovered_logical_state)} />
                        <TraceMetric label="Encoded" value={trace.encoded_state} />
                        <TraceMetric label="Correction" value={trace.decoder.predicted_correction} />
                        <TraceMetric
                          label="Confidence"
                          value={
                            trace.decoder.confidence === null
                              ? "--"
                              : `${(trace.decoder.confidence * 100).toFixed(1)}%`
                          }
                        />
                      </div>
                    </div>
                  </div>

                  {/* STATE FLOW */}
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                      State Transformation
                    </p>
                    <div className="mt-4 grid gap-3 md:grid-cols-5">
                      <TraceStateCard label="Encoded" value={trace.quantum_state.encoded} />
                      <TraceArrow />
                      <TraceStateCard label="Corrupted" value={trace.quantum_state.corrupted} danger />
                      <TraceArrow />
                      <TraceStateCard label="Corrected" value={trace.correction.corrected_state} success={trace.recovery.logical_success} />
                    </div>
                  </div>

                  {/* ROUND TIMELINE */}
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                      Round-by-Round Syndrome Timeline
                    </p>

                    <div className="mt-4 space-y-3">
                      {trace.syndrome.rounds.map((round) => (
                        <div
                          key={round.round}
                          className="rounded-xl border border-white/10 bg-[#07111f] p-4"
                        >
                          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                            <div className="flex items-center gap-3">
                              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-cyan-400/10 text-sm font-bold text-cyan-300">
                                {round.round}
                              </span>
                              <div>
                                <p className="text-sm font-semibold text-slate-200">
                                  Round {round.round}
                                </p>
                                <p className="text-xs text-slate-500">
                                  Physical error state
                                </p>
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                              <TraceMiniValue label="Error" value={round.physical_error_state} />
                              <TraceMiniValue label="Perfect" value={round.perfect_syndrome} />
                              <TraceMiniValue label="Observed" value={round.observed_syndrome} />
                              <TraceMiniValue label="Detection" value={round.detection_event} />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* AI DECODER */}
                  <div className="rounded-xl border border-cyan-400/20 bg-cyan-400/5 p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                      AI Decoder
                    </p>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                      <TraceInfo label="Decoder" value={trace.decoder.type} />
                      <TraceInfo label="Training Samples" value={String(trace.decoder.training_samples)} />
                      <TraceInfo label="RF Estimators" value={String(trace.decoder.random_forest_estimators)} />
                      <TraceInfo label="Prediction" value={trace.decoder.predicted_correction} />
                    </div>
                  </div>

                  {/* CORRECTION + RECOVERY */}
                  <div className="grid gap-4 lg:grid-cols-2">
                    <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
                      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                        Correction
                      </p>
                      <div className="mt-4 space-y-3">
                        <TraceInfo label="Actual Error" value={trace.correction.actual_error} />
                        <TraceInfo label="Predicted Correction" value={trace.correction.predicted_correction} />
                        <TraceInfo label="Corrected State" value={trace.correction.corrected_state} />
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
                      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                        Recovery Validation
                      </p>
                      <div className="mt-4 grid grid-cols-2 gap-3">
                        <TraceBoolean label="Logical Success" value={trace.recovery.logical_success} />
                        <TraceBoolean label="Physical Recovery" value={trace.recovery.physical_recovery} />
                        <TraceBoolean label="Exact Error Match" value={trace.recovery.exact_error_match} />
                        <TraceBoolean label="Logical Failure" value={trace.recovery.logical_failure} negative />
                      </div>
                    </div>
                  </div>

                  {/* NOISE DETAILS */}
                  <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                      Noise Details
                    </p>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                      <TraceInfo label="Physical Noise" value={trace.noise.physical_error_probability.toFixed(2)} />
                      <TraceInfo label="Measurement Noise" value={trace.noise.measurement_noise_probability.toFixed(2)} />
                      <TraceInfo label="Final Error" value={trace.noise.final_error_state} />
                      <TraceInfo label="Final Syndrome" value={trace.syndrome.final_perfect} />
                    </div>
                    <p className="mt-4 rounded-lg border border-white/5 bg-[#0b1b2d] p-3 text-xs text-slate-500">
                      {trace.noise.final_error_description}
                    </p>
                  </div>

                </div>
              )}

            </section>
            </div>

          </section>

        </div>

        {/* =================================================
            SCIENTIFIC EVALUATION
        ================================================= */}

        <section className="mt-6 rounded-2xl border border-cyan-400/20 bg-[#0b1b2d] p-6 shadow-xl">
          <div className="mb-6">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Scientific Evaluation
            </p>
            <h2 className="mt-2 text-xl font-semibold">
              Baseline vs AI-QEC
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Paired held-out evaluation across physical-noise conditions.
            </p>
          </div>

          {scientificLoading && (
            <div className="rounded-xl border border-white/10 bg-[#07111f] p-5 text-sm text-slate-400">
              Loading scientific evaluation...
            </div>
          )}

          {scientificError && !scientificLoading && (
            <div className="rounded-xl border border-red-400/20 bg-red-400/5 p-5 text-sm text-red-300">
              {scientificError}
            </div>
          )}

          {scientificEvaluation && !scientificLoading && (
            <div className="space-y-6">
              {(() => {
                const data = scientificEvaluation;
                const maxGain =
                  data.summary.maximum_absolute_gain.absolute_gain ??
                  data.summary.maximum_absolute_gain.gain ?? 0;
                const maxReduction =
                  data.summary.maximum_error_reduction.logical_error_reduction ??
                  data.summary.maximum_error_reduction.reduction ?? 0;
                const maxGainNoise = data.summary.maximum_absolute_gain.physical_noise;
                const maxReductionNoise = data.summary.maximum_error_reduction.physical_noise;

                return (
                  <>
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <ScientificMetric
                        label="Conditions"
                        value={String(data.summary.conditions)}
                        detail={`${data.summary.positive_gain_conditions} with positive AI gain`}
                      />
                      <ScientificMetric
                        label="Maximum AI Gain"
                        value={`+${(maxGain * 100).toFixed(2)} pp`}
                        detail={`at ${maxGainNoise.toFixed(2)} physical noise`}
                        highlight
                      />
                      <ScientificMetric
                        label="Logical Error Reduction"
                        value={`${(maxReduction * 100).toFixed(2)}%`}
                        detail={`at ${maxReductionNoise.toFixed(2)} physical noise`}
                        highlight
                      />
                      <ScientificMetric
                        label="Positive Conditions"
                        value={`${data.summary.positive_gain_conditions}/${data.summary.conditions}`}
                        detail="AI-QEC outperformed baseline"
                      />
                    </div>

                    <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
                      <div className="mb-5 flex flex-col justify-between gap-2 md:flex-row md:items-end">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                            Logical Success vs Physical Noise
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            Same held-out samples used for baseline and AI-QEC.
                          </p>
                        </div>
                        <div className="flex gap-4 text-xs text-slate-400">
                          <span>Baseline</span>
                          <span className="text-cyan-300">AI-QEC</span>
                        </div>
                      </div>

                      <div className="space-y-4">
                        {data.results.map((row) => (
                          <div key={`${row.physical_noise}-${row.measurement_noise}`} className="grid gap-2 md:grid-cols-[70px_1fr_75px] md:items-center">
                            <span className="text-sm font-semibold text-slate-300">
                              {(row.physical_noise * 100).toFixed(0)}%
                            </span>
                            <div className="space-y-2">
                              <ScientificBar label="Baseline" value={row.baseline_logical_success} />
                              <ScientificBar label="AI-QEC" value={row.ai_logical_success} highlight />
                            </div>
                            <div className="text-right text-sm font-semibold text-cyan-300">
                              +{(row.absolute_gain * 100).toFixed(2)} pp
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid gap-6 xl:grid-cols-2">
                      <ScientificLineChart
                        title="Logical Success vs Physical Noise"
                        data={data.results}
                        baselineKey="baseline_logical_success"
                        aiKey="ai_logical_success"
                      />
                      <ScientificGainChart data={data.results} />
                    </div>

                    <div className="grid gap-6 xl:grid-cols-2">
                      <ScientificLineChart
                        title="Logical Error Rate vs Physical Noise"
                        data={data.results}
                        baselineKey="baseline_logical_error"
                        aiKey="ai_logical_error"
                      />
                      <div className="rounded-xl border border-white/10 bg-[#07111f] p-5">
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Key Scientific Findings</p>
                        <div className="mt-5 space-y-3 text-sm text-slate-300">
                          <div className="rounded-lg border border-white/5 bg-[#0b1b2d] p-4">
                            <span className="font-semibold text-cyan-300">Maximum absolute gain:</span>{" "}
                            +{(maxGain * 100).toFixed(2)} pp at {(maxGainNoise * 100).toFixed(0)}% physical noise.
                          </div>
                          <div className="rounded-lg border border-white/5 bg-[#0b1b2d] p-4">
                            <span className="font-semibold text-cyan-300">Maximum logical-error reduction:</span>{" "}
                            {(maxReduction * 100).toFixed(2)}% at {(maxReductionNoise * 100).toFixed(0)}% physical noise.
                          </div>
                          <div className="rounded-lg border border-white/5 bg-[#0b1b2d] p-4 text-xs leading-5 text-slate-400">
                            Results come from paired held-out evaluations. Statistical intervals and permutation p-values are retained from the underlying per-seed experiments.
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="overflow-hidden rounded-xl border border-white/10 bg-[#07111f]">
                      <div className="border-b border-white/10 p-5">
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                          Statistical Results
                        </p>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-left">
                          <thead className="bg-[#0b1b2d]">
                            <tr className="text-xs uppercase tracking-wider text-slate-500">
                              <th className="px-4 py-3">Noise</th>
                              <th className="px-4 py-3">Baseline</th>
                              <th className="px-4 py-3">AI-QEC</th>
                              <th className="px-4 py-3">Gain</th>
                              <th className="px-4 py-3">Error Reduction</th>
                              <th className="px-4 py-3">95% CI</th>
                              <th className="px-4 py-3">p-value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {data.results.map((row) => (
                              <tr key={`stat-${row.physical_noise}-${row.measurement_noise}`} className="border-t border-white/5">
                                <td className="px-4 py-3 text-sm text-slate-300">
                                  {(row.physical_noise * 100).toFixed(0)}%
                                </td>
                                <td className="px-4 py-3 text-sm text-slate-300">
                                  {(row.baseline_logical_success * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-3 text-sm font-semibold text-cyan-300">
                                  {(row.ai_logical_success * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-3 text-sm font-semibold text-emerald-300">
                                  +{(row.absolute_gain * 100).toFixed(2)} pp
                                </td>
                                <td className="px-4 py-3 text-sm text-slate-300">
                                  {(row.logical_error_reduction * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-3 text-xs text-slate-400">
                                  [{(row.bootstrap_ci_low * 100).toFixed(2)}%, {(row.bootstrap_ci_high * 100).toFixed(2)}%]
                                </td>
                                <td className="px-4 py-3 text-xs text-slate-400">
                                  {row.permutation_p_value.toFixed(5)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          )}
        </section>

        {/* =================================================
            ANALYSIS
        ================================================= */}

        <section className="mt-6 rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

          <div className="mb-6">

            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
              Experiment Analysis
            </p>

            <h2 className="mt-2 text-xl font-semibold">
              Filter & Compare Results
            </h2>

            <p className="mt-2 text-sm text-slate-400">
              Analyze stored experiments using
              configuration filters and performance
              metrics.
            </p>

          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">

            <FilterSelect
              label="Rounds"
              value={roundFilter}
              onChange={setRoundFilter}
              options={[
                {
                  value: "all",
                  label: "All Rounds",
                },
                ...roundOptions.map(
                  (value) => ({
                    value: String(value),
                    label: `${value} rounds`,
                  })
                ),
              ]}
            />

            <FilterSelect
              label="Physical Noise"
              value={physicalNoiseFilter}
              onChange={setPhysicalNoiseFilter}
              options={[
                {
                  value: "all",
                  label: "All Physical Noise",
                },
                ...physicalNoiseOptions.map(
                  (value) => ({
                    value: String(value),
                    label: value.toFixed(2),
                  })
                ),
              ]}
            />

            <FilterSelect
              label="Measurement Noise"
              value={measurementNoiseFilter}
              onChange={
                setMeasurementNoiseFilter
              }
              options={[
                {
                  value: "all",
                  label: "All Measurement Noise",
                },
                ...measurementNoiseOptions.map(
                  (value) => ({
                    value: String(value),
                    label: value.toFixed(2),
                  })
                ),
              ]}
            />

            <FilterSelect
              label="Decoder"
              value={decoderFilter}
              onChange={setDecoderFilter}
              options={[
                {
                  value: "all",
                  label: "All Decoders",
                },
                ...decoderOptions.map(
                  (value) => ({
                    value,
                    label: value,
                  })
                ),
              ]}
            />

          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-2">

            <FilterSelect
              label="Sort By"
              value={sortMetric}
              onChange={setSortMetric}
              options={[
                {
                  value: "logical_accuracy",
                  label: "Logical Accuracy",
                },
                {
                  value: "physical_accuracy",
                  label: "Physical Accuracy",
                },
                {
                  value: "bit_accuracy",
                  label: "Bit Accuracy",
                },
                {
                  value: "exact_accuracy",
                  label: "Exact Accuracy",
                },
                {
                  value: "training_seconds",
                  label: "Training Time",
                },
                {
                  value: "inference_seconds",
                  label: "Inference Time",
                },
                {
                  value: "samples_per_second",
                  label: "Throughput",
                },
              ]}
            />

            <div>

              <label className="mb-2 block text-sm text-slate-300">
                Sort Direction
              </label>

              <div className="flex gap-3">

                <button
                  type="button"
                  onClick={() =>
                    setSortDescending(true)
                  }
                  className={`flex-1 rounded-xl border px-4 py-3 text-sm font-semibold transition ${
                    sortDescending
                      ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-300"
                      : "border-white/10 text-slate-400 hover:bg-white/5"
                  }`}
                >
                  Descending
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setSortDescending(false)
                  }
                  className={`flex-1 rounded-xl border px-4 py-3 text-sm font-semibold transition ${
                    !sortDescending
                      ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-300"
                      : "border-white/10 text-slate-400 hover:bg-white/5"
                  }`}
                >
                  Ascending
                </button>

              </div>

            </div>

          </div>

          <button
            type="button"
            onClick={resetFilters}
            className="mt-4 rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5"
          >
            Reset Filters
          </button>

          <div className="mt-6 grid gap-4 md:grid-cols-2">

            <AnalysisCard
              title="Best Experiment"
              experiment={bestExperiment}
              positive
            />

            <AnalysisCard
              title="Worst Experiment"
              experiment={worstExperiment}
              positive={false}
            />

          </div>

          <div className="mt-6 rounded-xl border border-white/10 bg-[#07111f] p-4">

            <p className="text-sm text-slate-400">
              Showing
            </p>

            <p className="mt-1 text-2xl font-bold text-cyan-300">
              {filteredExperiments.length}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              of {experiments.length} stored
              experiments
            </p>

          </div>

        </section>

        {/* =================================================
            COMPARISON
        ================================================= */}

        <section className="mt-6 rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

            <div>

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Experiment Comparison
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Compare Selected Runs
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Select up to three experiments from
                the history table.
              </p>

            </div>

            <button
              type="button"
              onClick={
                clearExperimentSelection
              }
              disabled={
                selectedExperiments.length === 0
              }
              className="rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Clear Selection
            </button>

          </div>

          <div className="mt-6 rounded-xl border border-white/10 bg-[#07111f] p-4">

            <div className="flex items-center justify-between">

              <div>

                <p className="text-xs uppercase tracking-wide text-slate-500">
                  Selected Experiments
                </p>

                <p className="mt-1 text-2xl font-bold text-cyan-300">
                  {selectedExperiments.length}
                </p>

              </div>

              <p className="text-sm text-slate-500">
                Maximum 3
              </p>

            </div>

          </div>

          {selectedComparisonExperiments.length >
            0 && (
            <div className="mt-6 overflow-x-auto">

              <table className="w-full min-w-[900px] border-collapse">

                <thead>

                  <tr className="border-b border-white/10">

                    <th className="px-4 py-3 text-left text-xs uppercase tracking-wide text-slate-500">
                      Metric
                    </th>

                    {selectedComparisonExperiments.map(
                      (experiment) => (
                        <th
                          key={
                            experiment.experiment_id
                          }
                          className="px-4 py-3 text-left text-xs uppercase tracking-wide text-slate-500"
                        >
                          <span className="font-mono">
                            {
                              experiment.experiment_id
                            }
                          </span>
                        </th>
                      )
                    )}

                  </tr>

                </thead>

                <tbody>

                  <ComparisonRow
                    label="Logical Accuracy"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${(
                        experiment.logical_accuracy *
                        100
                      ).toFixed(2)}%`
                    }
                  />

                  <ComparisonRow
                    label="Physical Accuracy"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${(
                        experiment.physical_accuracy *
                        100
                      ).toFixed(2)}%`
                    }
                  />

                  <ComparisonRow
                    label="Bit Accuracy"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${(
                        experiment.bit_accuracy *
                        100
                      ).toFixed(2)}%`
                    }
                  />

                  <ComparisonRow
                    label="Exact Accuracy"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${(
                        experiment.exact_accuracy *
                        100
                      ).toFixed(2)}%`
                    }
                  />

                  <ComparisonRow
                    label="Rounds"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      String(
                        experiment.rounds
                      )
                    }
                  />

                  <ComparisonRow
                    label="Physical Noise"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      experiment.physical_noise_probability.toFixed(
                        2
                      )
                    }
                  />

                  <ComparisonRow
                    label="Measurement Noise"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      experiment.measurement_noise_probability.toFixed(
                        2
                      )
                    }
                  />

                  <ComparisonRow
                    label="Training Time"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${experiment.training_seconds.toFixed(
                        4
                      )} s`
                    }
                  />

                  <ComparisonRow
                    label="Inference Time"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${experiment.inference_seconds.toFixed(
                        4
                      )} s`
                    }
                  />

                  <ComparisonRow
                    label="Throughput"
                    experiments={
                      selectedComparisonExperiments
                    }
                    getValue={(experiment) =>
                      `${experiment.samples_per_second.toFixed(
                        2
                      )} samples/s`
                    }
                  />

                </tbody>

              </table>

            </div>
          )}

          {selectedComparisonExperiments.length ===
            0 && (
            <div className="mt-6 rounded-xl border border-dashed border-white/10 bg-[#07111f] p-8 text-center">

              <p className="text-sm font-semibold text-slate-300">
                No experiments selected.
              </p>

              <p className="mt-2 text-sm text-slate-500">
                Select experiments from the history
                table below.
              </p>

            </div>
          )}

        </section>

        {/* =================================================
            VISUALIZATION
        ================================================= */}

        <section className="mt-6 rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

            <div>

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Experiment Visualization
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Quantum Error-Correction Analysis
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Visual analysis generated from stored
                experiment results.
              </p>

            </div>

            <button
              type="button"
              onClick={loadVisualization}
              disabled={visualizationLoading}
              className="rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
            >
              {visualizationLoading
                ? "Refreshing..."
                : "Refresh Charts"}
            </button>

          </div>

          {!visualization && (
            <div className="mt-6 rounded-xl border border-dashed border-white/10 bg-[#07111f] p-8 text-center">

              <p className="text-sm text-slate-400">
                No visualization data available yet.
              </p>

            </div>
          )}

          {visualization && (
            <div className="mt-8 space-y-6">

              <div className="grid gap-6 xl:grid-cols-2">

                <LineChartCard
                  title="Logical Success vs Syndrome Rounds"
                  description="How logical success changes as syndrome rounds increase."
                  data={visualization.charts.rounds
                    .filter(
                      (item) =>
                        typeof item.rounds ===
                          "number" &&
                        typeof item.logical_success ===
                          "number"
                    )
                    .map((item) => ({
                      x: item.rounds,
                      y: item.logical_success,
                      xLabel: `${item.rounds}`,
                    }))}
                  xLabel="Rounds"
                />

                <LineChartCard
                  title="Logical Success vs Physical Noise"
                  description="Effect of physical error probability on logical success."
                  data={visualization.charts.physical_noise
                    .filter(
                      (item) =>
                        typeof item.physical_noise ===
                          "number" &&
                        typeof item.logical_success ===
                          "number"
                    )
                    .map((item) => ({
                      x: item.physical_noise as number,
                      y: item.logical_success,
                      xLabel:
                        (item.physical_noise as number).toFixed(
                          2
                        ),
                    }))}
                  xLabel="Physical Noise"
                />

              </div>

              <div className="grid gap-6 xl:grid-cols-2">

                <LineChartCard
                  title="Logical Success vs Measurement Noise"
                  description="Effect of measurement noise on logical success."
                  data={visualization.charts.measurement_noise
                    .filter(
                      (item) =>
                        typeof item.measurement_noise ===
                          "number" &&
                        typeof item.logical_success ===
                          "number"
                    )
                    .map((item) => ({
                      x: item.measurement_noise as number,
                      y: item.logical_success,
                      xLabel:
                        (item.measurement_noise as number).toFixed(
                          2
                        ),
                    }))}
                  xLabel="Measurement Noise"
                />

                <BarChartCard
                  title="Decoder Comparison"
                  description="Logical success achieved by each decoder."
                  data={visualization.charts.decoders
                    .filter(
                      (item) =>
                        typeof item.decoder ===
                          "string" &&
                        typeof item.logical_success ===
                          "number"
                    )
                    .map((item) => ({
                      label: item.decoder,
                      value:
                        item.logical_success,
                    }))}
                />

              </div>

              <PerformanceTable
                data={visualization.performance}
              />

            </div>
          )}

        </section>

        {/* =================================================
            EXPERIMENT HISTORY
        ================================================= */}

        <section className="mt-6 rounded-2xl border border-white/10 bg-[#0b1b2d] p-6 shadow-xl">

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">

            <div>

              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                Experiment Database
              </p>

              <h2 className="mt-2 text-xl font-semibold">
                Experiment History
              </h2>

              <p className="mt-2 text-sm text-slate-400">
                Filtered and sorted stored experiments.
              </p>

            </div>

            <button
              type="button"
              onClick={loadExperiments}
              disabled={historyLoading}
              className="rounded-xl border border-white/10 px-4 py-3 text-sm font-semibold text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
            >
              {historyLoading
                ? "Refreshing..."
                : "Refresh History"}
            </button>

          </div>

          {filteredExperiments.length ===
            0 && (
            <div className="mt-6 rounded-xl border border-dashed border-white/10 bg-[#07111f] p-8 text-center">

              <p className="text-sm font-semibold text-slate-300">
                No matching experiments.
              </p>

              <p className="mt-2 text-sm text-slate-500">
                Try changing the filters or run
                an experiment.
              </p>

            </div>
          )}

          {filteredExperiments.length > 0 && (
            <div className="mt-6 overflow-x-auto">

              <table className="w-full min-w-[1100px] border-collapse">

                <thead>

                  <tr className="border-b border-white/10 text-left">

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Select
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Experiment
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Decoder
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Rounds
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Physical Noise
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Measurement Noise
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Logical
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Physical
                    </th>

                    <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                      Action
                    </th>

                  </tr>

                </thead>

                <tbody>

                  {filteredExperiments.map(
                    (experiment) => {

                      const selected =
                        selectedExperiments.includes(
                          experiment.experiment_id
                        );

                      const selectionDisabled =
                        !selected &&
                        selectedExperiments.length >=
                          3;

                      const currentResult =
                        selectedExperimentId ===
                        experiment.experiment_id;

                      return (
                        <tr
                          key={
                            experiment.experiment_id
                          }
                          className={`border-b border-white/5 transition ${
                            currentResult
                              ? "bg-cyan-400/10"
                              : selected
                              ? "bg-cyan-400/5"
                              : "hover:bg-white/[0.03]"
                          }`}
                        >

                          <td className="px-4 py-4">

                            <input
                              type="checkbox"
                              checked={selected}
                              disabled={
                                selectionDisabled
                              }
                              onChange={() =>
                                toggleExperimentSelection(
                                  experiment.experiment_id
                                )
                              }
                              className="h-4 w-4 cursor-pointer rounded border-white/20 bg-[#07111f] accent-cyan-400 disabled:cursor-not-allowed disabled:opacity-30"
                            />

                          </td>

                          <td className="px-4 py-4">

                            <p className="font-mono text-xs text-slate-300">
                              {
                                experiment.experiment_id
                              }
                            </p>

                            <p className="mt-1 text-xs text-slate-500">
                              {
                                experiment.qec_code
                              }
                            </p>

                          </td>

                          <td className="px-4 py-4 text-sm text-slate-300">
                            {
                              experiment.decoder_type
                            }
                          </td>

                          <td className="px-4 py-4 text-sm text-slate-300">
                            {experiment.rounds}
                          </td>

                          <td className="px-4 py-4 text-sm text-slate-300">
                            {experiment.physical_noise_probability.toFixed(
                              2
                            )}
                          </td>

                          <td className="px-4 py-4 text-sm text-slate-300">
                            {experiment.measurement_noise_probability.toFixed(
                              2
                            )}
                          </td>

                          <td className="px-4 py-4">

                            <span className="font-semibold text-cyan-300">
                              {(
                                experiment.logical_accuracy *
                                100
                              ).toFixed(2)}
                              %
                            </span>

                          </td>

                          <td className="px-4 py-4 text-sm text-slate-300">
                            {(
                              experiment.physical_accuracy *
                              100
                            ).toFixed(2)}
                            %
                          </td>

                          <td className="px-4 py-4">

                            <button
                              type="button"
                              onClick={() =>
                                loadExperiment(
                                  experiment.experiment_id
                                )
                              }
                              className={`rounded-lg border px-3 py-2 text-xs font-semibold transition ${
                                currentResult
                                  ? "border-green-400/30 bg-green-400/10 text-green-300"
                                  : "border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10"
                              }`}
                            >
                              {currentResult
                                ? "Loaded"
                                : "View Result"}
                            </button>

                          </td>

                        </tr>
                      );
                    }
                  )}

                </tbody>

              </table>

            </div>
          )}

        </section>

      </div>

      <SiteFooter />
    </main>
  );
}

/* =========================================================
   TRACE HELPERS
========================================================= */

function TraceMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#07111f] px-4 py-3">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm font-semibold text-slate-200">
        {value}
      </p>
    </div>
  );
}

function TraceStateCard({
  label,
  value,
  danger = false,
  success = false,
}: {
  label: string;
  value: string;
  danger?: boolean;
  success?: boolean;
}) {
  return (
    <div className={`rounded-xl border p-4 ${
      success
        ? "border-green-400/30 bg-green-400/5"
        : danger
        ? "border-red-400/20 bg-red-400/5"
        : "border-white/10 bg-[#07111f]"
    }`}>
      <p className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className={`mt-2 font-mono text-2xl font-bold ${
        success
          ? "text-green-300"
          : danger
          ? "text-red-300"
          : "text-cyan-300"
      }`}>
        {value}
      </p>
    </div>
  );
}

function TraceArrow() {
  return (
    <div className="hidden items-center justify-center text-2xl text-cyan-400 md:flex">
      →
    </div>
  );
}

function TraceMiniValue({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-[#0b1b2d] px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm font-semibold text-cyan-300">
        {value}
      </p>
    </div>
  );
}

function TraceInfo({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-white/5 bg-[#0b1b2d] p-3">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 break-all font-mono text-sm font-semibold text-slate-200">
        {value}
      </p>
    </div>
  );
}

function TraceBoolean({
  label,
  value,
  negative = false,
}: {
  label: string;
  value: boolean;
  negative?: boolean;
}) {
  const positive = value && !negative;
  const negativeResult = value && negative;

  return (
    <div className={`rounded-lg border p-3 ${
      positive
        ? "border-green-400/20 bg-green-400/5"
        : negativeResult
        ? "border-red-400/20 bg-red-400/5"
        : "border-white/5 bg-[#0b1b2d]"
    }`}>
      <p className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className={`mt-1 text-sm font-semibold ${
        positive
          ? "text-green-300"
          : negativeResult
          ? "text-red-300"
          : "text-slate-300"
      }`}>
        {value ? "YES" : "NO"}
      </p>
    </div>
  );
}

/* =========================================================
   NUMBER INPUT
========================================================= */

function NumberInput({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <div>

      <label className="mb-2 block text-sm text-slate-300">
        {label}
      </label>

      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) =>
          onChange(
            Number(event.target.value)
          )
        }
        className="w-full rounded-xl border border-white/10 bg-[#07111f] px-4 py-3 text-sm outline-none focus:border-cyan-400"
      />

    </div>
  );
}

/* =========================================================
   INPUT SELECT
========================================================= */

function InputSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: {
    value: string;
    label: string;
  }[];
  onChange: (value: string) => void;
}) {
  return (
    <div>

      <label className="mb-2 block text-sm text-slate-300">
        {label}
      </label>

      <select
        value={value}
        onChange={(event) =>
          onChange(
            event.target.value
          )
        }
        className="w-full rounded-xl border border-white/10 bg-[#07111f] px-4 py-3 text-sm outline-none focus:border-cyan-400"
      >

        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
          >
            {option.label}
          </option>
        ))}

      </select>

    </div>
  );
}

/* =========================================================
   FILTER SELECT
========================================================= */

function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: {
    value: string;
    label: string;
  }[];
  onChange: (value: string) => void;
}) {
  return (
    <div>

      <label className="mb-2 block text-sm text-slate-300">
        {label}
      </label>

      <select
        value={value}
        onChange={(event) =>
          onChange(
            event.target.value
          )
        }
        className="w-full rounded-xl border border-white/10 bg-[#07111f] px-4 py-3 text-sm outline-none focus:border-cyan-400"
      >

        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
          >
            {option.label}
          </option>
        ))}

      </select>

    </div>
  );
}

/* =========================================================
   METRIC CARD
========================================================= */

function MetricCard({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#0b1b2d] p-5 shadow-xl">

      <p className="text-sm text-slate-400">
        {title}
      </p>

      <p className="mt-3 text-2xl font-bold tracking-tight">
        {value}
      </p>

    </div>
  );
}

/* =========================================================
   INFO ROW
========================================================= */

function InfoRow({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#07111f] p-4">

      <p className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </p>

      <p className="mt-2 break-all text-sm font-medium text-slate-200">
        {value}
      </p>

    </div>
  );
}

/* =========================================================
   ANALYSIS CARD
========================================================= */

function AnalysisCard({
  title,
  experiment,
  positive,
}: {
  title: string;
  experiment: ExperimentSummary | null;
  positive: boolean;
}) {
  if (!experiment) {
    return (
      <div className="rounded-2xl border border-white/10 bg-[#07111f] p-5">

        <p className="text-xs uppercase tracking-wide text-slate-500">
          {title}
        </p>

        <p className="mt-3 text-sm text-slate-500">
          No experiment available.
        </p>

      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-[#07111f] p-5">

      <div className="flex items-center justify-between">

        <p className="text-xs uppercase tracking-wide text-slate-500">
          {title}
        </p>

        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            positive
              ? "bg-green-400/10 text-green-300"
              : "bg-red-400/10 text-red-300"
          }`}
        >
          {positive
            ? "BEST"
            : "LOWEST"}
        </span>

      </div>

      <p className="mt-4 font-mono text-xs text-slate-400">
        {experiment.experiment_id}
      </p>

      <div className="mt-4 grid grid-cols-2 gap-3">

        <SmallMetric
          label="Logical"
          value={`${(
            experiment.logical_accuracy *
            100
          ).toFixed(2)}%`}
        />

        <SmallMetric
          label="Physical"
          value={`${(
            experiment.physical_accuracy *
            100
          ).toFixed(2)}%`}
        />

        <SmallMetric
          label="Rounds"
          value={String(
            experiment.rounds
          )}
        />

        <SmallMetric
          label="Noise"
          value={experiment.physical_noise_probability.toFixed(
            2
          )}
        />

      </div>

    </div>
  );
}

/* =========================================================
   SMALL METRIC
========================================================= */

function SmallMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[#0b1b2d] p-3">

      <p className="text-[10px] uppercase tracking-wide text-slate-500">
        {label}
      </p>

      <p className="mt-1 text-sm font-semibold text-slate-200">
        {value}
      </p>

    </div>
  );
}

/* =========================================================
   COMPARISON ROW
========================================================= */

function ComparisonRow({
  label,
  experiments,
  getValue,
}: {
  label: string;
  experiments: ExperimentSummary[];
  getValue: (
    experiment: ExperimentSummary
  ) => string;
}) {
  return (
    <tr className="border-b border-white/5">

      <td className="px-4 py-4 text-sm font-medium text-slate-300">
        {label}
      </td>

      {experiments.map(
        (experiment) => (
          <td
            key={
              experiment.experiment_id
            }
            className="px-4 py-4 text-sm text-slate-200"
          >
            {getValue(experiment)}
          </td>
        )
      )}

    </tr>
  );
}

/* =========================================================
   LINE CHART
========================================================= */

function LineChartCard({
  title,
  description,
  data,
  xLabel,
}: {
  title: string;
  description: string;
  data: {
    x: number;
    y: number;
    xLabel: string;
  }[];
  xLabel: string;
}) {
  const width = 720;
  const height = 360;

  const paddingLeft = 65;
  const paddingRight = 25;
  const paddingTop = 35;
  const paddingBottom = 65;

  const plotWidth =
    width -
    paddingLeft -
    paddingRight;

  const plotHeight =
    height -
    paddingTop -
    paddingBottom;

  if (data.length === 0) {
    return (
      <ChartShell
        title={title}
        description={description}
      >
        <div className="flex h-[300px] items-center justify-center text-sm text-slate-500">
          No numeric data available.
        </div>
      </ChartShell>
    );
  }

  const minX = Math.min(
    ...data.map((item) => item.x)
  );

  const maxX = Math.max(
    ...data.map((item) => item.x)
  );

  const xRange =
    maxX === minX
      ? 1
      : maxX - minX;

  const xPosition = (x: number) =>
    paddingLeft +
    ((x - minX) / xRange) *
      plotWidth;

  const yPosition = (y: number) =>
    paddingTop +
    (1 - y) *
      plotHeight;

  const points = data
    .map(
      (item) =>
        `${xPosition(item.x)},${yPosition(
          item.y
        )}`
    )
    .join(" ");

  return (
    <ChartShell
      title={title}
      description={description}
    >

      <div className="overflow-x-auto">

        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto min-w-[650px] w-full"
          role="img"
          aria-label={title}
        >

          {/* Y GRID */}

          {[0, 0.25, 0.5, 0.75, 1].map(
            (value) => {

              const y =
                yPosition(value);

              return (
                <g key={value}>

                  <line
                    x1={paddingLeft}
                    y1={y}
                    x2={
                      width -
                      paddingRight
                    }
                    y2={y}
                    stroke="currentColor"
                    className="text-white/10"
                    strokeWidth="1"
                  />

                  <text
                    x={
                      paddingLeft - 10
                    }
                    y={y + 4}
                    textAnchor="end"
                    className="fill-slate-500 text-[11px]"
                  >
                    {Math.round(
                      value * 100
                    )}
                    %
                  </text>

                </g>
              );
            }
          )}

          {/* AXIS */}

          <line
            x1={paddingLeft}
            y1={
              paddingTop +
              plotHeight
            }
            x2={
              width -
              paddingRight
            }
            y2={
              paddingTop +
              plotHeight
            }
            stroke="currentColor"
            className="text-white/20"
          />

          <line
            x1={paddingLeft}
            y1={paddingTop}
            x2={paddingLeft}
            y2={
              paddingTop +
              plotHeight
            }
            stroke="currentColor"
            className="text-white/20"
          />

          {/* LINE */}

          <polyline
            points={points}
            fill="none"
            stroke="currentColor"
            className="text-cyan-400"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* POINTS */}

          {data.map(
            (item, index) => {

              const x =
                xPosition(item.x);

              const y =
                yPosition(item.y);

              return (
                <g key={index}>

                  <circle
                    cx={x}
                    cy={y}
                    r="5"
                    fill="currentColor"
                    className="text-cyan-400"
                  />

                  <text
                    x={x}
                    y={y - 12}
                    textAnchor="middle"
                    className="fill-slate-300 text-[10px]"
                  >
                    {(
                      item.y * 100
                    ).toFixed(1)}
                    %
                  </text>

                  <text
                    x={x}
                    y={
                      paddingTop +
                      plotHeight +
                      25
                    }
                    textAnchor="middle"
                    className="fill-slate-500 text-[10px]"
                  >
                    {item.xLabel}
                  </text>

                </g>
              );
            }
          )}

          {/* X LABEL */}

          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            className="fill-slate-500 text-[11px]"
          >
            {xLabel}
          </text>

        </svg>

      </div>

    </ChartShell>
  );
}

/* =========================================================
   BAR CHART
========================================================= */

function BarChartCard({
  title,
  description,
  data,
}: {
  title: string;
  description: string;
  data: {
    label: string;
    value: number;
  }[];
}) {
  const width = 720;
  const height = 360;

  const paddingLeft = 60;
  const paddingRight = 25;
  const paddingTop = 35;
  const paddingBottom = 55;

  const plotWidth =
    width -
    paddingLeft -
    paddingRight;

  const plotHeight =
    height -
    paddingTop -
    paddingBottom;

  if (data.length === 0) {
    return (
      <ChartShell
        title={title}
        description={description}
      >
        <div className="flex h-[300px] items-center justify-center text-sm text-slate-500">
          No decoder data available.
        </div>
      </ChartShell>
    );
  }

  const barGap = 18;

  const barWidth =
    Math.max(
      30,
      (plotWidth -
        barGap *
          (data.length - 1)) /
        data.length
    );

  return (
    <ChartShell
      title={title}
      description={description}
    >

      <div className="overflow-x-auto">

        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto min-w-[650px] w-full"
          role="img"
          aria-label={title}
        >

          {[0, 0.25, 0.5, 0.75, 1].map(
            (value) => {

              const y =
                paddingTop +
                (1 - value) *
                  plotHeight;

              return (
                <g key={value}>

                  <line
                    x1={paddingLeft}
                    y1={y}
                    x2={
                      width -
                      paddingRight
                    }
                    y2={y}
                    stroke="currentColor"
                    className="text-white/10"
                  />

                  <text
                    x={
                      paddingLeft - 10
                    }
                    y={y + 4}
                    textAnchor="end"
                    className="fill-slate-500 text-[11px]"
                  >
                    {Math.round(
                      value * 100
                    )}
                    %
                  </text>

                </g>
              );
            }
          )}

          {data.map(
            (item, index) => {

              const x =
                paddingLeft +
                index *
                  (barWidth +
                    barGap);

              const barHeight =
                Math.max(
                  0,
                  Math.min(
                    1,
                    item.value
                  )
                ) *
                plotHeight;

              const y =
                paddingTop +
                plotHeight -
                barHeight;

              return (
                <g key={`${item.label}-${index}`}>

                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    rx="6"
                    fill="currentColor"
                    className="text-cyan-400"
                    opacity="0.8"
                  />

                  <text
                    x={
                      x +
                      barWidth / 2
                    }
                    y={y - 10}
                    textAnchor="middle"
                    className="fill-slate-300 text-[11px]"
                  >
                    {(
                      item.value *
                      100
                    ).toFixed(2)}
                    %
                  </text>

                  <text
                    x={
                      x +
                      barWidth / 2
                    }
                    y={
                      paddingTop +
                      plotHeight +
                      25
                    }
                    textAnchor="middle"
                    className="fill-slate-500 text-[10px]"
                  >
                    {item.label}
                  </text>

                </g>
              );
            }
          )}

          <line
            x1={paddingLeft}
            y1={
              paddingTop +
              plotHeight
            }
            x2={
              width -
              paddingRight
            }
            y2={
              paddingTop +
              plotHeight
            }
            stroke="currentColor"
            className="text-white/20"
          />

        </svg>

      </div>

    </ChartShell>
  );
}

/* =========================================================
   CHART SHELL
========================================================= */

function ChartShell({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#07111f] p-5">

      <h3 className="text-lg font-semibold text-slate-200">
        {title}
      </h3>

      <p className="mt-2 text-sm text-slate-500">
        {description}
      </p>

      <div className="mt-5">
        {children}
      </div>

    </div>
  );
}

/* =========================================================
   PERFORMANCE TABLE
========================================================= */

function PerformanceTable({
  data,
}: {
  data: VisualizationData["performance"];
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#07111f] p-5">

      <h3 className="text-lg font-semibold text-slate-200">
        Performance Comparison
      </h3>

      <p className="mt-2 text-sm text-slate-500">
        Training, inference, recovery, and throughput
        across stored experiments.
      </p>

      {data.length === 0 ? (
        <div className="mt-6 rounded-xl border border-dashed border-white/10 p-8 text-center text-sm text-slate-500">
          No performance data available.
        </div>
      ) : (
        <div className="mt-5 overflow-x-auto">

          <table className="w-full min-w-[1000px] border-collapse">

            <thead>

              <tr className="border-b border-white/10 text-left">

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Experiment
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Decoder
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Rounds
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Physical Noise
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Measurement Noise
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Logical
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Physical
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Bit
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Training
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Inference
                </th>

                <th className="px-4 py-3 text-xs uppercase tracking-wide text-slate-500">
                  Throughput
                </th>

              </tr>

            </thead>

            <tbody>

              {data.map((item) => (

                <tr
                  key={item.experiment_id}
                  className="border-b border-white/5 hover:bg-white/[0.03]"
                >

                  <td className="px-4 py-4 font-mono text-xs text-slate-300">
                    {item.experiment_id}
                  </td>

                  <td className="px-4 py-4 text-xs text-slate-300">
                    {item.decoder ?? "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {item.rounds ?? "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.physical_noise ===
                    "number"
                      ? item.physical_noise.toFixed(
                          2
                        )
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.measurement_noise ===
                    "number"
                      ? item.measurement_noise.toFixed(
                          2
                        )
                      : "--"}
                  </td>

                  <td className="px-4 py-4 font-semibold text-cyan-300">
                    {typeof item.logical_success ===
                    "number"
                      ? `${(
                          item.logical_success *
                          100
                        ).toFixed(2)}%`
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.physical_recovery ===
                    "number"
                      ? `${(
                          item.physical_recovery *
                          100
                        ).toFixed(2)}%`
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.bit_accuracy ===
                    "number"
                      ? `${(
                          item.bit_accuracy *
                          100
                        ).toFixed(2)}%`
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.training_seconds ===
                    "number"
                      ? `${item.training_seconds.toFixed(
                          4
                        )} s`
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.inference_seconds ===
                    "number"
                      ? `${item.inference_seconds.toFixed(
                          4
                        )} s`
                      : "--"}
                  </td>

                  <td className="px-4 py-4 text-sm text-slate-300">
                    {typeof item.samples_per_second ===
                    "number"
                      ? item.samples_per_second.toFixed(
                          2
                        )
                      : "--"}
                  </td>

                </tr>

              ))}

            </tbody>

          </table>

        </div>
      )}

    </div>
  );
}