# Quantum QEC AI — Architecture and Data Flow

## 1. Big picture

The project has three major sides:

```text
Quantum/QEC Research
        |
        v
Experiment + AI Layer
        |
        v
Application Layer
```

More specifically:

```text
Qiskit/Aer
   |
   v
QEC + Noise + Syndrome
   |
   v
Dataset
   |
   v
AI Decoder
   |
   v
Evaluation
   |
   v
FastAPI
   |
   v
Next.js
```

---

# 2. Layer-by-layer architecture

## Layer 1 — Quantum simulation

Main folder:

```text
quantum/
```

Important files:

```text
quantum/circuit.py
quantum/simulator.py
quantum/repeated_qec.py
quantum/repeated_measurement.py
quantum/state_evaluator.py
```

Responsibilities:

- create quantum circuits
- run Qiskit Aer simulation
- construct repeated QEC circuits
- parse repeated measurements
- evaluate logical recovery

---

# 3. Layer 2 — QEC

Main folder:

```text
qec/
```

Current code:

```text
qec/bit_flip_3.py
```

The current code is the 3-qubit repetition code.

Logical encoding:

```text
|0>L = |000>
|1>L = |111>
```

---

# 4. Layer 3 — Noise

Main folder:

```text
noise/
```

The source includes:

```text
bit_flip.py
quantum_bit_flip.py
stochastic_repeated_noise.py
```

The project therefore contains both simple and more experimental noise models.

---

# 5. Layer 4 — Syndrome

Main folder:

```text
syndrome/
```

Important responsibilities:

```text
SyndromeExtractor
SyndromeMeasurementNoise
```

Conceptually:

```text
Physical error
      |
      v
Perfect syndrome
      |
      v
Measurement/readout noise
      |
      v
Observed syndrome
```

---

# 6. Layer 5 — Dataset

Main folder:

```text
dataset/
```

The project contains:

- standard dataset generation
- noisy dataset generation
- repeated-syndrome generation
- time-varying dataset generation
- preprocessing
- splitting
- validation
- analysis
- storage

The primary modern experiment path uses:

```text
TimeVaryingQECDatasetGenerator
```

---

# 7. Time-varying sample

A generated sample contains information such as:

```text
sample_id
qec_code
num_qubits
rounds
logical_state
encoded_state
physical_error_probability
measurement_noise_probability
syndrome_history
observed_syndrome_history
detection_events
physical_error_history
final_error_state
final_syndrome
final_observed_syndrome
final_error_description
```

The decoder should use observable information such as:

```text
observed_syndrome_history
detection_events
```

Ground truth such as:

```text
physical_error_history
final_error_state
perfect syndrome
```

is used for target construction and evaluation, not as normal inference input.

---

# 8. Layer 6 — Decoders

Main folder:

```text
decoders/
```

The repository contains several decoder families.

### Lookup

```text
lookup.py
repeated_lookup.py
```

### Classical ML

```text
ml_logistic.py
ml_random_forest.py
ml_mlp.py
```

### Logical-target models

```text
logical_target.py
logical_target_random_forest.py
logical_target_gru.py
```

### Temporal models

```text
temporal_gru.py
temporal_gru_classifier.py
time_varying_ml.py
```

The primary unified API experiment currently restricts the selected decoder to:

```text
logical_target_random_forest
```

Other decoders are present mainly for comparison and research experiments.

---

# 9. Layer 7 — Correction

Folder:

```text
correction/
```

Important classes:

```text
CorrectionEngine
TimeVaryingCorrectionEngine
```

The basic operation is:

```text
Corrupted state
      XOR
Predicted correction
      |
      v
Corrected state
```

---

# 10. Layer 8 — Logical recovery

File:

```text
evaluation/logical_recovery.py
```

The current logical recovery uses majority voting.

For logical 0:

```text
000 -> 0
001 -> 0
010 -> 0
100 -> 0
```

For logical 1:

```text
111 -> 1
110 -> 1
101 -> 1
011 -> 1
```

The important idea is:

```text
Physical state
      |
      v
Majority vote
      |
      v
Recovered logical state
```

---

# 11. Layer 9 — Evaluation

The project calculates several metrics.

```text
Exact error match
Physical recovery
Bit accuracy
Logical success
Training time
Inference time
Throughput
```

The main research metric is logical success.

---

# 12. Layer 10 — Experiment engine

File:

```text
experiments/engine.py
```

Class:

```text
ExperimentEngine
```

Its job is orchestration.

It does not contain the detailed algorithms.

It coordinates:

```text
Configuration
   |
   v
Dataset generation
   |
   v
Decoder creation
   |
   v
Training
   |
   v
Evaluation
   |
   v
Result
   |
   v
Storage
```

---

# 13. Experiment engine design

This separation is useful:

```text
ExperimentEngine
    = coordinator

Dataset generator
    = creates data

Decoder
    = learns/decodes

Evaluator
    = measures result

Storage
    = saves result
```

The engine does not try to implement all responsibilities itself.

---

# 14. Layer 11 — Result storage

File:

```text
experiments/result_storage.py
```

Class:

```text
ExperimentResultStorage
```

Storage is JSON-based.

Directory:

```text
experiments/results/
```

A result contains:

```text
experiment_id
config
training_samples
test_samples
logical_targets_learned
average_target_score
exact_accuracy
physical_accuracy
bit_accuracy
logical_accuracy
training_seconds
inference_seconds
samples_per_second
decoder_type
```

---

# 15. Layer 12 — Analysis

Important classes:

```text
ExperimentAnalysis
ExperimentResultQuery
ExperimentVisualization
ExperimentReportGenerator
```

They provide:

```text
filtering
sorting
comparison
summary
analysis
visualization
reporting
```

---

# 16. Scientific evaluation layer

File:

```text
evaluation/scientific_evaluation.py
```

Important design point:

> The scientific evaluation class does not rerun the experiment.

It reads previously generated paired results.

Input:

```text
experiments/paired_results/
paired_baseline_ai_results.json
```

Then it calculates reusable scientific summaries.

---

# 17. Scientific flow

```text
Paired experiment results
          |
          v
ScientificEvaluation
          |
          +--> baseline success
          |
          +--> AI success
          |
          +--> absolute gain
          |
          +--> relative gain
          |
          +--> error reduction
          |
          +--> bootstrap CI
          |
          +--> permutation p-value
          |
          v
Scientific report
```

---

# 18. FastAPI architecture

```text
backend/main.py
       |
       v
backend/api/routes.py
       |
       +--------------------+
       |                    |
       v                    v
Experiment engine      Scientific evaluator
       |
       v
Core research modules
```

Pydantic schemas live in:

```text
backend/api/schemas.py
```

---

# 19. API data flow: run experiment

```text
Browser
  |
  | POST /simulate
  v
FastAPI
  |
  v
SimulationRequest
  |
  v
ExperimentConfig
  |
  v
ExperimentEngine
  |
  v
Result
  |
  v
JSON storage
  |
  v
SimulationResponse
  |
  v
Browser
```

---

# 20. API data flow: trace

```text
Browser
  |
  | POST /simulate/trace
  v
FastAPI
  |
  v
SimulationTraceService
  |
  +--> train RF
  |
  +--> generate one sample
  |
  +--> Qiskit/Aer integration
  |
  +--> predict correction
  |
  +--> quantum correction
  |
  +--> logical recovery
  |
  v
Trace response
  |
  v
Browser
```

---

# 21. Frontend architecture

Main page:

```text
frontend/app/page.tsx
```

Reusable components:

```text
frontend/component/SiteHeader.tsx
frontend/component/SiteFooter.tsx
```

The page contains:

- experiment configuration
- experiment execution
- result history
- filtering
- sorting
- charts
- scientific evaluation
- trace display

---

# 22. Frontend communication

The frontend calls the backend through normal HTTP.

Example:

```text
fetch(
    `${API_BASE_URL}/simulate`
)
```

There is no WebSocket implementation visible in the uploaded frontend.

The current application communication is REST/HTTP + JSON.

---

# 23. Complete application flow

```text
USER
 |
 v
NEXT.JS
 |
 | HTTP/JSON
 v
FASTAPI
 |
 v
EXPERIMENT ENGINE
 |
 +--> DATASET
 |
 +--> AI DECODER
 |
 +--> EVALUATOR
 |
 +--> JSON STORAGE
 |
 v
RESPONSE
 |
 v
NEXT.JS
 |
 v
CHARTS / METRICS / TRACE
```

---

# 24. Important architecture principle

The project separates:

```text
Simulation
AI
Evaluation
Storage
API
UI
```

That makes it easier to change one layer without rewriting everything.

For example:

```text
Random Forest
      |
      v
could be replaced in research
      |
      v
GRU / another decoder
```

while keeping:

```text
API
Evaluation
Storage
UI
```

conceptually separate.

---

# 25. Architecture in one sentence

> The project is a layered quantum-simulation and AI research platform where QEC observations become machine-learning features, the decoder predicts a logical-useful correction, the result is evaluated scientifically, and FastAPI/Next.js expose the research workflow as an application.
