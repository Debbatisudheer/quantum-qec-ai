# Quantum QEC AI — Complete Project Guide

## 1. What is this project?

This project is an **AI-powered Quantum Error Correction (QEC) research platform**.

The important point is:

> The project studies quantum error correction using a classical computer, a quantum simulator, and machine-learning decoders.

It does **not** use a physical quantum computer.

The project currently focuses on a **3-qubit bit-flip repetition code**.

The main idea is:

```text
Create quantum state
        |
        v
Introduce errors
        |
        v
Observe syndrome information
        |
        v
Convert observations into ML features
        |
        v
AI decoder predicts a correction
        |
        v
Apply correction
        |
        v
Recover logical information
        |
        v
Measure the result
        |
        v
Compare AI with a baseline
```

---

# 2. Beginner explanation

Imagine we want to protect an important bit of information.

Instead of keeping one physical qubit, the current project uses three:

```text
Logical 0 -> 000
Logical 1 -> 111
```

If one physical qubit changes:

```text
000 -> 010
```

the system can use QEC information to understand that something went wrong.

The system does not simply ask:

> "What was the hidden error?"

Instead, the AI is intended to work from the information that a decoder would observe:

```text
Syndrome history
+
Detection-event history
```

The AI then predicts a useful correction.

---

# 3. What makes the project interesting?

The project combines several areas:

```text
Quantum Computing
       +
Quantum Error Correction
       +
Classical Simulation
       +
Machine Learning
       +
Statistics
       +
FastAPI
       +
Next.js
```

So this is not only a quantum-circuit project.

It is a complete research/application pipeline.

---

# 4. Main research question

The project asks:

> Can machine learning learn from QEC observations and predict useful corrections that improve logical quantum-information recovery?

The primary system-level metric is **logical success**.

That is important because:

```text
Exact physical-error prediction
```

and

```text
Logical information recovery
```

are not always the same thing.

---

# 5. Current project scope

The source code implements or explores:

- 3-qubit bit-flip repetition code
- repeated QEC rounds
- physical bit-flip noise
- syndrome/readout noise
- syndrome history
- detection events
- dataset generation
- feature engineering
- logical-aware target generation
- Random Forest decoding
- Logistic Regression decoder
- MLP decoder
- lookup-style decoders
- temporal/GRU decoder experiments
- time-varying experiments
- stochastic experiments
- Qiskit/Aer quantum-simulation integration
- correction
- logical recovery
- baseline comparison
- scientific evaluation
- bootstrap statistics
- paired permutation testing
- result storage
- FastAPI backend
- Next.js frontend
- single-sample trace visualization
- scientific charts
- automated tests

The **primary unified experiment path** currently uses:

```text
LogicalTargetRandomForestDecoder
```

---

# 6. Complete architecture

```text
                         USER
                          |
                          v
                  Next.js Dashboard
                          |
                          | REST/JSON
                          v
                    FastAPI Backend
                          |
                          v
                  Experiment Engine
                          |
             +------------+-------------+
             |                          |
             v                          v
       Dataset Generator          Result Storage
             |                          |
             v                          v
       QEC observations          JSON experiment files
             |
             v
       Feature Engineering
             |
             v
     Logical Target Builder
             |
             v
   Logical-Target Random Forest
             |
             v
      Predicted Correction
             |
             v
       Logical Recovery
             |
             v
          Metrics
             |
             v
       Scientific Analysis
```

There is also a more explicit quantum-simulation path:

```text
Training Data
     |
     v
AI Decoder
     |
     v
Qiskit/Aer repeated-QEC circuit
     |
     v
Syndrome measurements
     |
     v
Detection events
     |
     v
AI prediction
     |
     v
Quantum correction circuit
     |
     v
Final quantum measurement
     |
     v
Logical recovery
```

---

# 7. Important distinction: two related execution paths

The project contains two important forms of experimentation.

## Path A — Unified ML experiment

Used by `ExperimentEngine`.

```text
TimeVaryingQECDatasetGenerator
        |
        v
training samples
        |
        v
Random Forest training
        |
        v
independent test samples
        |
        v
AI evaluation
        |
        v
ExperimentResult
```

This path is fast and is the main API experiment path.

## Path B — Qiskit/Aer quantum integration

Used by the quantum integration and trace services.

```text
Qiskit/Aer
   |
   v
Repeated QEC circuit
   |
   v
Quantum syndrome measurement
   |
   v
Observation processing
   |
   v
AI decoder
   |
   v
Quantum correction
   |
   v
Final quantum measurement
```

This path connects the ML decoder to an actual simulated quantum circuit.

---

# 8. Main project folders

```text
quantum-qec-ai/
|
+-- quantum/
+-- qec/
+-- noise/
+-- syndrome/
+-- dataset/
+-- decoders/
+-- correction/
+-- evaluation/
+-- experiments/
+-- backend/
+-- frontend/
+-- scripts/
+-- tests/
|
+-- main.py
+-- requirements.txt
+-- README.md
+-- existing_pipeline.txt
```

---

# 9. What each folder means

| Folder | Purpose |
|---|---|
| `quantum/` | Qiskit circuits, simulator, repeated QEC, state evaluation |
| `qec/` | QEC code definitions |
| `noise/` | Physical and stochastic noise models |
| `syndrome/` | Syndrome extraction and measurement noise |
| `dataset/` | Dataset generation, validation, splitting, storage |
| `decoders/` | Lookup and ML decoders |
| `correction/` | Applying predicted corrections |
| `evaluation/` | Logical recovery and metrics |
| `experiments/` | Experiment orchestration and scientific studies |
| `backend/` | FastAPI API |
| `frontend/` | Next.js dashboard |
| `scripts/` | Research and analysis scripts |
| `tests/` | Automated tests |

---

# 10. Primary model

The main decoder is:

```text
LogicalTargetRandomForestDecoder
```

Internally it uses:

```text
RandomForestClassifier
+
MultiOutputClassifier
```

The output contains three correction bits:

```text
[q0, q1, q2]
```

For example:

```text
010
```

means the predicted correction acts on the middle qubit.

---

# 11. Why the target is logical-aware

The project does something important during training.

It does not simply say:

```text
hidden error -> exact error label
```

Instead, for each observed syndrome history it examines possible error states and candidate corrections.

Conceptually:

```text
Observed syndrome history
          |
          v
Possible error states
          |
          v
Try candidate corrections
          |
          v
Check logical recovery
          |
          v
Choose a useful correction
```

The training target is therefore based on **logical preservation**.

Ground-truth error information is used to build the training target, but it is not supposed to be passed as an input feature to the decoder.

---

# 12. Main feature representation

For each QEC round the Random Forest representation is:

```text
syndrome_1
syndrome_2
detection_1
detection_2
```

With five rounds:

```text
5 x 4 = 20 features
```

Conceptually:

```text
Round 1 -> s1 s2 d1 d2
Round 2 -> s1 s2 d1 d2
Round 3 -> s1 s2 d1 d2
Round 4 -> s1 s2 d1 d2
Round 5 -> s1 s2 d1 d2
```

---

# 13. Detection events

A detection event describes a change in the observed syndrome.

Conceptually:

```text
Detection[t] =
    Syndrome[t-1] XOR Syndrome[t]
```

The first round compares against:

```text
00
```

Example:

```text
Previous = 00
Current  = 10

Detection = 10
```

---

# 14. Main experiment flow

The primary API experiment is:

```text
POST /simulate
       |
       v
Validate configuration
       |
       v
Create ExperimentConfig
       |
       v
Generate training samples
       |
       v
Generate independent test samples
       |
       v
Create Random Forest decoder
       |
       v
Build logical-aware targets
       |
       v
Train decoder
       |
       v
Decode test samples
       |
       v
Calculate metrics
       |
       v
Save JSON result
       |
       v
Return response
```

---

# 15. Current defaults

The FastAPI simulation request defaults include:

```text
QEC code                  = bit_flip_3
Qubits                    = 3
Logical state             = 0
Rounds                    = 5
Physical noise            = 0.01
Measurement noise         = 0.10
Training samples          = 5000
Test samples              = 1000
Random Forest estimators  = 100
Seed                      = 42
```

The general `ExperimentConfig` class uses a physical-noise default of `0.10`.

The frontend's current default configuration uses:

```text
physical noise = 0.10
measurement noise = 0.10
training = 5000
test = 1000
rounds = 5
```

---

# 16. Metrics

The system records:

```text
Exact accuracy
Physical recovery
Bit accuracy
Logical accuracy
Training time
Inference time
Samples per second
```

The most important system-level metric is:

```text
Logical accuracy / logical success
```

because QEC exists to preserve logical information.

---

# 17. Baseline comparison

The project contains paired AI-vs-baseline experiments.

The baseline and AI are evaluated on the same held-out samples.

The scientific layer calculates:

- baseline logical success
- AI logical success
- absolute gain
- relative gain
- baseline logical error
- AI logical error
- logical error reduction
- bootstrap confidence interval
- permutation p-value

---

# 18. Reported scientific result

The uploaded project contains a formal scientific report.

For the reported configuration:

```text
Rounds:                 5
Training samples:       2000
Test samples/seed:      2000
Random Forest trees:    100
Seeds:                  42, 43, 44
Measurement noise:      0.0
Physical noise levels:  0.01, 0.03, 0.05,
                        0.10, 0.15, 0.20
```

The report gives:

| Physical noise | Baseline | AI-QEC | Absolute gain |
|---:|---:|---:|---:|
| 0.01 | 99.28% | 99.37% | +0.08 pp |
| 0.03 | 95.45% | 97.60% | +2.15 pp |
| 0.05 | 89.27% | 94.82% | +5.55 pp |
| 0.10 | 73.42% | 84.17% | +10.75 pp |
| 0.15 | 62.75% | 71.97% | +9.22 pp |
| 0.20 | 55.08% | 59.92% | +4.83 pp |

These are results from the project's tested configuration only.

They are not a claim about all QEC codes, all noise models, or physical quantum hardware.

---

# 19. Statistical design

The formal paired study uses:

```text
3 seeds
42
43
44
```

and:

```text
2000 test samples per seed
```

The project also uses:

```text
10,000 bootstrap iterations
```

and a paired permutation procedure.

The report itself correctly warns that three independent seeds limit the strength of broad generalization.

---

# 20. FastAPI

The backend is:

```text
FastAPI
+
Pydantic
+
Uvicorn
```

Main entry point:

```text
backend/main.py
```

The application exposes a health endpoint:

```text
GET /health
```

and includes the API router.

---

# 21. Main API endpoints

### Run experiment

```text
POST /simulate
```

### Generate a single trace

```text
POST /simulate/trace
```

### List experiments

```text
GET /experiments
```

### Best experiment

```text
GET /experiments/best
```

### Worst experiment

```text
GET /experiments/worst
```

### Compare experiments

```text
GET /experiments/compare
```

### Load one stored result

```text
GET /results/{experiment_id}
```

### Experiment summary

```text
GET /experiments/summary
```

### Experiment analysis

```text
GET /experiments/analysis
```

### Visualization data

```text
GET /experiments/visualization
```

### Scientific evaluation

```text
GET /scientific/evaluation
```

### Scientific rows

```text
GET /scientific/results
```

### Scientific summary

```text
GET /scientific/summary
```

---

# 22. Frontend

The frontend is:

```text
Next.js 16.3.4
React 19.2.8
TypeScript
Tailwind CSS 4
```

The main page is:

```text
frontend/app/page.tsx
```

It contains the dashboard logic.

The UI can:

- configure experiments
- run experiments
- load stored experiments
- filter history
- sort experiment results
- display metrics
- generate a single trace
- display scientific evaluation
- display charts
- display baseline-vs-AI information

---

# 23. Frontend/backend communication

The frontend reads:

```text
NEXT_PUBLIC_API_BASE_URL
```

from `.env.local`.

The uploaded project currently contains a value pointing to:

```text
https://quantum-qec-ai-4.onrender.com
```

The code also has a local fallback:

```text
http://127.0.0.1:8001
```

The source therefore supports the pattern:

```text
Next.js frontend
       |
       | HTTP/JSON
       v
FastAPI backend
```

---

# 24. Result storage

Experiment results are stored as JSON files under:

```text
experiments/results/
```

The storage class is:

```text
ExperimentResultStorage
```

It supports:

```text
save
load
exists
list
load_all
delete
count
```

There is no relational database in the uploaded implementation.

The persistent experiment store is file-based JSON.

---

# 25. Scientific results storage

The project also contains:

```text
experiments/paired_results/
```

and:

```text
experiments/scientific_evaluation/
```

The scientific evaluation layer reads:

```text
experiments/paired_results/
paired_baseline_ai_results.json
```

and generates reusable scientific outputs.

---

# 26. Testing

The project contains a large test suite.

There are:

```text
88 test_*.py files
```

and approximately:

```text
98 test functions/methods
```

by static inspection of the uploaded source tree.

Tests cover areas such as:

- QEC
- quantum simulation
- dataset generation
- dataset validation
- noise
- decoders
- logical recovery
- experiment engine
- result storage
- backend endpoints
- scientific analysis
- time-varying decoding
- repeated QEC
- AI-vs-baseline behavior

---

# 27. Verification performed during this analysis

The uploaded Python source was statically compiled.

Result:

```text
Python source files checked: 176
Compile errors: 0
```

The frontend production build could not be executed in this analysis container because the uploaded `node_modules` contains Windows-installed executables and the Linux environment reports:

```text
next: Permission denied
```

The project itself includes a Windows-oriented development environment.

Qiskit and Qiskit Aer were also not installed in the analysis container, so the quantum runtime tests were not executed here.

This is different from saying the project tests are broken. It means this analysis environment did not contain the project's required runtime setup.

---

# 28. Very important project boundary

The project is:

```text
AI + QEC + Quantum Simulation
```

It is not:

```text
Physical quantum hardware
```

It is not:

```text
Universal QEC
```

It is not:

```text
Surface-code QEC
```

It is not:

```text
Production hardware decoder
```

The safest description is:

> A classical simulation and research platform for studying AI-based quantum-error decoding and logical recovery.

---

# 29. Beginner mental model

Remember these five things:

```text
Quantum simulation
    = creates the quantum experiment

QEC
    = creates/observes error information

Dataset
    = turns observations into ML examples

AI decoder
    = predicts a useful correction

Logical recovery
    = checks whether the original information survived
```

---

# 30. One-minute explanation

> "This project is an AI-powered quantum error-correction research platform. It uses Qiskit and Aer on a classical computer to simulate a 3-qubit bit-flip repetition code. Physical and measurement noise create a decoding problem. The system records syndrome history and detection events, converts them into machine-learning features, and trains a logical-target Random Forest decoder to predict a useful correction. The correction is applied and the final logical state is checked. The project also contains Qiskit-based quantum integration, a FastAPI backend, a Next.js dashboard, experiment storage, scientific evaluation, statistical testing, and a large automated test suite."

---

# 31. Final architecture to remember

```text
              QUANTUM / QEC SIDE
                      |
                      v
              Noise + Syndrome
                      |
                      v
              Observed History
                      |
                      v
             Feature Engineering
                      |
                      v
              Logical Target
                      |
                      v
               AI Decoder
                      |
                      v
             Predicted Correction
                      |
                      v
              Quantum/Classical
                Correction
                      |
                      v
               Logical Recovery
                      |
                      v
                 Evaluation
                      |
              +-------+-------+
              |               |
              v               v
           Baseline         AI-QEC
              \               /
               \             /
                v           v
                Paired Study
                     |
                     v
             Scientific Results
                     |
               +-----+-----+
               |           |
               v           v
            FastAPI      Next.js
```
