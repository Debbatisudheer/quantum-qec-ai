AI-Powered Quantum Error Correction System

An AI-focused quantum error-correction research platform built entirely with classical quantum simulation.

The project investigates whether a machine-learning decoder can learn from quantum error-correction observations and predict useful corrections that improve logical quantum-information recovery.

Important: This project currently uses Qiskit/Aer simulation on a classical computer. It does not use physical quantum hardware.

1. Project Goal

The central AI research question is:

Can machine learning learn to decode syndrome information and choose corrections that improve logical quantum-error recovery?

The high-level flow is:

Quantum Simulation
       ↓
Quantum Noise
       ↓
Syndrome Generation
       ↓
Syndrome + Detection-Event Features
       ↓
AI Decoder
       ↓
Predicted Correction
       ↓
Apply Correction
       ↓
Logical Recovery
       ↓
Evaluation
       ↓
AI vs Baseline

The current primary AI model is a:

Logical-target Random Forest decoder

2. What the AI Does

The AI does not directly receive the hidden physical error.

Instead, it receives information derived from the QEC measurement process:

Observed Syndrome History
+
Detection-Event History
        ↓
Feature Vector
        ↓
Random Forest
        ↓
3-bit Predicted Correction

The predicted correction is then applied to the simulated quantum state.

The final question is:

Did the correction preserve the original logical information?

3. Current Project Scope

The current implementation focuses on:

3-qubit bit-flip repetition code

Repeated QEC rounds

Simulated physical bit-flip noise

Simulated syndrome/measurement noise

Syndrome history

Detection events

Supervised machine learning

Logical-target correction learning

Random Forest decoding

Baseline comparison

Paired statistical evaluation

Scientific experiment sweeps

FastAPI backend

Next.js frontend

Live QEC/AI trace

Experiment result storage

Scientific evaluation and visualization

4. Important Scope Boundary

Do not describe the current project as:

a physical quantum computer

hardware QEC

a universal quantum decoder

a surface-code implementation

a production hardware decoder

The accurate description is:

A classical simulation platform for researching AI-based quantum-error decoding and logical recovery.

5. Core AI Architecture

                    DATA GENERATION
                          │
                          ▼
                  QEC SIMULATION DATA
                          │
                          ▼
                 SYNDROME OBSERVATION
                          │
                          ▼
                  DETECTION EVENTS
                          │
                          ▼
                 FEATURE ENGINEERING
                          │
                          ▼
                  TARGET ENGINEERING
                          │
                          ▼
                   TRAINING DATA
                          │
                          ▼
                RANDOM FOREST TRAINING
                          │
                          ▼
                   TRAINED DECODER
                          │
                          ▼
                    NEW SAMPLE
                          │
                          ▼
                  FEATURE ENCODING
                          │
                          ▼
                    AI INFERENCE
                          │
                          ▼
                PREDICTED CORRECTION
                          │
                          ▼
                  CORRECTION ENGINE
                          │
                          ▼
                  LOGICAL RECOVERY
                          │
                          ▼
                     EVALUATION
                          │
                          ▼
                  BASELINE COMPARISON
                          │
                          ▼
                 SCIENTIFIC ANALYSIS

6. Quantum/QEC Layer — Only What AI Needs

The AI needs QEC information as input.

The current code uses a 3-qubit repetition code.

Logical states are encoded as:

Logical 0 → 000
Logical 1 → 111

Physical errors can change individual qubits.

The QEC system measures stabilizer information and produces a syndrome.

For the current 3-qubit code:

Syndrome
00 → no detected single-qubit error
01 → one error pattern
10 → another error pattern
11 → another error pattern

The exact syndrome mapping is implemented in the QEC layer.

The AI does not need to know the hidden error directly.

7. Repeated QEC

The project can perform multiple QEC rounds.

Default:

rounds = 5

Conceptually:

Round 1 → syndrome
Round 2 → syndrome
Round 3 → syndrome
Round 4 → syndrome
Round 5 → syndrome

The AI uses this history instead of relying only on one final observation.

8. Syndrome History

Example:

00
10
10
01
00

This is the syndrome history across five rounds.

The history provides temporal information about how the observed error signal changes.

9. Detection Events

A detection event describes a change between consecutive syndrome observations.

Formula:

DetectionEvent[t]
=
Syndrome[t-1] XOR Syndrome[t]

Example:

Previous syndrome = 00
Current syndrome  = 10

Detection event = 10

The AI therefore receives two related kinds of information:

Syndrome history
+
Changes in syndrome history

10. Measurement Noise

The project can corrupt syndrome observations with measurement noise.

Therefore:

Perfect Syndrome
       ↓
Measurement Noise
       ↓
Observed Syndrome

The AI normally works with the observed information, not the hidden perfect syndrome.

This makes the decoding problem harder and more realistic.

11. AI Dataset Pipeline

Every generated sample follows approximately:

Choose logical state
       ↓
Encode logical state
       ↓
Generate physical error history
       ↓
Run QEC rounds
       ↓
Generate syndrome history
       ↓
Apply measurement noise
       ↓
Create observed syndrome history
       ↓
Calculate detection events
       ↓
Encode features
       ↓
Create logical-aware target
       ↓
Store sample

12. Training Data

A training dataset contains many examples of:

Input:
syndrome/detection-event features

Target:
useful correction

Example conceptually:

Features → [0,1,0,0,1,...]
Target   → [0,1,0]

The actual feature length depends on the number of QEC rounds.

13. Test Data

Test samples are held out from training.

The structure is:

Training samples
      ↓
AI learns

Held-out test samples
      ↓
AI is evaluated

The test set must not be used to train the model.

This protects the validity of the evaluation.

14. Why Synthetic Data Is Used

The project is simulation-based.

The simulator knows the ground truth for each generated sample.

That allows us to create supervised-learning targets and evaluate whether the AI's predicted correction actually succeeds.

15. Random Seeds

Experiments use controlled random seeds.

Examples:

42
43
44

This improves reproducibility and allows repeated experiments under different random realizations.

16. Feature Engineering

The feature-engineering stage converts QEC observations into numerical machine-learning input.

Conceptually:

Observed Syndrome History
+
Detection Events
        ↓
Feature Encoder
        ↓
Numerical Feature Vector

The current logical-target Random Forest uses a per-round representation conceptually like:

Round 1:
s1, s2, d1, d2

Round 2:
s1, s2, d1, d2

...

Round N:
s1, s2, d1, d2

17. Feature Interface Adapter

One integration path represents features as:

all syndrome bits
+
all detection-event bits

The logical-target Random Forest expects:

s1, s2, d1, d2

for each round.

Therefore an adapter converts between these representations.

Integration Features
        ↓
Feature Adapter
        ↓
RF Decoder Features

The adapter exists so the existing decoder can be reused without changing its core logic.

18. The AI Target

The AI predicts a three-bit correction.

Example:

010

The three bits correspond to correction actions on the three physical qubits.

19. Why the Target Is Logical-Aware

A major design decision in this project is that the ultimate objective is logical recovery, not simply exact identification of the physical error.

Conceptually:

Observation
    ↓
Candidate corrections
    ↓
Test logical outcome
    ↓
Choose useful correction
    ↓
Training target

This is called the logical-target approach.

20. Exact Error vs Logical Success

These are different metrics.

Exact error match

Actual error == Predicted correction

Logical success

After applying predicted correction,
the original logical information is recovered.

An AI prediction can therefore:

fail exact matching

but still:

succeed logically

This is important when interpreting results.

21. Main AI Model

The primary decoder is:

LogicalTargetRandomForestDecoder

It is based on:

RandomForestClassifier

with:

MultiOutputClassifier

because the decoder predicts multiple correction bits.

22. Random Forest

A Random Forest combines many decision trees.

Current default:

n_estimators = 100

Conceptually:

Features
   │
   ├── Tree 1
   ├── Tree 2
   ├── Tree 3
   ├── ...
   └── Tree 100
          ↓
   Combined prediction
          ↓
   Correction bits

23. Why Random Forest

Random Forest was selected as the primary decoder because it works well with structured/tabular features and provides a relatively simple, fast, interpretable machine-learning baseline.

It is not claimed to be universally optimal.

24. Multi-Output Classification

The correction contains three bits.

Example:

010

Conceptually:

Output bit 0 → 0
Output bit 1 → 1
Output bit 2 → 0

These predictions form the final correction vector.

25. AI Training

Training follows:

Training samples
       ↓
Feature matrix X
       +
Target matrix y
       ↓
Random Forest
       ↓
fit()
       ↓
Trained decoder

The learned mapping is approximately:

syndrome-related observations
              ↓
        useful correction

26. AI Inference

When a new sample arrives:

New syndrome observations
        ↓
Detection events
        ↓
Feature encoding
        ↓
Trained Random Forest
        ↓
Predicted correction

Example:

Prediction = 010

27. Decoder Interface

The decoder supports operations conceptually including:

train(samples)
predict(X)
predict_proba(X)
decode(sample)
decode_batch(samples)

This gives the AI subsystem a clean interface.

It also makes it possible to compare or replace models later.

28. AI Confidence

The Random Forest can provide probability information for its predictions.

The project can expose a confidence-like value in the trace.

Important:

Model probability/confidence is not a guarantee that the correction is correct.

29. AI → Correction

The AI produces the decision:

Predicted correction = 010

The correction engine then applies that decision to the simulated state.

Separation:

AI Decoder
    ↓
Decision

Correction Engine
    ↓
Action

30. Correction Mathematics

Let:

e = actual physical error
c = AI-predicted correction

Then:

Corrupted state
=
Encoded state XOR e

and:

Corrected state
=
Corrupted state XOR c

Therefore:

Corrected state
=
Encoded state XOR e XOR c

If:

e = c

the physical error is exactly cancelled.

But exact cancellation is stricter than logical recovery.

31. Logical Recovery

After correction, the system evaluates whether the original logical state is recovered.

For the current 3-qubit repetition code, majority information determines the recovered logical state.

The AI's ultimate purpose is:

Predicted correction
       ↓
Logical recovery
       ↓
Preserve logical information

32. AI Evaluation Metrics

The project evaluates several dimensions.

Exact Accuracy

Did AI predict the exact target?

Exact matches / total samples

Physical Recovery

Did AI exactly reverse the physical error?

Bit Accuracy

How many individual correction bits were correct?

Logical Accuracy

Did the correction preserve the logical state?

Logical successes / total samples

Logical Error Rate

1 - logical accuracy

Training Time

How long did the model take to train?

Inference Time

How long did prediction take?

Throughput

samples / second

33. Primary AI Metric

For this project, the most important outcome metric is:

Logical Success

because the system exists to preserve logical information.

The hierarchy is:

AI Prediction
      ↓
Correction
      ↓
Logical Recovery
      ↓
System Objective

34. Baseline

AI performance needs a reference point.

The project therefore compares:

Baseline
vs
AI-QEC

The baseline represents traditional/non-AI logical recovery behavior used for comparison.

35. Why Paired Comparison Was Used

The stronger comparison uses the same held-out test samples for both methods.

Same test sample
       │
   ┌───┴────┐
   ↓        ↓
Baseline    AI
   ↓        ↓
Result    Result

This makes the comparison more controlled.

36. AI Gain

Formula:

AI Gain =
AI Logical Success
-
Baseline Logical Success

Example:

AI       = 84.17%
Baseline = 73.42%

Gain = +10.75 percentage points

37. Relative Gain

Formula:

Relative Gain =
(AI - Baseline) / Baseline

This describes improvement relative to the baseline level.

38. Logical Error Reduction

First:

Baseline Error = 1 - Baseline Success
AI Error       = 1 - AI Success

Then:

Error Reduction =
(Baseline Error - AI Error)
/
Baseline Error

This measures how much of the baseline logical error was removed by AI.

39. Statistical Validation

The project uses:

Multiple random seeds
+
Paired test samples
+
Bootstrap confidence intervals
+
Permutation testing

The purpose is to determine whether the observed AI improvement is reasonably stable within the experiment.

40. Bootstrap

Bootstrap resampling:

Observed paired results
        ↓
Repeated resampling
        ↓
Calculate gain repeatedly
        ↓
Gain distribution
        ↓
Confidence interval

The project uses descriptive 95% bootstrap intervals.

41. Permutation Test

The paired differences are subjected to random sign changes to construct a null distribution.

The implementation uses:

10,000 sign randomizations

Therefore a displayed p-value around:

0.00010

is the resolution limit of this finite permutation procedure, not an infinitely precise probability.

42. Statistical Limitation

The main paired experiments use:

3 independent seeds

Therefore the results are useful experimental evidence for the tested setup, but they should not be presented as universal proof of AI superiority across all QEC systems.

43. AI Noise Sweep

The AI was tested at physical-noise levels:

0.00
0.01
0.03
0.05
0.10
0.15
0.20

Approximate AI logical success:

Physical Noise

AI Logical Success

0.00

100.00%

0.01

99.47%

0.03

97.60%

0.05

94.07%

0.10

83.40%

0.15

71.33%

0.20

59.40%

As noise increases, decoding becomes harder and logical success decreases.

44. AI + Measurement Noise

Combined noise was also tested.

Example:

Physical noise      = 0.10
Measurement noise   = 0.10

Observed mean AI logical success was approximately:

73.40%

This demonstrates that imperfect syndrome observations make the AI decoding problem harder.

45. Important Measurement-Noise Caveat

The measurement-only experiment:

Physical noise = 0
Measurement noise > 0

produced:

100% logical success

This should not be overinterpreted as strong proof of measurement-noise robustness.

Why?

Because there was no physical error to correct.

The more meaningful robustness test is combined physical + measurement noise.

46. Paired AI Results

The strongest paired comparison produced:

Physical Noise

Baseline

AI-QEC

Gain

0.01

99.28%

99.37%

+0.08 pp

0.03

95.45%

97.60%

+2.15 pp

0.05

89.27%

94.82%

+5.55 pp

0.10

73.42%

84.17%

+10.75 pp

0.15

62.75%

71.97%

+9.22 pp

0.20

55.08%

59.92%

+4.83 pp

The largest observed absolute gain was:

+10.75 percentage points

at:

10% physical noise

47. What the Results Mean

Within the tested simulation configuration:

AI-QEC > Baseline

at every tested nonzero physical-noise level in the paired study.

This suggests that the learned decoder can exploit syndrome-derived information to improve logical recovery relative to the tested baseline.

Do not generalize this result beyond the tested configuration without additional experiments.

48. AI Experiment Engine

The experiment engine automates:

Configuration
    ↓
Training data generation
    ↓
AI training
    ↓
Test data generation
    ↓
AI inference
    ↓
Evaluation
    ↓
Result storage

Important configuration values include:

QEC code
Number of qubits
Logical state
Rounds
Physical noise probability
Measurement noise probability
Training samples
Test samples
Decoder type
Random Forest estimators
Random seed

49. AI Result Storage

Experiment results are stored as JSON.

Stored information includes:

Experiment ID
Configuration
Training sample count
Test sample count
Target information
Accuracy metrics
Training time
Inference time
Throughput
Decoder type

This makes experiments reproducible and comparable later.

50. AI Result Analysis

Stored experiments can be:

Filtered
Sorted
Compared
Summarized

For example:

Find experiments at 10% physical noise
        ↓
Sort by logical accuracy
        ↓
Compare decoders

51. Scientific Evaluation Layer

The scientific evaluation layer converts paired experiment results into reusable scientific metrics.

It calculates:

Baseline success
AI success
Absolute gain
Relative gain
Baseline error
AI error
Error reduction
Bootstrap CI
Permutation p-value
Seed count
Test samples per seed

52. AI Visualization

Scientific plots include:

Logical success vs noise
Paired AI gain
Bootstrap confidence intervals
Logical error rate
Combined-noise heatmap

These help understand where the AI performs well and where performance degrades.

53. AI Live Trace

The application provides a single-sample AI trace.

Conceptually:

Sample
 ↓
Physical error
 ↓
Perfect syndrome
 ↓
Observed syndrome
 ↓
Detection events
 ↓
AI decoder
 ↓
Predicted correction
 ↓
Corrected state
 ↓
Logical recovery

This makes one AI decision explainable step-by-step.

54. Important Trace Interpretation

A trace may show:

Actual error      = 000
Predicted correction = 010

and:

Exact match = NO

while:

Logical success = YES

This is not necessarily a bug.

It demonstrates the distinction between:

Exact physical prediction

and:

Logical preservation

55. Backend AI Architecture

The backend exposes the AI/QEC system through FastAPI.

Main AI-related operations include:

POST /simulate
POST /simulate/trace

GET /experiments
GET /experiments/best
GET /experiments/worst
GET /experiments/compare

GET /results/{experiment_id}

GET /experiments/summary
GET /experiments/analysis
GET /experiments/visualization

GET /scientific/evaluation
GET /scientific/results
GET /scientific/summary

56. Frontend AI Architecture

The Next.js frontend displays:

Experiment configuration
        ↓
Simulation
        ↓
AI/QEC trace
        ↓
Predicted correction
        ↓
Logical recovery
        ↓
Scientific results
        ↓
Experiment analysis

The frontend communicates with the backend API.

Current development ports:

Backend  → 8001
Frontend → 3001

Ports may be changed if those ports are unavailable.

57. Project Structure

quantum-qec-ai/
│
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── schemas.py
│   │   └── routes.py
│   └── services/
│       └── trace_service.py
│
├── quantum/
│   ├── circuit.py
│   ├── simulator.py
│   ├── repeated_qec.py
│   ├── repeated_measurement.py
│   └── state_evaluator.py
│
├── qec/
│   └── bit_flip_3.py
│
├── noise/
│   ├── bit_flip.py
│   ├── quantum_bit_flip.py
│   └── stochastic_repeated_noise.py
│
├── syndrome/
│   └── ...
│
├── dataset/
│   └── ...
│
├── decoders/
│   └── ...
│
├── correction/
│   └── ...
│
├── evaluation/
│   ├── ...
│   └── scientific_evaluation.py
│
├── experiments/
│   ├── config.py
│   ├── engine.py
│   ├── result.py
│   ├── result_storage.py
│   ├── result_query.py
│   ├── analysis.py
│   ├── report.py
│   ├── export.py
│   ├── visualization.py
│   ├── quantum_ai_decoder_integration.py
│   └── results/
│
├── scripts/
│   ├── scientific_validation.py
│   ├── baseline_comparison.py
│   ├── ai_noise_sweep.py
│   ├── noise_robustness_matrix.py
│   ├── statistical_analysis.py
│   ├── paired_baseline_ai_comparison.py
│   └── scientific_plots.py
│
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   └── components/
│       ├── SiteHeader.tsx
│       └── SiteFooter.tsx
│
├── tests/
│
├── configs/
├── storage/
├── notebooks/
├── docs/
└── README.md

58. Important Files to Remember

AI decoder

Look inside:

decoders/

especially the logical-target Random Forest implementation.

AI + quantum integration

experiments/quantum_ai_decoder_integration.py

This connects the AI decoder with the real Qiskit simulation flow.

Experiment configuration

experiments/config.py

Experiment execution

experiments/engine.py

Scientific evaluation

evaluation/scientific_evaluation.py

AI trace API service

backend/services/trace_service.py

API routes

backend/api/routes.py

Frontend

frontend/app/page.tsx

Header

frontend/components/SiteHeader.tsx

Footer

frontend/components/SiteFooter.tsx

59. Environment Setup

Use the project virtual environment.

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Verify Python:

where.exe python

Expected:

...\quantum-qec-ai\.venv\Scripts\python.exe

Verify Qiskit:

python -c "import qiskit; print(qiskit.__file__)"

The Qiskit path should point into:

.venv\Lib\site-packages

60. Start Backend

Activate the virtual environment first.

Then:

python -m uvicorn backend.main:app --port 8001

Backend:

http://127.0.0.1:8001

Health check:

GET /health

Expected:

{
  "status": "ok",
  "service": "quantum-qec-ai"
}

61. Start Frontend

From:

frontend/

run the normal Next.js development command configured by the project.

Current development server was using:

http://localhost:3001

The frontend API base URL must point to the backend:

http://127.0.0.1:8001

62. Run Tests

From project root:

python -m pytest -q

Current full-suite status at the time this README was written:

99 passed
29 warnings

The warnings are pytest diagnostic warnings caused by some tests returning values rather than using assertions. They are warnings, not failed tests.

63. Run Scientific Validation

Use module execution from the project root:

python -m scripts.scientific_validation

Do not use:

python scripts\scientific_validation.py

if it causes Python import-path problems.

The module form keeps the project root available for imports.

64. Run Baseline Comparison

python -m scripts.baseline_comparison

This evaluates the non-AI baseline across physical-noise levels.

65. Run AI Noise Sweep

python -m scripts.ai_noise_sweep

This evaluates AI logical recovery across physical-noise levels.

66. Run Combined Noise Matrix

python -m scripts.noise_robustness_matrix

This evaluates combinations of:

Physical noise
+
Measurement noise

67. Run Statistical Analysis

python -m scripts.statistical_analysis

This analyzes:

Baseline
AI-QEC
Gain
Logical error
Error reduction
Confidence intervals
Combined-noise behavior

68. Run Paired Comparison

python -m scripts.paired_baseline_ai_comparison

This performs the controlled same-test-sample comparison between baseline and AI-QEC.

The paired results are stored under:

experiments/paired_results/

69. Generate Scientific Plots

python -m scripts.scientific_plots

Generated outputs include:

experiments/paired_results/plots/

with logical-success, gain, confidence-interval, logical-error, and heatmap visualizations.

70. Scientific Evaluation Outputs

The scientific evaluation can produce:

experiments/scientific_evaluation/
├── scientific_results.json
├── scientific_results.csv
└── scientific_report.txt

71. Important Development Problems Already Solved

Qiskit environment mismatch

Problem:

ModuleNotFoundError: No module named 'qiskit'

Cause:

Uvicorn reload was using the global Python installation instead of the virtual environment.

Solution:

.\.venv\Scripts\Activate.ps1

Then verify:

where.exe python

72. Port 8000 Conflict

Port 8000 was already occupied.

The backend was therefore moved to:

8001

Run:

python -m uvicorn backend.main:app --port 8001

73. Frontend Port

Port 3000 was already occupied, so Next.js used:

3001

This is independent from the backend port.

Frontend → 3001
Backend  → 8001

74. Frontend API Connection

The frontend initially attempted to call:

8000

while the backend was running on:

8001

The frontend API base URL was corrected to:

http://127.0.0.1:8001

75. Trace API Type Bug

The trace API initially returned:

actual_error

as a list.

The frontend schema expected a string.

The service was corrected to convert error bits to a string representation.

76. Trace Semantic Bug

The trace initially used a post-correction state as the corrupted state.

That was semantically incorrect.

The correct relationship is:

Encoded
   XOR
Actual error
   ↓
Corrupted

The service now calculates the corrupted state explicitly.

77. Decoder Feature Mismatch

The integration layer and Random Forest decoder used different feature layouts.

An adapter was added:

Integration features
       ↓
Adapter
       ↓
RF feature format

This avoided rewriting the decoder.

78. Important AI Interpretation Rule

Never say:

AI predicted 010
therefore actual error was 010

unless exact error matching has actually been verified.

Instead say:

AI predicted correction = 010

Then evaluate:

Did this correction successfully recover
the logical information?

79. Current AI Research Position

The project has progressed beyond a simple AI demo.

It now contains:

AI model
+
Dataset generation
+
Logical target design
+
Real quantum simulation integration
+
Correction
+
Logical evaluation
+
Baseline
+
Paired comparison
+
Statistical validation
+
Noise sweeps
+
Scientific plots
+
API
+
Dashboard
+
Live trace

80. What Is Already Complete

Core development is complete for the current scope:

Quantum simulation

3-qubit QEC

Noise simulation

Syndrome generation

Detection events

Dataset generation

AI decoder

Logical-target learning

Correction

Logical recovery

Evaluation

Baseline

Scientific experiments

Statistical comparison

Result storage

FastAPI

AI trace

Next.js dashboard

Scientific visualization

Header/footer components

Automated tests

Remaining work is mainly:

Production polish
Documentation
UI refinement
Warning cleanup
Additional research
Larger AI models
Larger QEC codes
Hardware validation

81. Current AI Model

Remember:

Primary decoder:
LogicalTargetRandomForestDecoder

Default:

Random Forest estimators = 100

82. Current AI Dataset Defaults

The main experiment configuration currently uses:

Training samples = 5000
Test samples     = 1000
Rounds           = 5
Seed             = 42

Scientific experiments may intentionally use different sample counts for controlled sweeps.

Always check the script/configuration being run rather than assuming every experiment uses the same values.

83. Current QEC Defaults

QEC code:
bit_flip_3

Physical qubits:
3

Rounds:
5

84. Current Noise Defaults

The experiment configuration supports:

Physical noise probability
Measurement noise probability

Both are configurable between:

0.0 and 1.0

85. AI Interview Explanation

If asked:

"What exactly did AI do in your project?"

Answer:

"I used machine learning as the decoding layer. The QEC simulation generates syndrome histories and detection events, which I convert into numerical features. I then train a logical-target Random Forest decoder to predict a useful three-bit correction. The predicted correction is applied to the simulated quantum state, and I evaluate the result based on logical recovery rather than only exact physical-error matching. Finally, I compare the AI decoder with a baseline across different noise levels using paired test samples and statistical analysis."

86. AI Flow to Memorize

Noise
 ↓
Syndrome
 ↓
Features
 ↓
AI Decoder
 ↓
Correction
 ↓
Logical Recovery
 ↓
Evaluation
 ↓
AI vs Baseline

87. The Five Most Important AI Concepts

If returning to the project after a long time, remember these first:

1. Input

Syndrome history
+
Detection events

2. Features

Numerical representation of those observations

3. Target

Logical-aware correction

4. Model

Random Forest

5. Objective

Improve logical recovery

88. One-Minute Project Memory Refresh

When you come back after several weeks, read this:

I built an AI-powered QEC system.

The quantum system is simulated using Qiskit/Aer.

I use a 3-qubit bit-flip repetition code.

Noise creates physical errors.

QEC produces syndrome observations.

I preserve syndrome history and calculate detection events.

Those observations become machine-learning features.

The target is a logically useful correction rather than
only an exact physical-error label.

The primary AI model is a logical-target Random Forest.

It learns:
syndrome features → correction.

For a new sample:
syndrome → features → AI → predicted correction.

The correction is applied to the simulated state.

Then I measure logical recovery.

I compare AI-QEC against a baseline using the same held-out
test samples.

I tested multiple noise levels and multiple random seeds.

I also performed bootstrap confidence intervals and permutation
testing.

The strongest paired result was at 10% physical noise:
Baseline = 73.42%
AI-QEC   = 84.17%
Gain     = +10.75 percentage points.

The system also has:
FastAPI backend
Next.js dashboard
live AI/QEC trace
scientific evaluation
experiment storage
visualizations
and automated tests.

89. Final Architecture

                         USER
                          │
                          ▼
                    EXPERIMENT CONFIG
                          │
                          ▼
                  QUANTUM SIMULATION
                          │
                          ▼
                       NOISE
                          │
                          ▼
                     QEC SYSTEM
                          │
                          ▼
                    SYNDROME DATA
                          │
                          ▼
                  FEATURE ENGINEERING
                          │
                          ▼
                  LOGICAL TARGET DATA
                          │
                          ▼
                 ┌────────────────────┐
                 │   AI DECODER       │
                 │                    │
                 │ Logical-Target RF  │
                 └─────────┬──────────┘
                           │
                           ▼
                  PREDICTED CORRECTION
                           │
                           ▼
                   CORRECTION ENGINE
                           │
                           ▼
                   LOGICAL RECOVERY
                           │
                           ▼
                       METRICS
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
        BASELINE                      AI-QEC
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  PAIRED COMPARISON
                           │
                           ▼
                  STATISTICAL ANALYSIS
                           │
                           ▼
                    SCIENTIFIC RESULTS
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
          FASTAPI                    NEXT.JS
          BACKEND                   DASHBOARD

90. Final Reminder

If you forget everything else, remember:

This project is fundamentally an AI decoding project.

The quantum simulator generates the problem.

The QEC layer generates the observations.

The dataset converts those observations into learning examples.

The AI decoder learns the mapping from syndrome information to useful corrections.

The correction engine executes the AI decision.

Logical recovery tells us whether the AI decision actually helped.

And the scientific evaluation tells us whether the AI performs better than the baseline.

QEC creates the decoding problem.
        ↓
Data represents the problem.
        ↓
AI learns the decoding pattern.
        ↓
Correction applies the AI decision.
        ↓
Logical recovery measures the real outcome.
        ↓
Scientific comparison validates the AI.

Project Status

Current status: Core AI-QEC development complete for the present 3-qubit simulation scope.

The next major research directions are:

Larger QEC codes
        ↓
More realistic noise
        ↓
Larger datasets
        ↓
Hyperparameter optimization
        ↓
Deep-learning decoders
        ↓
Temporal models
        ↓
Graph neural networks
        ↓
Real quantum-hardware data

Final Project Identity

Project: AI-Powered Quantum Error Correction System

Primary research area:
Machine Learning + Quantum Error Correction

Primary AI model:
Logical-target Random Forest

AI input:
Syndrome history + detection events

AI output:
Predicted physical correction

Primary objective:
Logical information preservation

Current quantum environment:
Classical Qiskit/Aer simulation

Primary scientific comparison:
AI-QEC vs baseline

Primary reported result:
At 10% physical noise, paired AI-QEC logical success was 84.17% versus 73.42% for the baseline, a +10.75 percentage-point gain in the tested configuration.

END OF README