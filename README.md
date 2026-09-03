AI-Powered Quantum Error Correction System

An AI-focused quantum error-correction research platform built entirely with classical quantum simulation.

The project studies whether a machine-learning decoder can learn from quantum error-correction observations and predict useful corrections that improve logical quantum-information recovery.

Important: The current system uses Qiskit/Aer simulation on a classical computer. It does not use physical quantum hardware.

Table of Contents

Overview

Research Question

How the System Works

Current Scope

AI Decoder

Why the Target Is Logical-Aware

Dataset and Features

Training and Inference

Correction and Logical Recovery

Evaluation Metrics

AI vs Baseline

Experimental Results

Statistical Validation

Noise Robustness

Application Architecture

Project Structure

Important Files

Installation and Setup

Running the Project

Testing

Scientific Experiments

Development Problems Solved

Limitations

Future Research

Interview Explanation

Quick Memory Refresh

Project Status

Overview

Quantum systems are sensitive to errors. Quantum error correction (QEC) attempts to protect logical quantum information by observing error-related information and applying appropriate corrections.

This project adds machine learning to that decoding stage.

The quantum simulator creates the noisy QEC problem. The QEC layer produces observations. Those observations are converted into machine-learning features. The AI decoder predicts a correction, the correction is applied, and the final result is evaluated by checking whether the original logical information was recovered.

The central flow is:

Quantum Simulation
       |
       v
Quantum Noise
       |
       v
Syndrome Generation
       |
       v
Syndrome + Detection Events
       |
       v
AI Decoder
       |
       v
Predicted Correction
       |
       v
Apply Correction
       |
       v
Logical Recovery
       |
       v
Evaluation
       |
       v
AI vs Baseline

The primary AI model is a logical-target Random Forest decoder.

Research Question

The main research question is:

Can machine learning learn to decode syndrome information and choose corrections that improve logical quantum-error recovery?

The project is designed to answer that question experimentally within the current simulation scope.

How the System Works

At a high level, the complete AI workflow is:

Noise
  |
  v
Syndrome Observations
  |
  v
Detection Events
  |
  v
Feature Engineering
  |
  v
AI Decoder
  |
  v
Predicted Correction
  |
  v
Correction Engine
  |
  v
Logical Recovery
  |
  v
Metrics
  |
  v
AI vs Baseline
  |
  v
Scientific Analysis

The AI does not receive the hidden physical error as its input.

Instead, it works from information that would be available through the QEC observation process:

Observed Syndrome History
          +
Detection-Event History
          |
          v
     Feature Vector
          |
          v
    Random Forest
          |
          v
  3-bit Correction

This distinction is important because a decoder should make its decision from observed information rather than being given the answer directly.

Current Scope

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

Automated testing

Scope boundary

The current project should not be described as:

a physical quantum computer

hardware QEC

a universal quantum decoder

a surface-code implementation

a production hardware decoder

The accurate description is:

A classical simulation platform for researching AI-based quantum-error decoding and logical recovery.

The QEC Problem

The current system uses a 3-qubit repetition code.

Logical states are encoded as:

Logical 0 -> 000
Logical 1 -> 111

A physical error can change one or more individual qubits.

The QEC system measures stabilizer information and produces a syndrome. For the current 3-qubit code, the syndrome represents the observed error pattern.

Conceptually:

00 -> no detected single-qubit error
01 -> one error pattern
10 -> another error pattern
11 -> another error pattern

The exact syndrome mapping is implemented by the QEC layer.

The AI does not need direct access to the hidden physical error.

Repeated QEC Rounds

The system can perform multiple QEC rounds.

The current default is:

rounds = 5

Conceptually:

Round 1 -> syndrome
Round 2 -> syndrome
Round 3 -> syndrome
Round 4 -> syndrome
Round 5 -> syndrome

Instead of using only one observation, the AI can use the history across multiple rounds.

For example:

00
10
10
01
00

This sequence represents the syndrome history across five rounds.

The history provides temporal information about how the observed error signal changes.

Detection Events

A detection event describes a change between consecutive syndrome observations.

The relationship is:

DetectionEvent[t] =
    Syndrome[t-1] XOR Syndrome[t]

For example:

Previous syndrome = 00
Current syndrome  = 10

Detection event   = 10

Therefore, the AI receives two related kinds of information:

Syndrome History
       +
Changes in Syndrome History
       |
       v
Machine-Learning Features

Measurement Noise

The simulator can also corrupt syndrome observations with measurement noise.

The process is:

Perfect Syndrome
       |
       v
Measurement Noise
       |
       v
Observed Syndrome

The AI normally works with the observed information rather than the hidden perfect syndrome.

This makes the decoding problem more difficult and provides a more realistic simulation of imperfect observations.

AI Decoder

AI Architecture

The core AI pipeline is:

Data Generation
       |
       v
QEC Simulation Data
       |
       v
Syndrome Observation
       |
       v
Detection Events
       |
       v
Feature Engineering
       |
       v
Target Engineering
       |
       v
Training Data
       |
       v
Random Forest Training
       |
       v
Trained Decoder
       |
       v
New Sample
       |
       v
Feature Encoding
       |
       v
AI Inference
       |
       v
Predicted Correction
       |
       v
Correction Engine
       |
       v
Logical Recovery
       |
       v
Evaluation

Primary Model

The primary decoder is:

LogicalTargetRandomForestDecoder

It is based on:

RandomForestClassifier
+
MultiOutputClassifier

MultiOutputClassifier is used because the decoder predicts multiple correction bits.

The current default is:

n_estimators = 100

Why Random Forest?

Random Forest was selected because it:

works well with structured/tabular features

provides a relatively simple machine-learning approach

is fast enough for the current simulation

is comparatively interpretable

provides a useful baseline for future model comparisons

The project does not claim that Random Forest is universally optimal for QEC decoding.

What the AI Predicts

The AI predicts a three-bit correction.

For example:

010

The three bits correspond to correction actions on the three physical qubits.

Conceptually:

Output bit 0 -> correction for qubit 0
Output bit 1 -> correction for qubit 1
Output bit 2 -> correction for qubit 2

The predicted correction is then passed to the correction engine.

Why the Target Is Logical-Aware

One of the most important design decisions is that the objective is logical recovery, not simply exact identification of the hidden physical error.

The reasoning is:

Observed Information
        |
        v
Candidate Corrections
        |
        v
Test Logical Outcome
        |
        v
Choose Useful Correction
        |
        v
Training Target

This is called the logical-target approach.

A correction does not necessarily need to reproduce the exact physical error pattern if another correction produces the same desired logical outcome.

Exact Error Match vs Logical Success

These are different concepts and must not be confused.

Exact physical-error match

Actual Error == Predicted Correction

Logical success

After applying the predicted correction:

Original logical information is recovered

Therefore, an AI prediction can fail exact physical-error matching and still succeed logically.

For this project, logical success is the more important system-level outcome.

Dataset and Features

Dataset Generation

Each synthetic sample follows approximately:

Choose Logical State
       |
       v
Encode Logical State
       |
       v
Generate Physical Error History
       |
       v
Run QEC Rounds
       |
       v
Generate Syndrome History
       |
       v
Apply Measurement Noise
       |
       v
Create Observed Syndrome History
       |
       v
Calculate Detection Events
       |
       v
Encode Features
       |
       v
Create Logical-Aware Target
       |
       v
Store Sample

Because the system is simulation-based, the simulator knows the ground truth for generated samples. This makes supervised-learning target generation and evaluation possible.

Training Data

A training dataset contains many examples of:

Input:
syndrome and detection-event features

Target:
useful correction

Conceptually:

Features -> [0, 1, 0, 0, 1, ...]
Target   -> [0, 1, 0]

The actual feature length depends on the number of QEC rounds.

Test Data

Test samples are held out from training:

Training Samples
       |
       v
AI Learns
       |
       v
Held-Out Test Samples
       |
       v
AI Evaluation

The test set must not be used to train the model. This helps protect the validity of the evaluation.

Feature Engineering

Feature engineering converts QEC observations into numerical machine-learning input.

The basic process is:

Observed Syndrome History
          +
Detection Events
          |
          v
    Feature Encoder
          |
          v
   Numerical Feature Vector

The logical-target Random Forest uses a per-round representation conceptually like:

Round 1:
s1, s2, d1, d2

Round 2:
s1, s2, d1, d2

...

Round N:
s1, s2, d1, d2

Feature Interface Adapter

One integration path represents features as:

all syndrome bits
+
all detection-event bits

The logical-target Random Forest expects:

s1, s2, d1, d2

for each round.

Therefore, an adapter converts between these representations:

Integration Features
       |
       v
Feature Adapter
       |
       v
Random Forest Feature Format

The adapter allows the existing decoder to be reused without changing its core logic.

Training and Inference

Training

Training follows:

Training Samples
       |
       +------> Feature Matrix X
       |
       +------> Target Matrix y
                    |
                    v
              Random Forest
                    |
                   fit()
                    |
                    v
             Trained Decoder

The learned relationship is approximately:

Syndrome-Related Observations
             |
             v
      Useful Correction

Inference

For a new sample:

New Syndrome Observations
          |
          v
Detection Events
          |
          v
Feature Encoding
          |
          v
Trained Random Forest
          |
          v
Predicted Correction

Example:

Prediction = 010

Decoder Interface

The decoder supports operations conceptually including:

train(samples)
predict(X)
predict_proba(X)
decode(sample)
decode_batch(samples)

This keeps the AI subsystem modular and makes it possible to compare or replace models later.

AI Confidence

The Random Forest can provide probability information for predictions.

The project can expose a confidence-like value in the trace.

However:

Model probability/confidence is not a guarantee that the predicted correction is correct.

Correction and Logical Recovery

AI to Correction

The separation between decision-making and action is intentional:

AI Decoder
     |
     v
Decision
     |
     v
Correction Engine
     |
     v
Action

For example:

Predicted correction = 010

The correction engine applies that decision to the simulated quantum state.

Correction Mathematics

Let:

e = actual physical error
c = AI-predicted correction

Then:

Corrupted state
=
Encoded state XOR e

After the AI correction:

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

However, exact cancellation is stricter than logical recovery.

Logical Recovery

After correction, the system evaluates whether the original logical state was recovered.

For the current 3-qubit repetition code, majority information determines the recovered logical state.

The important objective is:

Predicted Correction
       |
       v
Logical Recovery
       |
       v
Preserve Logical Information

Evaluation Metrics

The project evaluates multiple dimensions of AI performance.

Metric

Question

Exact Accuracy

Did the AI predict the exact target?

Physical Recovery

Did the AI exactly reverse the physical error?

Bit Accuracy

How many individual correction bits were correct?

Logical Accuracy

Was the original logical state recovered?

Logical Error Rate

What fraction of samples failed logical recovery?

Training Time

How long did model training take?

Inference Time

How long did prediction take?

Throughput

How many samples can be processed per second?

Primary AI Metric

The most important outcome metric for this project is:

Logical Success

because the system exists to preserve logical information.

The hierarchy is:

AI Prediction
     |
     v
Correction
     |
     v
Logical Recovery
     |
     v
System Objective

AI vs Baseline

AI performance needs a reference point.

The project therefore compares:

Baseline
   vs
AI-QEC

The baseline represents the traditional/non-AI logical recovery behavior used for comparison.

Paired Comparison

The stronger comparison uses the same held-out test samples for both methods:

              Same Test Sample
                     |
              +------+------+
              |             |
              v             v
           Baseline       AI-QEC
              |             |
              v             v
           Result          Result

Using the same samples makes the comparison more controlled.

AI Gain

Absolute AI gain is:

AI Logical Success
-
Baseline Logical Success

Example:

AI       = 84.17%
Baseline = 73.42%

Gain = +10.75 percentage points

Relative Gain

Relative gain is:

(AI - Baseline) / Baseline

This describes improvement relative to the baseline level.

Logical Error Reduction

First:

Baseline Error = 1 - Baseline Success
AI Error       = 1 - AI Success

Then:

Error Reduction =
(Baseline Error - AI Error)
/
Baseline Error

This measures how much of the baseline logical error was removed by AI.

Experimental Results

Paired AI vs Baseline Results

The strongest paired comparison produced the following results:

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

Interpretation

Within the tested simulation configuration:

AI-QEC > Baseline

at every tested nonzero physical-noise level in the paired study.

This suggests that the learned decoder can exploit syndrome-derived information to improve logical recovery relative to the tested baseline.

This result should not be generalized beyond the tested configuration without additional experiments.

Statistical Validation

The project uses:

multiple random seeds

paired test samples

bootstrap confidence intervals

permutation testing

The purpose is to determine whether the observed AI improvement is reasonably stable within the experiment.

Random Seeds

Controlled random seeds include:

42
43
44

Using multiple seeds helps evaluate different random realizations and improves reproducibility.

The main paired experiments use 3 independent seeds.

Bootstrap

Bootstrap analysis follows:

Observed Paired Results
        |
        v
Repeated Resampling
        |
        v
Calculate Gain Repeatedly
        |
        v
Gain Distribution
        |
        v
Confidence Interval

The project uses descriptive 95% bootstrap intervals.

Permutation Test

The paired differences are subjected to random sign changes to construct a null distribution.

The implementation uses:

10,000 sign randomizations

Therefore, a displayed p-value around:

0.00010

is the resolution limit of this finite permutation procedure, not an infinitely precise probability.

Statistical Limitation

Because the main paired experiments use only three independent seeds, the results should be treated as experimental evidence for the tested setup rather than universal proof of AI superiority across all QEC systems.

Noise Robustness

Physical Noise Sweep

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

Physical + Measurement Noise

Combined noise was also tested.

For example:

Physical noise     = 0.10
Measurement noise  = 0.10

Observed mean AI logical success was approximately:

73.40%

This demonstrates that imperfect syndrome observations make the AI decoding problem harder.

Measurement-Only Caveat

The measurement-only experiment used:

Physical noise = 0
Measurement noise > 0

and produced:

100% logical success

This should not be interpreted as strong evidence of measurement-noise robustness because there was no physical error to correct.

The more meaningful robustness test is the combined physical + measurement noise experiment.

Application Architecture

The project is organized into several layers.

                       USER
                        |
                        v
                Experiment Config
                        |
                        v
                Quantum Simulation
                        |
                        v
                      Noise
                        |
                        v
                    QEC System
                        |
                        v
                  Syndrome Data
                        |
                        v
               Feature Engineering
                        |
                        v
              Logical Target Data
                        |
                        v
                +---------------+
                |   AI Decoder  |
                | Logical-Target|
                | Random Forest |
                +-------+-------+
                        |
                        v
              Predicted Correction
                        |
                        v
                Correction Engine
                        |
                        v
                 Logical Recovery
                        |
                        v
                     Metrics
                    /       \
                   v         v
              Baseline     AI-QEC
                   \         /
                    \       /
                     v     v
                Paired Comparison
                        |
                        v
                Statistical Analysis
                        |
                        v
                 Scientific Results
                    /           \
                   v             v
              FastAPI          Next.js
              Backend          Dashboard

Experiment Engine

The experiment engine automates:

Configuration
     |
     v
Training Data Generation
     |
     v
AI Training
     |
     v
Test Data Generation
     |
     v
AI Inference
     |
     v
Evaluation
     |
     v
Result Storage

Important configuration values include:

QEC code

number of qubits

logical state

number of rounds

physical noise probability

measurement noise probability

training samples

test samples

decoder type

Random Forest estimators

random seed

Result Storage and Analysis

Experiment results are stored as JSON.

Stored information includes:

experiment ID

configuration

training sample count

test sample count

target information

accuracy metrics

training time

inference time

throughput

decoder type

Stored experiments can then be:

Filtered
Sorted
Compared
Summarized

For example:

Find experiments at 10% physical noise
        |
        v
Sort by logical accuracy
        |
        v
Compare decoders

Scientific Evaluation and Visualization

The scientific evaluation layer converts paired experiment results into reusable scientific metrics.

It calculates:

baseline success

AI success

absolute gain

relative gain

baseline error

AI error

error reduction

bootstrap confidence interval

permutation p-value

seed count

test samples per seed

Scientific plots include:

logical success vs noise

paired AI gain

bootstrap confidence intervals

logical error rate

combined-noise heatmap

These visualizations help show where the AI performs well and where performance degrades.

AI Live Trace

The application provides a single-sample AI trace.

Conceptually:

Sample
  |
  v
Physical Error
  |
  v
Perfect Syndrome
  |
  v
Observed Syndrome
  |
  v
Detection Events
  |
  v
AI Decoder
  |
  v
Predicted Correction
  |
  v
Corrected State
  |
  v
Logical Recovery

This makes one AI decision explainable step by step.

Important Trace Interpretation

A trace may show:

Actual error          = 000
Predicted correction  = 010
Exact match           = NO
Logical success       = YES

This is not necessarily a bug.

It demonstrates the distinction between:

Exact Physical Prediction

and:

Logical Preservation

Never assume that a predicted correction is the hidden physical error unless exact matching has actually been verified.

Backend and Frontend

FastAPI Backend

The backend exposes the AI/QEC system through FastAPI.

Important operations include:

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

Current development port:

Backend -> 8001

Health endpoint:

GET /health

Expected response:

{
  "status": "ok",
  "service": "quantum-qec-ai"
}

Next.js Frontend

The frontend provides:

Experiment Configuration
        |
        v
Simulation
        |
        v
AI/QEC Trace
        |
        v
Predicted Correction
        |
        v
Logical Recovery
        |
        v
Scientific Results
        |
        v
Experiment Analysis

Current development port:

Frontend -> 3001

The frontend API base URL is currently:

http://127.0.0.1:8001

Ports may be changed if they are unavailable.

Project Structure

quantum-qec-ai/
|
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── schemas.py
│   │   └── routes.py
│   └── services/
│       └── trace_service.py
|
├── quantum/
│   ├── circuit.py
│   ├── simulator.py
│   ├── repeated_qec.py
│   ├── repeated_measurement.py
│   └── state_evaluator.py
|
├── qec/
│   └── bit_flip_3.py
|
├── noise/
│   ├── bit_flip.py
│   ├── quantum_bit_flip.py
│   └── stochastic_repeated_noise.py
|
├── syndrome/
│   ├── extractor.py
│   └── measurement_noise.py
|
├── dataset/
│   └── ...
|
├── decoders/
│   └── ...
|
├── correction/
│   └── ...
|
├── evaluation/
│   └── ...
|
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
│   └── ...
|
├── scripts/
│   ├── scientific_validation.py
│   ├── baseline_comparison.py
│   ├── ai_noise_sweep.py
│   ├── noise_robustness_matrix.py
│   ├── statistical_analysis.py
│   ├── paired_baseline_ai_comparison.py
│   └── scientific_plots.py
|
├── frontend/
│   ├── app/
│   │   └── page.tsx
│   └── component/
│       ├── SiteHeader.tsx
│       └── SiteFooter.tsx
|
├── tests/
├── configs/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

Important Files

Purpose

File

Primary AI decoder

decoders/logical_target_random_forest.py

AI + quantum integration

experiments/quantum_ai_decoder_integration.py

Experiment configuration

experiments/config.py

Experiment execution

experiments/engine.py

Scientific evaluation

evaluation/scientific_evaluation.py

AI trace service

backend/services/trace_service.py

API routes

backend/api/routes.py

Frontend

frontend/app/page.tsx

Header

frontend/component/SiteHeader.tsx

Footer

frontend/component/SiteFooter.tsx

Installation and Setup

The project is developed with a Python virtual environment.

Windows PowerShell

Activate the environment:

.\.venv\Scripts\Activate.ps1

Verify Python:

where.exe python

The Python executable should point into:

...\quantum-qec-ai\.venv\Scripts\python.exe

Verify Qiskit:

python -c "import qiskit; print(qiskit.__file__)"

The Qiskit installation should point into:

.venv\Lib\site-packages

Running the Project

Start the Backend

From the project root:

python -m uvicorn backend.main:app --port 8001

Backend:

http://127.0.0.1:8001

Health check:

http://127.0.0.1:8001/health

Start the Frontend

Move into:

cd frontend

Then run the normal Next.js development command configured by the project.

The current development server was using:

http://localhost:3001

The frontend should communicate with:

http://127.0.0.1:8001

Testing

From the project root:

python -m pytest -q

Current full-suite status when this README was prepared:

99 passed
29 warnings

The warnings are pytest diagnostic warnings caused by some tests returning values rather than using assertions. They are warnings, not failed tests.

Scientific Experiments

Run scripts from the project root using Python module execution.

Scientific Validation

python -m scripts.scientific_validation

Use module execution from the project root rather than:

python scripts\scientific_validation.py

when the latter causes Python import-path problems.

Baseline Comparison

python -m scripts.baseline_comparison

Evaluates the non-AI baseline across physical-noise levels.

AI Noise Sweep

python -m scripts.ai_noise_sweep

Evaluates AI logical recovery across physical-noise levels.

Combined Noise Matrix

python -m scripts.noise_robustness_matrix

Evaluates combinations of:

Physical Noise
+
Measurement Noise

Statistical Analysis

python -m scripts.statistical_analysis

Analyzes:

baseline

AI-QEC

gain

logical error

error reduction

confidence intervals

combined-noise behavior

Paired Baseline vs AI Comparison

python -m scripts.paired_baseline_ai_comparison

This performs the controlled same-test-sample comparison between baseline and AI-QEC.

Paired results are stored under:

experiments/paired_results/

Scientific Plots

python -m scripts.scientific_plots

Generated outputs include:

experiments/paired_results/plots/

with logical-success, gain, confidence-interval, logical-error, and heatmap visualizations.

Scientific Evaluation Outputs

The scientific evaluation can produce:

experiments/scientific_evaluation/
├── scientific_results.json
├── scientific_results.csv
└── scientific_report.txt

Development Problems Solved

Qiskit Environment Mismatch

Problem

ModuleNotFoundError: No module named 'qiskit'

Cause

Uvicorn reload was using the global Python installation instead of the virtual environment.

Solution

.\.venv\Scripts\Activate.ps1

Then verify:

where.exe python

Backend Port Conflict

Port 8000 was already occupied.

The backend was therefore moved to:

8001

Run:

python -m uvicorn backend.main:app --port 8001

Frontend Port Conflict

Port 3000 was already occupied, so Next.js used:

3001

The ports are independent:

Frontend -> 3001
Backend  -> 8001

Frontend API Connection

The frontend initially attempted to call port 8000 while the backend was running on 8001.

The frontend API base URL was corrected to:

http://127.0.0.1:8001

Trace API Type Bug

The trace API initially returned actual_error as a list while the frontend schema expected a string.

The service was corrected to convert the error bits into a string representation.

Trace Semantic Bug

The trace initially used a post-correction state as the corrupted state.

That was semantically incorrect.

The correct relationship is:

Encoded State
      XOR
Actual Error
      |
      v
Corrupted State

The service now calculates the corrupted state explicitly.

Decoder Feature Mismatch

The integration layer and Random Forest decoder used different feature layouts.

An adapter was added:

Integration Features
       |
       v
Adapter
       |
       v
RF Feature Format

This avoided rewriting the decoder.

Limitations

The current results should be interpreted within the scope of the implementation.

Simulation only

The system currently runs on a classical computer using Qiskit/Aer simulation.

It has not been validated on physical quantum hardware.

Small QEC code

The current QEC implementation is a 3-qubit bit-flip repetition code.

It is not a surface-code implementation or a general-purpose quantum decoder.

Limited random seeds

The main paired experiments use three independent seeds:

42, 43, 44

More independent trials would strengthen statistical confidence.

Measurement-only experiment

A measurement-noise-only experiment produced 100% logical success because physical noise was zero. This is weak evidence for measurement-noise robustness.

Combined physical and measurement noise is more meaningful.

Exact physical error is not the same as logical success

A decoder can fail to reproduce the exact physical error and still preserve the logical state.

Therefore, results should be interpreted using logical recovery as the primary system-level metric.

Statistical scope

Bootstrap and permutation results provide evidence for the tested experimental configuration. They do not prove universal AI superiority across all QEC codes, noise models, hardware platforms, or datasets.

Future Research

The next major research directions are:

Larger QEC Codes
       |
       v
More Realistic Noise
       |
       v
Larger Datasets
       |
       v
Hyperparameter Optimization
       |
       v
Deep-Learning Decoders
       |
       v
Temporal Models
       |
       v
Graph Neural Networks
       |
       v
Real Quantum-Hardware Data

Possible future model directions include:

larger Random Forest studies

temporal neural networks

GRU-based decoders

graph neural networks

larger QEC codes

more realistic noise models

hardware-generated datasets

Interview Explanation

If asked:

What exactly did AI do in your project?

A concise answer is:

I used machine learning as the decoding layer of a quantum error-correction system. The QEC simulation generates syndrome histories and detection events, which I convert into numerical features. I then train a logical-target Random Forest decoder to predict a useful three-bit correction. The predicted correction is applied to the simulated quantum state, and I evaluate the result based on logical recovery rather than only exact physical-error matching. Finally, I compare the AI decoder with a baseline across different noise levels using paired test samples and statistical analysis.

Quick Memory Refresh

If returning to the project after several weeks, remember this:

QEC simulation creates noisy quantum states.

        ↓

QEC produces syndrome observations.

        ↓

Syndrome history + detection events become
machine-learning features.

        ↓

The AI uses a logical-aware correction target.

        ↓

The primary model is a Random Forest.

        ↓

AI learns:

syndrome features -> useful correction

        ↓

For a new sample:

syndrome -> features -> AI -> correction

        ↓

The correction is applied to the simulated state.

        ↓

Logical recovery determines whether the
original logical information was preserved.

        ↓

AI-QEC is compared with a baseline.

        ↓

Scientific analysis evaluates the observed difference.

Five things to remember

Concept

Current implementation

Input

Syndrome history + detection events

Features

Numerical representation of those observations

Target

Logical-aware correction

Model

Logical-target Random Forest

Objective

Improve logical recovery

One-Minute Project Summary

I built an AI-powered quantum error-correction system using Qiskit/Aer simulation on a classical computer.

The system uses a 3-qubit bit-flip repetition code. Noise creates physical errors, and QEC produces syndrome observations. I preserve the syndrome history and calculate detection events. These observations become machine-learning features.

The primary AI model is a logical-target Random Forest decoder. Instead of learning only the hidden physical-error label, it learns to predict a correction that is useful for logical recovery.

For a new sample:

Syndrome
   ->
Features
   ->
AI Decoder
   ->
Predicted Correction
   ->
Logical Recovery

The AI decoder is evaluated using logical success, logical error rate, exact accuracy, physical recovery, bit accuracy, training time, inference time, and throughput.

I also compare AI-QEC with a baseline using the same held-out test samples, multiple random seeds, bootstrap confidence intervals, and permutation testing.

The strongest paired result in the tested configuration was at 10% physical noise:

Baseline = 73.42%
AI-QEC   = 84.17%
Gain     = +10.75 percentage points

The system also includes:

FastAPI backend

Next.js dashboard

live AI/QEC trace

scientific evaluation

experiment storage

scientific visualizations

automated tests

Final Architecture

                         USER
                          |
                          v
                  EXPERIMENT CONFIG
                          |
                          v
                  QUANTUM SIMULATION
                          |
                          v
                        NOISE
                          |
                          v
                       QEC SYSTEM
                          |
                          v
                     SYNDROME DATA
                          |
                          v
                  FEATURE ENGINEERING
                          |
                          v
                 LOGICAL TARGET DATA
                          |
                          v
              +-----------------------+
              |       AI DECODER      |
              | Logical-Target RF    |
              +-----------+-----------+
                          |
                          v
                 PREDICTED CORRECTION
                          |
                          v
                   CORRECTION ENGINE
                          |
                          v
                   LOGICAL RECOVERY
                          |
                          v
                       METRICS
                      /       \
                     v         v
                 BASELINE    AI-QEC
                     \         /
                      \       /
                       v     v
                  PAIRED COMPARISON
                          |
                          v
                  STATISTICAL ANALYSIS
                          |
                          v
                  SCIENTIFIC RESULTS
                     /           \
                    v             v
                FASTAPI        NEXT.JS
                BACKEND        DASHBOARD

Project Status

Current status: Core AI-QEC development is complete for the present 3-qubit simulation scope.

The project currently contains:

quantum simulation

3-qubit QEC

noise simulation

syndrome generation

detection events

dataset generation

AI decoder

logical-target learning

correction

logical recovery

evaluation

baseline comparison

scientific experiments

statistical comparison

result storage

FastAPI backend

AI trace

Next.js dashboard

scientific visualization

automated tests

Remaining work is mainly:

production polish

documentation refinement

UI refinement

warning cleanup

additional research

larger QEC codes

more realistic noise models

larger datasets

hardware validation

Final Project Identity

Item

Current Project

Project

AI-Powered Quantum Error Correction System

Research Area

Machine Learning + Quantum Error Correction

Primary AI Model

Logical-target Random Forest

AI Input

Syndrome history + detection events

AI Output

Predicted physical correction

Primary Objective

Logical information preservation

Quantum Environment

Classical Qiskit/Aer simulation

Scientific Comparison

AI-QEC vs baseline

Strongest Reported Result

At 10% physical noise, AI-QEC achieved 84.17% logical success vs 73.42% for the baseline, a +10.75 percentage-point gain in the tested configuration

Final Reminder

This project is fundamentally an AI decoding project.

QEC creates the decoding problem.
        |
        v
Data represents the problem.
        |
        v
AI learns the decoding pattern.
        |
        v
Correction applies the AI decision.
        |
        v
Logical recovery measures the real outcome.
        |
        v
Scientific comparison evaluates the AI.

The quantum simulator creates the problem.

The QEC layer creates the observations.

The dataset converts those observations into learning examples.

The AI decoder learns the mapping from syndrome information to useful corrections.

The correction engine executes the AI decision.

Logical recovery tells us whether the AI decision actually helped.

Scientific evaluation tells us how the AI performed compared with the baseline.