# Quantum QEC AI — Experiments and Scientific Evaluation

## 1. Why does the project have an experiment layer?

The experiment layer turns individual components into repeatable research experiments.

Instead of manually doing:

```text
generate data
train model
test model
save result
```

the project has:

```text
ExperimentEngine
```

to coordinate the process.

---

# 2. Experiment engine

File:

```text
experiments/engine.py
```

Class:

```text
ExperimentEngine
```

Responsibilities:

1. validate configuration
2. generate training data
3. generate test data
4. create decoder
5. train decoder
6. evaluate decoder
7. create result
8. optionally save result

---

# 3. Experiment configuration

File:

```text
experiments/config.py
```

Main settings:

```text
qec_code
num_qubits
logical_state
rounds
physical_noise_probability
measurement_noise_probability
training_samples
test_samples
decoder_type
random_forest_estimators
seed
```

---

# 4. Current supported standard experiment

The configuration currently validates:

```text
qec_code = bit_flip_3
num_qubits = 3
decoder_type = logical_target_random_forest
```

Therefore the standard API path is deliberately limited to the current primary experiment.

---

# 5. Experiment lifecycle

```text
ExperimentConfig
       |
       v
Validate
       |
       v
Generate training samples
       |
       v
Generate test samples
       |
       v
Create decoder
       |
       v
Train
       |
       v
Evaluate
       |
       v
Create ExperimentResult
       |
       v
Save JSON
```

---

# 6. Training/test seeds

The engine uses:

```text
training seed = configured seed
```

and:

```text
test seed = configured seed + 10000
```

This creates an independent generated test dataset.

---

# 7. Result object

Class:

```text
ExperimentResult
```

It stores:

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

# 8. Result storage

Results are saved under:

```text
experiments/results/
```

Each result becomes:

```text
<experiment_id>.json
```

---

# 9. Querying experiments

The API can:

```text
list
filter
sort
find best
find worst
compare
summarize
analyze
visualize
```

The main query class is:

```text
ExperimentResultQuery
```

---

# 10. Scientific comparison

The repository contains a paired baseline-vs-AI experiment.

The key idea is:

```text
same held-out samples
       |
       +--> baseline
       |
       +--> AI
       |
       v
paired difference
```

This is stronger than comparing unrelated random datasets.

---

# 11. Scientific metrics

The formal evaluation calculates:

```text
baseline_logical_success
ai_logical_success
absolute_gain
relative_gain
baseline_logical_error
ai_logical_error
logical_error_reduction
bootstrap_ci_low
bootstrap_ci_high
permutation_p_value
seed_count
test_samples_per_seed
```

---

# 12. Absolute gain

Formula:

```text
AI success - baseline success
```

Example:

```text
AI = 84.17%
Baseline = 73.42%

Gain = 10.75 percentage points
```

---

# 13. Relative gain

Conceptually:

```text
absolute gain / baseline success
```

This expresses the improvement relative to the baseline level.

---

# 14. Logical error reduction

Logical error is:

```text
1 - logical success
```

Then:

```text
error reduction =
(baseline error - AI error)
/
baseline error
```

---

# 15. Bootstrap confidence interval

The paired result generator stores bootstrap confidence interval values.

The formal scientific evaluator then exposes them as:

```text
bootstrap_ci_low
bootstrap_ci_high
```

These describe uncertainty around the measured paired gain.

---

# 16. Permutation test

The paired experiment also calculates a permutation p-value.

The report warns that:

```text
0.00010
```

is the finite-resolution limit of the configured permutation procedure.

It is not an infinitely precise probability.

---

# 17. Seeds

The formal report uses:

```text
42
43
44
```

That is:

```text
3 independent seeds
```

This is useful but limited.

More independent seeds would provide stronger evidence.

---

# 18. Formal reported experiment

The uploaded report uses:

```text
Rounds                = 5
Training samples      = 2000
Test samples/seed     = 2000
RF estimators         = 100
Seeds                 = 42, 43, 44
Measurement noise     = 0.0
Physical noise levels = 0.01
                        0.03
                        0.05
                        0.10
                        0.15
                        0.20
```

---

# 19. Reported results

| Physical noise | Baseline | AI-QEC | Gain |
|---:|---:|---:|---:|
| 0.01 | 99.28% | 99.37% | +0.08 pp |
| 0.03 | 95.45% | 97.60% | +2.15 pp |
| 0.05 | 89.27% | 94.82% | +5.55 pp |
| 0.10 | 73.42% | 84.17% | +10.75 pp |
| 0.15 | 62.75% | 71.97% | +9.22 pp |
| 0.20 | 55.08% | 59.92% | +4.83 pp |

The largest reported absolute gain is:

```text
10.75 percentage points
```

at:

```text
physical noise = 0.10
```

---

# 20. Reported logical-error reduction

The formal report gives:

| Physical noise | Logical-error reduction |
|---:|---:|
| 0.01 | 11.63% |
| 0.03 | 47.25% |
| 0.05 | 51.71% |
| 0.10 | 40.44% |
| 0.15 | 24.74% |
| 0.20 | 10.76% |

The largest reported reduction is:

```text
51.71%
```

at:

```text
physical noise = 0.05
```

---

# 21. Important interpretation

These numbers describe:

```text
one QEC code
one simulation model
one tested decoder
three seeds
specific training/test sizes
specific noise settings
```

They do not establish a universal result across:

```text
surface codes
larger codes
different hardware
different noise channels
different decoder architectures
physical quantum processors
```

---

# 22. Research scripts

The `scripts/` folder contains tools for:

```text
ai_noise_sweep.py
baseline_comparison.py
noise_robustness_matrix.py
paired_baseline_ai_comparison.py
scientific_plots.py
scientific_validation.py
statistical_analysis.py
```

These support broader experimentation.

---

# 23. Research direction

The repository clearly has a broader research direction than the current primary API path.

It contains experiments involving:

```text
different rounds
different noise combinations
multiple decoder types
GRU models
MLP
Logistic Regression
Random Forest
lookup approaches
stochastic noise
time-varying QEC
```

This makes the repository a research platform rather than only a single demo.

---

# 24. One important distinction

The project has:

```text
primary production-like path
```

and:

```text
research exploration path
```

The primary path is:

```text
ExperimentEngine
+
LogicalTargetRandomForestDecoder
```

The broader research path contains many alternative experiments.

---

# 25. One-minute scientific explanation

> "I compare a logical-target Random Forest decoder with a baseline using identical held-out noisy samples. The evaluation measures logical success, absolute and relative gain, logical-error reduction, bootstrap confidence intervals, and paired permutation statistics. The formal report uses three seeds and 2,000 test samples per seed across six physical-noise levels. The results are specific to the 3-qubit simulation configuration and are not presented as universal QEC claims."
