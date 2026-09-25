# Quantum QEC AI — Dataset, Noise, and Features

## 1. Why does the project need a dataset?

Machine learning needs examples.

The project generates synthetic QEC examples containing:

```text
observations
+
ground truth
```

The observations are used by the decoder.

The ground truth is used to build training targets and evaluate the result.

---

# 2. Dataset generation

The primary generator is:

```text
dataset/time_varying_generator.py
```

Class:

```text
TimeVaryingQECDatasetGenerator
```

It generates repeated-QEC samples.

---

# 3. One generated sample

A sample includes:

```text
logical state
encoded state
physical error history
syndrome history
observed syndrome history
detection events
final error state
```

This allows the project to study how much useful information is present in the observations.

---

# 4. Logical state

Each sample randomly chooses:

```text
logical_state = 0
```

or:

```text
logical_state = 1
```

Then:

```text
0 -> 000
1 -> 111
```

---

# 5. Physical noise

The generator starts with:

```text
error_state = [0, 0, 0]
```

For each round, each qubit independently has a probability of receiving an X-error transition.

Conceptually:

```text
for q0:
    maybe flip

for q1:
    maybe flip

for q2:
    maybe flip
```

The error state is accumulated across rounds.

---

# 6. Example

Suppose physical error probability is:

```text
0.10
```

A round may produce:

```text
previous = 000
current  = 010
```

Another round could change:

```text
010 -> 011
```

The generator records the history.

---

# 7. Perfect syndrome

For the time-varying synthetic model:

```text
S1 = q0 XOR q1
S2 = q1 XOR q2
```

So the syndrome is:

```text
"s1s2"
```

For example:

```text
01
```

---

# 8. Measurement noise

The perfect syndrome can then be passed through:

```text
SyndromeMeasurementNoise
```

The result is:

```text
observed_syndrome
```

So:

```text
Perfect
syndrome
   |
   v
Measurement noise
   |
   v
Observed syndrome
```

---

# 9. Detection events

The project calculates changes in observed syndrome.

For each round:

```text
event =
previous_observed_syndrome
XOR
current_observed_syndrome
```

The first previous value is:

```text
00
```

---

# 10. Feature representation

The Random Forest decoder combines:

```text
observed syndrome
+
detection event
```

for every round.

One round:

```text
[s1, s2, d1, d2]
```

Five rounds:

```text
[
 [s1,s2,d1,d2],
 [s1,s2,d1,d2],
 [s1,s2,d1,d2],
 [s1,s2,d1,d2],
 [s1,s2,d1,d2]
]
```

The Random Forest version flattens this into a vector.

---

# 11. Why detection events are useful

A syndrome value tells:

```text
"What is being observed now?"
```

A detection event tells:

```text
"What changed?"
```

Therefore:

```text
Syndrome history
+
Change history
```

provides more temporal information than a single syndrome value.

---

# 12. Dataset splitting

The repository also contains builders for train/validation/test splitting.

The time-varying ML dataset builder uses:

```text
train_test_split
```

and can create:

```text
training set
validation set
test set
```

The unified experiment engine uses a simpler independent training/test generation strategy:

```text
training seed = seed
test seed = seed + 10000
```

---

# 13. Dataset validation

The repository contains:

```text
dataset/validator.py
```

It checks conditions such as:

- dataset not empty
- valid QEC code
- valid number of qubits
- valid logical state
- valid error qubit
- valid error type
- corrupted state consistency
- syndrome consistency
- target consistency

---

# 14. Dataset storage

The project can store standard dataset samples as:

```text
CSV
JSON
```

using:

```text
QECDatasetStorage
```

The main experiment result store is separate and uses JSON under:

```text
experiments/results/
```

---

# 15. Ground truth vs features

This distinction is critical.

### Decoder input

```text
observed_syndrome_history
detection_events
```

### Ground truth

```text
physical_error_history
final_error_state
perfect syndrome
```

Ground truth is needed for supervised learning and evaluation.

But it should not be secretly given to the decoder as an input.

---

# 16. Why this matters

If the model were given:

```text
final_error_state
```

as an input, the experiment would no longer test whether the model can infer the correction from QEC observations.

The project explicitly separates:

```text
what the decoder sees
```

from:

```text
what the simulator knows
```

---

# 17. Dataset and AI flow

```text
Generator
   |
   +--> hidden physical error
   |
   +--> perfect syndrome
   |
   +--> observed syndrome
   |
   +--> detection events
            |
            v
       AI features
            |
            v
         Decoder
```

The hidden information remains useful for:

```text
target construction
evaluation
```

---

# 18. Noise experiments

The repository contains scripts for:

```text
AI noise sweep
baseline comparison
noise robustness matrix
paired baseline/AI comparison
statistical analysis
scientific validation
scientific plots
```

This means the project is not limited to one noise setting.

---

# 19. Important scientific caveat

The formal scientific report currently uses:

```text
measurement_noise = 0.0
```

for its main paired six-condition result.

So the strongest reported table is primarily a **physical-noise sweep**, not a combined physical+measurement-noise sweep.

The repository contains separate measurement-noise experiments, but those should not be confused with the main paired table.

---

# 20. Measurement-only caveat

The README reports a measurement-noise-only experiment with:

```text
physical noise = 0
measurement noise > 0
```

and:

```text
100% logical success
```

That does not provide strong evidence for correcting physical errors because there were no physical errors to correct.

The more meaningful case is:

```text
physical noise > 0
+
measurement noise > 0
```

---

# 21. Beginner mental model

Think of the dataset like this:

```text
HIDDEN WORLD
    |
    | simulator knows everything
    v
+--------------------------+
| Physical error           |
| Perfect syndrome         |
+--------------------------+
             |
             v
      Observation process
             |
             v
+--------------------------+
| Observed syndrome        |
| Detection events         |
+--------------------------+
             |
             v
           AI
```

---

# 22. One-line explanation

> The dataset generator creates controlled noisy QEC situations, records what the decoder can observe, and keeps the hidden state separately for training-target construction and evaluation.
