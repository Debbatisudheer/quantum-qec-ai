# Quantum QEC AI — AI Decoder and Machine Learning Guide

## 1. What does the AI do?

The AI is the **decoder**.

Its job is:

> Look at QEC observations and predict a useful correction.

The main model is:

```text
LogicalTargetRandomForestDecoder
```

---

# 2. What goes into the AI?

The AI does not receive the hidden physical error as a normal input.

It receives information derived from:

```text
Observed syndrome history
+
Detection-event history
```

For five rounds:

```text
5 rounds x 4 values
=
20 numerical features
```

Each round contributes:

```text
syndrome bit 1
syndrome bit 2
detection bit 1
detection bit 2
```

---

# 3. Example input

Suppose:

```text
Observed syndrome:

00
10
10
01
00
```

and detection events are:

```text
00
10
00
11
01
```

The decoder turns those values into a numerical vector.

Conceptually:

```text
[0,0,0,0,
 1,0,1,0,
 1,0,0,0,
 0,1,1,1,
 0,0,0,1]
```

The exact vector comes from the project's encoder.

---

# 4. What comes out?

The model predicts three correction bits:

```text
[q0, q1, q2]
```

Example:

```text
010
```

This means:

```text
correction on q1
```

---

# 5. Why MultiOutputClassifier?

The output has three bits.

Conceptually:

```text
Input
 |
 +--> q0 correction
 |
 +--> q1 correction
 |
 +--> q2 correction
```

The implementation uses:

```text
MultiOutputClassifier
```

around:

```text
RandomForestClassifier
```

---

# 6. Why Random Forest?

The project uses Random Forest because it works well for structured/tabular features.

It is also useful for research because it is:

- relatively simple
- practical for the current dataset size
- reasonably interpretable
- fast enough for simulation experiments
- easy to compare against other classical ML models

The project does not claim Random Forest is universally the best decoder.

---

# 7. Logical-target learning

This is the central AI design.

The model is not simply trained on:

```text
Actual physical error
```

Instead, the project builds a correction target based on logical recovery.

The process is:

```text
Observed syndrome history
          |
          v
Group training samples
          |
          v
Look at possible final error states
          |
          v
Try all 8 possible 3-bit corrections
          |
          v
Measure logical success
          |
          v
Choose the best correction
          |
          v
Training target
```

There are only:

```text
2^3 = 8
```

possible correction patterns.

They are:

```text
000
001
010
011
100
101
110
111
```

---

# 8. Target score

For each observed syndrome history, the target builder calculates a score.

Conceptually:

```text
score =
logical successes
/
samples in that observation group
```

The correction with the highest score becomes the target.

This score is stored in:

```text
target_scores
```

---

# 9. Important information boundary

During target construction, the system knows the generated ground truth.

That is normal for supervised learning.

But the actual decoder input is still:

```text
observed syndrome
+
detection events
```

not:

```text
actual error
```

This separation is important.

---

# 10. Training

The primary training flow is:

```text
Generate training data
        |
        v
Build logical targets
        |
        v
Encode features
        |
        v
Create X and y
        |
        v
Random Forest training
```

The model then becomes:

```text
trained decoder
```

---

# 11. Testing

The test set is generated independently.

The engine uses:

```text
training seed
```

and:

```text
seed + 10000
```

for test generation.

So the training and test datasets are not simply the same generated samples.

---

# 12. Inference

For a new sample:

```text
New QEC sample
      |
      v
Feature encoding
      |
      v
Random Forest
      |
      v
3-bit prediction
```

---

# 13. Correction

The predicted correction is applied by XOR-style state logic.

Conceptually:

```text
Corrupted state
      XOR
Predicted correction
      |
      v
Corrected state
```

---

# 14. Logical recovery

The corrected state is passed to:

```text
LogicalRecovery
```

which uses majority voting.

Then:

```text
Recovered logical state
```

is compared to:

```text
Original logical state
```

---

# 15. AI metrics

The project records:

### Exact accuracy

```text
Predicted error == actual error
```

### Physical recovery

Whether the physical codeword was exactly restored.

### Bit accuracy

How many individual correction bits were correct.

### Logical accuracy

Whether the logical information was recovered.

The last one is the main system-level metric.

---

# 16. Confidence

The trace service can calculate a supplementary confidence estimate from Random Forest probabilities.

The confidence is derived from per-output class probabilities.

It is supplementary.

If probability information is unavailable, the trace returns:

```text
confidence = null
```

The trace itself does not depend on confidence.

---

# 17. Other AI models in the repository

The repository contains additional research models.

### Logistic Regression

```text
decoders/ml_logistic.py
```

### MLP

```text
decoders/ml_mlp.py
```

### Random Forest

```text
decoders/ml_random_forest.py
```

### Temporal GRU

```text
decoders/temporal_gru.py
decoders/temporal_gru_classifier.py
```

### Logical-target GRU

```text
decoders/logical_target_gru.py
```

### Time-varying ML

```text
decoders/time_varying_ml.py
```

These show that the repository was used as a research platform, not only as one fixed Random Forest implementation.

---

# 18. Why the API uses Random Forest as the primary decoder

The `ExperimentConfig.validate()` method currently restricts:

```text
decoder_type =
logical_target_random_forest
```

So the API's standard experiment path is deliberately narrower than the complete research repository.

This is useful to remember:

```text
Repository research scope
        >
Primary API experiment scope
```

The repository contains more experimental decoders than the current standard API path exposes.

---

# 19. AI vs classical algorithm

This project uses machine learning for the decoder.

But not every part of the project is AI.

For example:

```text
Syndrome calculation
XOR operations
Majority voting
Dataset validation
Correction
Statistical calculations
```

are ordinary algorithms.

The machine-learning part is mainly:

```text
Feature -> learned correction
```

---

# 20. Beginner example

Imagine:

```text
Observed syndrome:
00, 10, 10, 01, 00
```

The AI converts it to features.

Random Forest processes the features.

It predicts:

```text
010
```

The system applies that correction.

Then logical recovery asks:

```text
Did the original logical 0/1 survive?
```

---

# 21. One-line explanation

> AI in this project learns from observed QEC history and predicts a correction pattern whose goal is to preserve the logical information.

---

# 22. Interview answer

> "The main decoder is a logical-target Random Forest. I convert observed syndrome history and detection events into numerical features. During training, a target builder groups observations and evaluates all possible three-bit corrections according to logical recovery, so the model learns a correction that is useful for the logical state rather than simply memorizing the hidden physical error. During inference, the decoder receives only the observable features and outputs a three-bit correction."

---

# 23. Final AI flow

```text
QEC observations
      |
      v
Feature engineering
      |
      v
Logical target construction
      |
      v
Random Forest
      |
      v
Predicted correction
      |
      v
Correction
      |
      v
Logical recovery
      |
      v
Success / failure
```
