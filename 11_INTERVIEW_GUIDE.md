# Quantum QEC AI — Interview Guide

## 1. What is the project?

> "It is an AI-powered quantum error-correction research platform. I use Qiskit/Aer to simulate a 3-qubit bit-flip repetition code on a classical computer, generate noisy syndrome observations, train a machine-learning decoder, apply its predicted correction, and measure whether the original logical information is recovered."

---

# 2. Why did you build it?

The research question is:

> Can machine learning learn from QEC observations and predict useful corrections that improve logical recovery?

---

# 3. What is the QEC code?

The current code is:

```text
3-qubit bit-flip repetition code
```

Encoding:

```text
0 -> 000
1 -> 111
```

---

# 4. What kind of error is modeled?

The main physical error is a:

```text
bit-flip / X error
```

The project also models syndrome measurement noise.

---

# 5. What is the AI input?

The AI receives:

```text
observed syndrome history
+
detection-event history
```

It does not receive the hidden final physical error as its inference input.

---

# 6. What does the AI output?

A three-bit correction:

```text
[q0, q1, q2]
```

Example:

```text
010
```

---

# 7. What model do you use?

The primary model is:

```text
LogicalTargetRandomForestDecoder
```

It uses:

```text
RandomForestClassifier
+
MultiOutputClassifier
```

---

# 8. Why Random Forest?

> "The current features are structured binary/tabular features, so Random Forest is a practical model for the current research scope. It is relatively simple, fast enough for these experiments, and provides a useful baseline for comparing more advanced models."

---

# 9. What is special about the target?

The project uses a:

```text
logical-aware target
```

Instead of only asking:

```text
What was the exact hidden physical error?
```

it asks:

```text
Which correction gives the best logical recovery
for this observed syndrome history?
```

---

# 10. How are targets built?

```text
Observed syndrome history
       |
       v
Group training examples
       |
       v
Find possible final errors
       |
       v
Try all 8 correction patterns
       |
       v
Measure logical success
       |
       v
Choose highest-scoring correction
```

---

# 11. What is a detection event?

A detection event is a syndrome change:

```text
Detection[t] =
Syndrome[t-1] XOR Syndrome[t]
```

It gives temporal information.

---

# 12. What is logical recovery?

For the 3-qubit repetition code, majority voting is used.

Example:

```text
100 -> logical 0
110 -> logical 1
```

---

# 13. Why is logical accuracy more important than exact accuracy?

Because QEC is intended to protect logical information.

A prediction can differ from the hidden physical error and still leave the correct logical state.

So:

```text
Exact physical match
```

is not always required for:

```text
Logical success
```

---

# 14. Does the project use a real quantum computer?

No.

It uses:

```text
Qiskit
Qiskit Aer
```

on a classical machine.

---

# 15. Does it have a quantum-simulation path?

Yes.

The repository contains a repeated QEC circuit and a quantum-AI integration path.

The trace service can run:

```text
Qiskit circuit
   |
   v
syndrome measurement
   |
   v
AI decoder
   |
   v
quantum correction
   |
   v
final measurement
   |
   v
logical recovery
```

---

# 16. What is the FastAPI backend?

FastAPI exposes the research system as APIs.

Important endpoints:

```text
POST /simulate
POST /simulate/trace

GET /experiments
GET /results/{experiment_id}
GET /experiments/summary
GET /experiments/analysis
GET /experiments/visualization

GET /scientific/evaluation
GET /scientific/results
GET /scientific/summary
```

---

# 17. What is the frontend?

It is:

```text
Next.js
React
TypeScript
Tailwind CSS
```

It provides:

```text
experiment configuration
experiment execution
history
metrics
scientific charts
trace visualization
```

---

# 18. Does it use a database?

The uploaded project does not use PostgreSQL/MySQL for experiment result persistence.

It uses:

```text
JSON files
```

under:

```text
experiments/results/
```

---

# 19. How does the frontend communicate with the backend?

Using:

```text
HTTP/JSON
```

through:

```text
fetch()
```

The frontend does not show a WebSocket client in the uploaded implementation.

---

# 20. What is the experiment engine?

The experiment engine is the coordinator.

It connects:

```text
configuration
dataset
decoder
evaluation
storage
```

without implementing all algorithms itself.

---

# 21. What metrics do you use?

```text
Exact accuracy
Physical recovery
Bit accuracy
Logical accuracy
Training time
Inference time
Samples per second
```

---

# 22. How do you compare AI with baseline?

The project has a paired comparison.

The same held-out samples are evaluated by:

```text
baseline
```

and:

```text
AI-QEC
```

Then the difference is measured.

---

# 23. What statistics are used?

The project uses:

```text
bootstrap confidence intervals
paired permutation tests
multiple seeds
```

The formal stored study uses:

```text
seeds = 42, 43, 44
```

---

# 24. What is the strongest stored result?

The formal report shows:

```text
Physical noise = 0.10

Baseline logical success = 73.42%
AI-QEC logical success   = 84.17%

Absolute gain = +10.75 percentage points
```

This is a result for the tested configuration.

---

# 25. What is the largest logical-error reduction in the report?

At:

```text
physical noise = 0.05
```

the report gives approximately:

```text
51.71% logical-error reduction
```

Again, this is configuration-specific simulation evidence.

---

# 26. What are the main limitations?

The most important are:

```text
3-qubit code
classical simulation
three seeds in the formal paired study
main paired study uses zero measurement noise
controlled noise model
no physical quantum hardware
```

---

# 27. If asked "Is this production quantum software?"

A safe answer:

> "No. It is a research and simulation platform. It is useful for studying AI-based decoding and logical recovery, but it is not presented as production hardware QEC software or a universal quantum decoder."

---

# 28. If asked "Why not just use a lookup table?"

A useful answer:

> "A lookup decoder is useful for a small code, and the repository contains lookup approaches. The research question is whether a learned decoder can use temporal syndrome information and generalize within the simulated problem. That is why the project also explores Random Forest, Logistic Regression, MLP, and GRU-based approaches."

---

# 29. If asked "Where is AI actually used?"

Answer:

```text
QEC observations
       |
       v
ML features
       |
       v
Random Forest
       |
       v
Predicted correction
```

The physics, syndrome calculations, XOR logic, majority recovery, storage, and API handling are not themselves machine learning.

---

# 30. If asked "What did you personally learn?"

You can explain:

```text
Quantum error correction
Qiskit/Aer
syndrome decoding
machine learning
feature engineering
logical-aware targets
statistical evaluation
FastAPI
Next.js
experiment storage
scientific visualization
```

---

# 31. 30-second answer

> "I built a classical simulation platform for AI-based quantum error correction. The current system uses a 3-qubit bit-flip repetition code in Qiskit/Aer. Physical and measurement noise produce syndrome observations. I convert syndrome history and detection events into ML features and train a logical-target Random Forest to predict useful corrections. I then evaluate exact recovery and, more importantly, logical recovery. The platform includes a FastAPI backend, Next.js dashboard, experiment storage, scientific comparison, and Qiskit-based end-to-end trace."

---

# 32. Final interview diagram

```text
Qiskit/Aer
    |
    v
3-Qubit QEC
    |
    v
Noise
    |
    v
Syndrome History
    |
    v
Detection Events
    |
    v
Feature Engineering
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
Scientific Evaluation
    |
    +------> FastAPI
    |
    +------> Next.js
```
