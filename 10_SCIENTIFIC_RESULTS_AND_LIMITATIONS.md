# Quantum QEC AI — Scientific Results and Limitations

## 1. Important rule

The numbers in this document are **project-reported results from the uploaded source tree**.

They should not be generalized beyond the tested configuration.

---

# 2. Formal reported experiment

The stored scientific report uses:

```text
QEC code:
bit_flip_3

Rounds:
5

Training samples:
2000

Test samples per seed:
2000

Random Forest estimators:
100

Seeds:
42, 43, 44

Measurement noise:
0.0

Physical noise:
0.01
0.03
0.05
0.10
0.15
0.20
```

---

# 3. Reported logical success

| Physical noise | Baseline | AI-QEC |
|---:|---:|---:|
| 0.01 | 99.28% | 99.37% |
| 0.03 | 95.45% | 97.60% |
| 0.05 | 89.27% | 94.82% |
| 0.10 | 73.42% | 84.17% |
| 0.15 | 62.75% | 71.97% |
| 0.20 | 55.08% | 59.92% |

---

# 4. Reported absolute gain

| Physical noise | Gain |
|---:|---:|
| 0.01 | +0.08 percentage points |
| 0.03 | +2.15 percentage points |
| 0.05 | +5.55 percentage points |
| 0.10 | +10.75 percentage points |
| 0.15 | +9.22 percentage points |
| 0.20 | +4.83 percentage points |

The largest reported gain is:

```text
+10.75 percentage points
```

at:

```text
physical noise = 0.10
```

---

# 5. Reported logical-error reduction

| Physical noise | Error reduction |
|---:|---:|
| 0.01 | 11.63% |
| 0.03 | 47.25% |
| 0.05 | 51.71% |
| 0.10 | 40.44% |
| 0.15 | 24.74% |
| 0.20 | 10.76% |

The largest reported logical-error reduction is:

```text
51.71%
```

at:

```text
physical noise = 0.05
```

---

# 6. Statistical values in the report

The report includes paired permutation p-values:

| Physical noise | p-value |
|---:|---:|
| 0.01 | 1.00000 |
| 0.03 | 0.00040 |
| 0.05 | 0.00010 |
| 0.10 | 0.00010 |
| 0.15 | 0.00010 |
| 0.20 | 0.00630 |

Bootstrap intervals are also stored.

The report explains that:

```text
0.00010
```

is the resolution limit of the finite permutation procedure.

---

# 7. Three-seed limitation

The main formal paired study uses:

```text
42
43
44
```

Only three independent seeds are used.

That means the results are useful experimental evidence for the tested setup, but they should not be presented as universal proof.

---

# 8. QEC-code limitation

The project uses:

```text
3-qubit bit-flip repetition code
```

It does not implement:

```text
surface code
```

or:

```text
large-scale fault-tolerant QEC
```

---

# 9. Hardware limitation

The project uses:

```text
Qiskit Aer
```

on a classical computer.

It does not use:

```text
physical quantum hardware
```

Therefore, the results are simulation results.

---

# 10. Noise-model limitation

The project contains several noise experiments.

However, the strongest formal paired table uses:

```text
measurement noise = 0
```

So the main reported scientific table is primarily a physical-noise sweep.

The repository also contains combined physical + measurement-noise experiments.

Those should be reported separately.

---

# 11. Measurement-only limitation

The README reports a measurement-noise-only experiment with:

```text
physical noise = 0
```

and:

```text
100% logical success
```

This is not strong evidence of physical-error correction because there is no physical error in that condition.

---

# 12. Exact error vs logical success

A decoder can produce:

```text
wrong exact physical correction
```

while still producing:

```text
correct logical state
```

This is expected for a repetition code.

Therefore:

```text
exact accuracy
```

and:

```text
logical accuracy
```

must not be treated as identical.

---

# 13. Training-target limitation

The logical-target approach uses simulator ground truth to construct training targets.

That is valid for supervised simulation research.

But the target-building process depends on the controlled synthetic environment.

It should not be described as if the model magically learns labels without any ground truth during training.

---

# 14. Generalization limitation

The tested results do not establish performance across:

```text
different QEC codes
different circuit depths
different hardware
different noise channels
different measurement processes
different decoder architectures
different distributions
```

---

# 15. Scientific interpretation

The safest interpretation is:

> Within the tested 3-qubit bit-flip simulation configuration, the stored paired experiments report higher logical success for the logical-target Random Forest than the selected baseline across the listed physical-noise conditions.

That statement is deliberately limited to the experiment.

---

# 16. What would strengthen future research?

Potential directions include:

```text
more independent seeds
larger test sets
more QEC codes
surface-code experiments
more realistic noise channels
hardware experiments
decoder calibration
cross-noise generalization
cross-code generalization
larger model comparisons
```

---

# 17. One-minute scientific explanation

> "The project has a formal paired evaluation across six physical-noise levels using three seeds and 2,000 held-out samples per seed. The reported AI-QEC logical success is higher than the baseline in the tested configuration, with the largest absolute gain of 10.75 percentage points at 10% physical noise. However, the experiment is limited to a 3-qubit simulated repetition code, uses only three independent seeds, and the main formal table has zero measurement noise. So the result should be presented as configuration-specific simulation evidence rather than a universal QEC claim."
