# Quantum QEC AI — Beginner Guide to the Quantum and QEC Side

## 1. Start with the basic idea

Quantum Error Correction means:

> Protect useful quantum information from errors.

This project uses a very small QEC example so the complete process can be understood.

The code is:

```text
3-qubit bit-flip repetition code
```

---

# 2. Logical information

The project represents:

```text
Logical 0 -> 000
Logical 1 -> 111
```

These are called codewords.

The three physical qubits together represent one logical bit.

---

# 3. Why three qubits?

If one physical qubit changes, the other two can still indicate the original majority.

Example:

```text
Original:

000

One error:

010
```

There are still two zeroes.

So majority voting can recover logical 0.

For logical 1:

```text
111

One error:

101
```

There are still two ones.

So majority voting recovers logical 1.

---

# 4. What is a bit-flip error?

A bit flip is represented by an X operation.

Classically, think:

```text
0 -> 1
1 -> 0
```

For three qubits:

```text
000
```

could become:

```text
100
010
001
```

after one error.

---

# 5. QEC does not directly look at the hidden error

The decoder should use syndrome information.

The current 3-qubit code uses two stabilizer checks.

Conceptually:

```text
S1 = q0 XOR q1
S2 = q1 XOR q2
```

The two syndrome bits provide information about the error pattern.

---

# 6. Syndrome examples

Conceptually:

```text
No error
000
 |
 v
00
```

A different physical error produces a different syndrome.

The exact mapping is implemented by the project code.

The important idea is:

```text
Physical error
      |
      v
Syndrome
```

---

# 7. Repeated QEC rounds

The project can repeat syndrome observation.

Default:

```text
5 rounds
```

Example:

```text
Round 1 -> 00
Round 2 -> 10
Round 3 -> 10
Round 4 -> 01
Round 5 -> 00
```

The AI can use the entire history.

---

# 8. Why history matters

One syndrome value is only one observation.

A sequence gives temporal information:

```text
00
10
10
01
00
```

This tells the model how the observed syndrome changed over time.

---

# 9. Detection events

A detection event records syndrome changes.

Formula:

```text
event[t] =
    syndrome[t-1] XOR syndrome[t]
```

For example:

```text
Previous = 00
Current  = 10

Event = 10
```

So the decoder can receive:

```text
Syndrome history
+
Detection history
```

---

# 10. Measurement noise

The project can add noise to syndrome observations.

The flow is:

```text
Perfect syndrome
       |
       v
Measurement noise
       |
       v
Observed syndrome
```

This means the AI may not always receive the perfect syndrome.

That makes decoding harder.

---

# 11. Physical noise vs measurement noise

These are different.

## Physical noise

Changes the physical qubit error state.

```text
Qubit
  |
  v
Physical bit flip
```

## Measurement noise

Changes what the observer sees.

```text
Correct syndrome
      |
      v
Noisy measurement
      |
      v
Observed syndrome
```

So:

```text
Physical noise
= problem in the quantum state

Measurement noise
= problem in the observation
```

---

# 12. Qiskit simulation

The project uses:

```text
Qiskit
Qiskit Aer
```

Aer runs the circuits on the classical computer.

So:

```text
Python
  |
  v
Qiskit circuit
  |
  v
Aer simulator
  |
  v
Measurement results
```

There is no physical quantum processor in the uploaded project.

---

# 13. Basic Qiskit simulator

The wrapper is:

```text
quantum/simulator.py
```

Class:

```text
QuantumSimulator
```

It creates:

```text
AerSimulator()
```

and runs a Qiskit circuit.

---

# 14. Repeated quantum QEC circuit

The project has:

```text
quantum/repeated_qec.py
```

Class:

```text
RepeatedQuantumQEC
```

For every round it creates two syndrome ancillas.

Conceptually:

```text
Data qubits:
q0
q1
q2

Round 1:
ancilla S1
ancilla S2

Round 2:
ancilla S1
ancilla S2

...
```

The implementation uses separate ancillas for rounds instead of resetting the same ancillas.

---

# 15. Stabilizer checks

The documented checks are:

```text
S1 = Z0 Z1
S2 = Z1 Z2
```

The circuit extracts the syndrome using CNOT interactions.

Conceptually:

```text
q0 ----CNOT----\
                > S1 ancilla
q1 ----CNOT----/

q1 ----CNOT----\
                > S2 ancilla
q2 ----CNOT----/
```

---

# 16. Error transitions

The repeated circuit does not blindly reapply the complete accumulated error every round.

It compares:

```text
previous error state
current error state
```

and applies only changed X-error parity.

Example:

```text
Previous = 000
Current  = 101
```

Apply:

```text
X(q0)
X(q2)
```

---

# 17. Final measurement

After correction, the data qubits are measured.

The result is a physical 3-bit state.

Example:

```text
000
```

or:

```text
111
```

or an intermediate state such as:

```text
100
```

---

# 18. Logical recovery

The project uses majority voting.

For example:

```text
100 -> logical 0
```

because there is only one `1`.

And:

```text
110 -> logical 1
```

because there are two `1`s.

---

# 19. Logical success

The final question is:

```text
Did we recover the original logical state?
```

Example:

```text
Original logical state = 0
Corrected physical state = 100

Majority = 0

Logical success = YES
```

---

# 20. Exact correction is not always necessary

This is one of the most important ideas in the project.

Suppose:

```text
Actual error = 100
Predicted correction = 000
```

The physical error was not exactly corrected.

But:

```text
100
```

still has majority zero.

Therefore:

```text
Logical recovery = 0
```

and logical success can still be true.

So:

```text
Exact physical recovery
!=
Logical recovery
```

---

# 21. Why the project uses logical success as the main metric

QEC exists to protect the logical information.

Therefore, the most important question is:

```text
Did the logical information survive?
```

rather than only:

```text
Did the AI reproduce the hidden physical error exactly?
```

---

# 22. Beginner quantum flow

Remember this:

```text
Logical 0/1
     |
     v
Encode into 3 qubits
     |
     v
Physical error
     |
     v
Syndrome measurement
     |
     v
Observed syndrome
     |
     v
AI decoder
     |
     v
Correction
     |
     v
Final measurement
     |
     v
Majority vote
     |
     v
Logical success/failure
```

---

# 23. What the project does not claim

It does not claim:

```text
physical quantum hardware
```

It does not implement:

```text
surface code
```

It does not claim:

```text
universal quantum error correction
```

It is a controlled simulation/research environment.

---

# 24. One-minute interview explanation

> "I used a 3-qubit bit-flip repetition code and Qiskit Aer to simulate QEC on a classical computer. The logical states are 000 and 111. Physical bit-flip noise creates errors, and repeated syndrome measurements provide observations. I also model measurement noise. The decoder receives syndrome history and detection events, predicts a correction, and the corrected state is evaluated using majority-based logical recovery."

---

# 25. Final mental picture

```text
      LOGICAL INFORMATION
              |
              v
       3-QUBIT ENCODING
              |
              v
       PHYSICAL ERRORS
              |
              v
        QEC SYNDROME
              |
              v
       OBSERVED HISTORY
              |
              v
          AI DECODER
              |
              v
         CORRECTION
              |
              v
        FINAL STATE
              |
              v
      LOGICAL RECOVERY
```
