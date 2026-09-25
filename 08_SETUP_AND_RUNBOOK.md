# Quantum QEC AI — Simple Setup and Runbook

## 1. Important first point

The uploaded project contains a Windows-oriented Python virtual environment and a frontend `node_modules` directory.

For a clean setup, it is better to recreate the environments rather than depend on the bundled generated environment.

---

# 2. Backend requirements

The repository lists:

```text
qiskit
qiskit-aer
numpy
pandas
scikit-learn
matplotlib
pytest
fastapi
uvicorn
```

in:

```text
requirements.txt
```

---

# 3. Recommended Python setup on Windows

Open PowerShell in the project root.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify Python:

```powershell
python --version
```

Verify Qiskit:

```powershell
python -c "import qiskit; print(qiskit.__version__)"
```

Verify Aer:

```powershell
python -c "import qiskit_aer; print('Aer OK')"
```

---

# 4. Backend start

The project README documents a backend port of:

```text
8001
```

Start:

```powershell
python -m uvicorn backend.main:app --port 8001
```

Health check:

```text
http://127.0.0.1:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "quantum-qec-ai"
}
```

---

# 5. Frontend setup

Open a second PowerShell window.

Go to:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Run:

```powershell
npm run dev
```

Open the address printed by Next.js.

The README commonly uses:

```text
http://localhost:3000
```

The project history also documents a case where port `3001` was used because `3000` was occupied.

---

# 6. Frontend API configuration

Create or edit:

```text
frontend/.env.local
```

For local backend use:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001
```

Restart Next.js after changing environment variables.

---

# 7. Basic application test

Start backend.

Then open:

```text
GET /health
```

Then start frontend.

Then:

1. configure an experiment
2. click run
3. wait for the experiment
4. inspect the metrics
5. load experiment history
6. run a trace
7. inspect scientific results

---

# 8. Running the backend experiment directly

The main API call is:

```text
POST /simulate
```

Example JSON:

```json
{
  "qec_code": "bit_flip_3",
  "num_qubits": 3,
  "logical_state": 0,
  "rounds": 5,
  "physical_noise_probability": 0.10,
  "measurement_noise_probability": 0.10,
  "training_samples": 5000,
  "test_samples": 1000,
  "random_forest_estimators": 100,
  "seed": 42
}
```

---

# 9. Running a trace

Use:

```text
POST /simulate/trace
```

Example:

```json
{
  "qec_code": "bit_flip_3",
  "num_qubits": 3,
  "logical_state": 0,
  "rounds": 5,
  "physical_noise_probability": 0.10,
  "measurement_noise_probability": 0.10,
  "training_samples": 5000,
  "random_forest_estimators": 100,
  "seed": 42
}
```

---

# 10. Run tests

From the project root:

```powershell
pytest
```

The repository contains a large test suite.

Some tests require:

```text
Qiskit
Qiskit Aer
PyTorch
scikit-learn
FastAPI
```

depending on the test.

---

# 11. Important performance note

The trace endpoint trains a Random Forest before generating the trace.

Therefore:

```text
POST /simulate/trace
```

is not just a cheap static display operation.

It performs actual work.

---

# 12. Experiment result files

Results appear under:

```text
experiments/results/
```

Do not confuse these with:

```text
experiments/paired_results/
```

The first stores individual experiment results.

The second contains paired scientific comparison data.

---

# 13. Scientific evaluation

The backend reads:

```text
experiments/paired_results/paired_baseline_ai_results.json
```

and exposes:

```text
/scientific/evaluation
/scientific/results
/scientific/summary
```

---

# 14. Build frontend

The package defines:

```text
npm run build
```

The uploaded frontend includes a complete Next.js project.

---

# 15. Troubleshooting: Qiskit not found

If you see:

```text
ModuleNotFoundError: No module named 'qiskit'
```

check:

```powershell
where.exe python
```

Make sure the active Python is:

```text
<project>\.venv\Scripts\python.exe
```

Then:

```powershell
pip install -r requirements.txt
```

---

# 16. Troubleshooting: backend port

If 8001 is occupied, choose another port.

Example:

```powershell
python -m uvicorn backend.main:app --port 8002
```

Then update:

```text
NEXT_PUBLIC_API_BASE_URL
```

accordingly.

---

# 17. Troubleshooting: frontend port

If 3000 is occupied, Next.js can use another available port.

For example:

```powershell
npm run dev -- -p 3001
```

The frontend port and backend port are independent.

---

# 18. Troubleshooting: frontend cannot reach backend

Check:

```text
NEXT_PUBLIC_API_BASE_URL
```

Then check:

```text
backend /health
```

Example:

```text
http://127.0.0.1:8001/health
```

If health works but the frontend does not, check:

```text
CORS
API URL
frontend restart
```

---

# 19. Clean rebuild

If the frontend has dependency problems:

```powershell
cd frontend
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install
npm run build
```

Only use this if necessary because deleting the lock file changes dependency resolution.

Prefer:

```powershell
npm ci
```

when the lock file is known to be valid.

---

# 20. Important source-control note

The uploaded project contains a working tree with many tracked files modified relative to the repository HEAD.

Static inspection showed:

```text
199 files changed
```

relative to the HEAD commit.

A large part of this appears consistent with line-ending/working-tree changes, but the ZIP should be treated as a **working-tree snapshot**, not automatically as a clean commit snapshot.

Do not blindly discard those changes.

---

# 21. Environment-size note

The uploaded archive contains generated environments.

Approximate extracted sizes observed during analysis:

```text
.venv              ~1.2 GB
frontend/node_modules ~486 MB
frontend/.next       ~98 MB
```

These are generated dependencies/build artifacts.

For source sharing, they are usually unnecessary.

The source itself is much smaller.

---

# 22. Recommended clean project archive

For future sharing, exclude:

```text
.venv/
frontend/node_modules/
frontend/.next/
__pycache__/
.pytest_cache/
```

Keep:

```text
source code
tests
requirements.txt
package.json
package-lock.json
README.md
experiment results
configuration
```

unless a generated artifact is specifically needed.

---

# 23. Beginner startup order

Use this order:

```text
1. Open project
       |
2. Activate Python .venv
       |
3. Install/verify dependencies
       |
4. Start FastAPI
       |
5. Check /health
       |
6. Start Next.js
       |
7. Open browser
       |
8. Run experiment
       |
9. Inspect result
       |
10. Run trace
```

---

# 24. Golden demo flow

For a simple demonstration:

```text
Open dashboard
     |
     v
Choose 3-qubit bit-flip
     |
     v
5 rounds
     |
     v
10% physical noise
     |
     v
10% measurement noise
     |
     v
Run experiment
     |
     v
Show logical accuracy
     |
     v
Show experiment history
     |
     v
Run trace
     |
     v
Explain:
syndrome -> AI -> correction -> recovery
```

---

# 25. One-minute run explanation

> "The backend is a FastAPI service. I activate the Python virtual environment, install the Qiskit/ML dependencies, and start Uvicorn on port 8001. The Next.js frontend runs separately and uses `NEXT_PUBLIC_API_BASE_URL` to call the backend. From the dashboard I can run an experiment, inspect stored results, view scientific evaluation, and generate a detailed single-sample QEC trace."
