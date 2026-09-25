# Quantum QEC AI — Next.js Frontend Guide

## 1. Frontend purpose

The frontend is a Mission/Research-style dashboard for the AI-QEC system.

Technology:

```text
Next.js 16.3.4
React 19.2.8
TypeScript
Tailwind CSS 4
```

---

# 2. Main files

```text
frontend/
|
+-- app/
|   +-- page.tsx
|   +-- layout.tsx
|   +-- globals.css
|
+-- component/
|   +-- SiteHeader.tsx
|   +-- SiteFooter.tsx
|
+-- package.json
+-- next.config.ts
```

---

# 3. Main page

The main dashboard is:

```text
frontend/app/page.tsx
```

It is a client component:

```text
"use client"
```

because it uses interactive React state.

---

# 4. Main frontend state

The page keeps state for:

```text
experiment configuration
current result
experiment history
visualization data
scientific evaluation
loading status
errors
trace
filters
sorting
```

---

# 5. Experiment configuration

The UI exposes:

```text
QEC code
number of qubits
logical state
rounds
physical noise
measurement noise
training samples
test samples
Random Forest estimators
seed
```

---

# 6. Run experiment button

The frontend sends:

```text
POST /simulate
```

with the configuration as JSON.

Conceptually:

```text
UI
 |
 | JSON
 v
POST /simulate
 |
 v
FastAPI
 |
 v
ExperimentEngine
 |
 v
Result
 |
 v
UI
```

---

# 7. Experiment history

The UI loads:

```text
GET /experiments
```

It can then display stored experiment summaries.

The interface also supports filtering and sorting.

---

# 8. Visualization data

The UI calls:

```text
GET /experiments/visualization
```

The backend returns frontend-ready chart data.

---

# 9. Scientific evaluation

The UI calls:

```text
GET /scientific/evaluation
```

The response includes:

```text
metadata
result rows
summary
```

The frontend displays scientific metrics and charts.

---

# 10. Single trace

The UI calls:

```text
POST /simulate/trace
```

This is the most educational part of the application.

The trace can show:

```text
Encoded state
Corrupted state
Physical error history
Perfect syndrome history
Observed syndrome history
Detection events
Predicted correction
Corrected state
Logical recovery
```

---

# 11. Why the trace is useful

Instead of only showing:

```text
Accuracy = 84%
```

the trace explains one sample.

For example:

```text
Encoded:
000

Actual error:
100

Observed syndrome:
...

AI correction:
000

Corrected state:
100

Recovered logical:
0

Logical success:
true
```

This makes the AI decision easier to understand.

---

# 12. Scientific charts

The page contains custom SVG chart components.

Examples include:

```text
ScientificLineChart
ScientificGainChart
```

The frontend can display:

```text
baseline vs AI logical success
AI absolute gain
physical-noise trends
scientific evaluation tables
```

---

# 13. API base URL

The frontend reads:

```text
NEXT_PUBLIC_API_BASE_URL
```

from:

```text
frontend/.env.local
```

The uploaded file currently contains a Render backend URL.

If the variable is not set, the source falls back to:

```text
http://127.0.0.1:8001
```

---

# 14. Local architecture

A typical local setup is:

```text
Browser
  |
  v
Next.js
localhost:3000
  |
  | HTTP
  v
FastAPI
localhost:8001
  |
  v
QEC/AI system
```

The exact port can be changed according to the environment.

---

# 15. Production-style architecture in the source

The source is prepared for separated frontend/backend hosting:

```text
Browser
  |
  v
Hosted Next.js frontend
  |
  | HTTPS API request
  v
Hosted FastAPI backend
  |
  v
AI-QEC system
```

The `.env.local` value in the uploaded source points to a Render-hosted backend.

This is a configuration fact from the uploaded project, not a claim that the endpoint was re-verified during this analysis.

---

# 16. Header

The header identifies the application as:

```text
AI / Quantum Research Platform
```

and:

```text
AI-Powered Quantum Error Correction
```

---

# 17. Footer

The footer identifies the project owner text as:

```text
Sudheer Debbati
```

---

# 18. Frontend communication style

The current frontend uses:

```text
fetch()
```

and normal HTTP.

The uploaded source does not show a WebSocket client.

So the current frontend communication is:

```text
REST/HTTP + JSON
```

not WebSocket.

---

# 19. Beginner mental model

Think of the frontend as the control panel.

```text
You
 |
 | choose experiment
 v
Dashboard
 |
 | send request
 v
Backend
 |
 | run QEC/AI
 v
Result
 |
 v
Dashboard
 |
 +--> metrics
 +--> history
 +--> charts
 +--> scientific results
 +--> trace
```

---

# 20. Interview answer

> "The frontend is a Next.js and TypeScript dashboard. It lets the user configure and run the QEC experiment, load stored experiments, view performance metrics, inspect scientific evaluation results, and generate a single detailed AI/QEC trace. It communicates with FastAPI through HTTP and JSON using a configurable API base URL."
