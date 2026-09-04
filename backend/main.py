from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router


app = FastAPI(
    title="Quantum QEC AI API",
    description=(
        "Backend API for the "
        "AI-Powered Quantum Error Correction System"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
#
# The frontend is deployed separately from the FastAPI
# backend. CORS allows the browser-based frontend to call
# the backend API.
#
# Local development:
#   http://localhost:3000
#   http://localhost:3001
#   http://127.0.0.1:3000
#   http://127.0.0.1:3001
#
# Production:
#   Vercel frontend
#
# allow_credentials=False is intentional because the
# current frontend/backend communication does not require
# browser cookies or credentialed requests.
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "quantum-qec-ai",
    }


# ---------------------------------------------------------
# API routes
# ---------------------------------------------------------

app.include_router(router)