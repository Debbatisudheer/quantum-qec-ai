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
# The Next.js frontend runs locally on ports 3000/3001 and
# in production on Vercel.
#
# The browser treats these as different origins, so the
# backend must explicitly allow the frontend origins.
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local frontend
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",

        # Production frontend
        "https://quantum-qec-ai-4.vercel.app",
    ],
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