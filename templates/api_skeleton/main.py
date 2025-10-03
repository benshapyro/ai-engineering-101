"""
FastAPI skeleton for production LLM applications.

Run with:
    uvicorn templates.api_skeleton.main:app --reload

Or:
    python -m uvicorn templates.api_skeleton.main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from .routers import generate, rag, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="LLM Application API",
    description="Production-ready API for LLM applications",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()

    # Get request info
    path = request.url.path
    method = request.method

    logger.info(f"{method} {path} - Started")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    logger.info(f"{method} {path} - Completed in {duration_ms:.2f}ms (status: {response.status_code})")

    # Add timing header
    response.headers["X-Process-Time-Ms"] = str(int(duration_ms))

    return response


# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(generate.router, prefix="/api/v1", tags=["generation"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Application API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/healthz",
            "metrics": "/api/v1/metrics",
            "generate": "/api/v1/generate",
            "rag_query": "/api/v1/rag/query"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
