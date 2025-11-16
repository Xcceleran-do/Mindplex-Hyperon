"""FastAPI application entry point (Step 8).

Features:
 - Health endpoint (`GET /health`).
 - Recommendations endpoint mounted from `api.recommendations` router.
 - Admin endpoints (placeholder) from `api.admin`.
 - Lazy loading of latest trained model + dataset features during first recommendation request.

Run (development):
  uvicorn backend.app.main:app --reload

Environment variables respected:
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD for graph queries when filtering by creator.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.recommendations import router as recommendations_router
from .api.admin import router as admin_router
from .services.neo4j_client import health_check

app = FastAPI(title="Content Optimizer API", version="0.1.0")

@app.get("/health")
def health():  # simple synchronous endpoint
    try:
        h = health_check()
        return {"status": "ok", **h}
    except Exception as e:  # broad catch to avoid 500 spam during early startup
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# Mount routers
app.include_router(recommendations_router, tags=["recommendations"])
app.include_router(admin_router, tags=["admin"])

__all__ = ["app"]
