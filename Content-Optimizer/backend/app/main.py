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
import os
import threading
import time
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.recommendations import router as recommendations_router
from .api.strategy import router as strategy_router
from .api.admin import router as admin_router
from .services.neo4j_client import health_check

app = FastAPI(title="Content Optimizer API", version="0.1.0")

# CORS (frontend dev origin)
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/health")
def health():  # simple synchronous endpoint
    try:
        h = health_check()
        return {"status": "ok", **h}
    except Exception as e:  # broad catch to avoid 500 spam during early startup
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# Mount routers
app.include_router(recommendations_router, tags=["recommendations"])
app.include_router(strategy_router, tags=["strategy"])
app.include_router(admin_router, tags=["admin"])

__all__ = ["app"]


# --- Startup orchestration ---------------------------------------------------------

_log = logging.getLogger(__name__)

def _wait_for_neo4j(max_wait_sec: int = 60) -> None:
  from .services.neo4j_client import health_check
  start = time.time()
  last_err: Exception | None = None
  while time.time() - start < max_wait_sec:
    try:
      h = health_check()
      if h and isinstance(h, dict) and 'content_nodes' in h:
        return
    except Exception as e:  # pragma: no cover
      last_err = e
    time.sleep(2)
  if last_err:
    raise last_err

def _run_pipeline_on_startup():  # pragma: no cover (integration side-effect)
  try:
    _log.info("[startup] Waiting for Neo4j to be ready...")
    _wait_for_neo4j()
    from .api.admin import run_full_pipeline
    dataset_limit_env = os.getenv('DATASET_LIMIT')
    dataset_limit = int(dataset_limit_env) if dataset_limit_env else 5000
    _log.info("[startup] Running full pipeline (dataset_limit=%s)...", dataset_limit)
    res = run_full_pipeline(dataset_limit=dataset_limit)
    _log.info("[startup] Pipeline completed: %s", {k: res.get(k) for k in ['status','dataset','models_dir']})
  except Exception as e:
    _log.exception("[startup] Pipeline failed: %s", e)


@app.on_event("startup")
def startup_orchestrate():  # pragma: no cover
  run_on_start = os.getenv('RUN_PIPELINE_ON_STARTUP', 'true').lower() in ('1','true','yes')
  if run_on_start:
    t = threading.Thread(target=_run_pipeline_on_startup, daemon=True)
    t.start()
