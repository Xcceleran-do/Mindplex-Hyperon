"""End-to-end and smoke tests (Step 11).

These tests validate that:
 - FastAPI app mounts /health and /recommendations endpoints.
 - Recommendation responses have expected schema fields.
 - Dataset assembly returns a DataFrame (possibly empty but with expected columns).
 - Training script produces a model artifact when dataset is non-empty (skipped if empty).

Note: The tests assume Neo4j is running locally with embeddings written. If not, recommendations
may return zero scores or an empty list; that's acceptable for a smoke test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import importlib

import pytest

# Ensure repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    # neo4j fields may be None if server not reachable, but keys should exist
    for key in ["neo4j_version", "content_nodes", "has_embeddings"]:
        assert key in data


def test_recommendations_get(client):
    r = client.get("/recommendations", params={"topK": 5})
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    # If we have items, validate structure
    if data["recommendations"]:
        item = data["recommendations"][0]
        for field in ["contentId", "score", "explanation"]:
            assert field in item
        assert "modelVersion" in data


def test_recommendations_post(client):
    r = client.post("/recommendations", json={"creatorId": "", "topK": 3})
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 3


def test_dataset_assembly():
    # Import assemble_dataset directly
    mod = importlib.import_module("backend.train.prepare_dataset")
    assemble_dataset = getattr(mod, "assemble_dataset")
    df = assemble_dataset(limit=20)
    # DataFrame should have expected columns even if empty
    assert list(df.columns) == ["contentId", "title", "features", "target"]
    # If non-empty, features should be list-like
    if not df.empty:
        row = df.iloc[0]
        assert isinstance(row["features"], list)


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skip training in CI for speed")
def test_training_script(tmp_path):
    """Run training script against existing dataset or on-the-fly assembled dataset.

    Skips if assembled dataset is empty (no content nodes)."""
    mod = importlib.import_module("backend.train.prepare_dataset")
    assemble_dataset = getattr(mod, "assemble_dataset")
    df = assemble_dataset(limit=50)
    if df.empty:
        pytest.skip("No data available for training")
    dataset_path = tmp_path / "dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    train_mod = importlib.import_module("backend.train.train_ranker")
    train_main = getattr(train_mod, "main")
    models_dir = tmp_path / "models"
    train_main(input_path=str(dataset_path), out_dir=str(models_dir))
    artifacts = list(models_dir.glob("*.joblib"))
    assert artifacts, "Expected at least one model artifact (.joblib)"
