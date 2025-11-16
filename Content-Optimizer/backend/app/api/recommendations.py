"""Recommendations API (Step 8).

Logic:
 - Lazily load latest trained model (LightGBM) and dataset parquet to build an in-memory feature & score cache.
 - Endpoint allows optional filtering by creatorId (using Neo4j live query to fetch that creator's content IDs).
 - Returns topK items sorted by score descending.
 - Explanation currently includes raw score and source fields; can be expanded.

Routes:
 - POST /recommendations        (body: RecommendationRequest)
 - GET  /recommendations        (query: creatorId, topK) convenience for browsers
 - POST /recommendations/refresh (rebuild cache)
 - GET  /recommendations/refresh (same as POST refresh)

NOTE: This is a simple baseline recommender. For real-time personalization, you'd compute user/segment embeddings and re-score dynamically.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import (
    RecommendationRequest,
    RecommendationItem,
    RecommendationResponse,
)
from backend.app.services.neo4j_client import run_read
from backend.app.services.ranker import load_latest_model
import math

router = APIRouter()

_INIT_LOCK = threading.Lock()
_MODEL = None
_MODEL_META: Dict[str, Any] = {}
_CACHE: List[Dict[str, Any]] = []  # list of {contentId,title,score,features}
_DATASET_PATH: Optional[Path] = None


def _find_dataset() -> Optional[Path]:
    candidates = [
        Path('data/dataset.parquet'),
        Path('data/test_dataset.parquet'),
        Path('data/notebook_dataset.parquet'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _build_features_df(limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch content and engagement aggregates from Neo4j (mirrors training assemble logic)."""
    params = {"limit": limit} if limit else {}
    cypher = (
        "MATCH (c:Content) OPTIONAL MATCH (s:AudienceSegment)-[e:ENGAGED_WITH]->(c) "
        "WITH c, coalesce(sum(e.views),0) AS views, coalesce(sum(e.likes),0) AS likes, coalesce(avg(e.watch_time),0) AS avg_watch_time "
        "RETURN c.contentId AS contentId, c.title AS title, c.lengthSec AS lengthSec, c.text_embedding AS text_embedding, c.embedding AS gsage_embedding, views, likes, avg_watch_time "
    )
    rows = run_read(cypher, params)
    if not rows:
        return pd.DataFrame([])
    # infer dims
    def _coerce_emb(x: Any) -> List[float]:
        if x is None:
            return []
        if isinstance(x, list):
            return [float(v) for v in x]
        return [float(v) for v in list(x)]

    text_dim = next((len(_coerce_emb(r.get('text_embedding'))) for r in rows if r.get('text_embedding')), 384)
    graph_dim = next((len(_coerce_emb(r.get('gsage_embedding') or r.get('embedding'))) for r in rows if (r.get('gsage_embedding') or r.get('embedding'))), 64)

    recs = []
    for r in rows:
        te = _coerce_emb(r.get('text_embedding')) or [0.0] * text_dim
        ge = _coerce_emb(r.get('gsage_embedding') or r.get('embedding')) or [0.0] * graph_dim
        lengthSec = float(r.get('lengthSec') or 0.0)
        views = float(r.get('views') or 0.0)
        likes = float(r.get('likes') or 0.0)
        watch_time = float(r.get('avg_watch_time') or 0.0)
        features = ge + te + [lengthSec, views, likes, watch_time]
        recs.append({
            'contentId': r.get('contentId'),
            'title': r.get('title'),
            'features': features,
        })
    return pd.DataFrame(recs)


def _initialize_cache():
    global _MODEL, _MODEL_META, _CACHE, _DATASET_PATH
    if _CACHE:  # already initialized
        return
    with _INIT_LOCK:
        if _CACHE:  # double-check
            return
        _MODEL, _MODEL_META = load_latest_model('./models')
        # Prefer cached dataset parquet if exists (faster), else rebuild from Neo4j
        _DATASET_PATH = _find_dataset()
        if _DATASET_PATH and _DATASET_PATH.exists():
            try:
                df = pd.read_parquet(_DATASET_PATH)
                # Expect columns contentId,title,features; if not rebuild
                if not {'contentId','title','features'}.issubset(df.columns):
                    df = _build_features_df()
            except Exception:
                df = _build_features_df()
        else:
            df = _build_features_df()
        if df.empty:
            _CACHE = []
            return
        X = np.vstack(df['features'].values)
        if _MODEL is not None:
            scores = _MODEL.predict(X)
        else:
            scores = np.zeros(len(df))
        _CACHE = [
            {
                'contentId': cid,
                'title': title,
                'score': float(score),
                'features': feat,
            }
            for cid, title, score, feat in zip(df['contentId'], df['title'], scores, df['features'])
        ]
        # sort descending score
        _CACHE.sort(key=lambda r: r['score'], reverse=True)


def _filter_by_creator(creator_id: str) -> List[str]:
    cypher = (
        "MATCH (cr:Creator {creatorId: $creatorId})-[:CREATED]->(c:Content) "
        "RETURN c.contentId AS contentId"
    )
    rows = run_read(cypher, {"creatorId": creator_id})
    return [r['contentId'] for r in rows]


def _cosine(a: List[float], b: List[float]) -> float:
    """Robust cosine similarity for list/array inputs (empty-safe)."""
    if a is None or b is None:
        return 0.0
    # Convert numpy arrays to lists for uniform handling
    if hasattr(a, 'tolist'):
        a = a.tolist()
    if hasattr(b, 'tolist'):
        b = b.tolist()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    dot = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        sum_a += x * x
        sum_b += y * y
    na = math.sqrt(sum_a) or 1.0
    nb = math.sqrt(sum_b) or 1.0
    return dot / (na * nb)


def _compute_recommendations(creator_id: str, topK: int) -> RecommendationResponse:
    """Compute recommendations, augmenting explanation with nearest examples + top features.

    Top features are indexes of highest model feature importances (best-effort).
    Nearest examples are top 3 other cached items by cosine similarity of feature vectors.
    """
    _initialize_cache()
    if not _CACHE:
        raise HTTPException(status_code=503, detail='recommendation cache empty (no data)')
    items = _CACHE
    if creator_id:
        creator_content_ids = set(_filter_by_creator(creator_id))
        filtered = [r for r in items if r['contentId'] in creator_content_ids]
        if filtered:
            items = filtered
    selected = items[: max(topK, 1)]

    # feature importances (if available)
    top_features: List[int] = []
    if _MODEL is not None and hasattr(_MODEL, 'feature_importances_'):
        importances = list(getattr(_MODEL, 'feature_importances_'))
        if importances:
            # take top 5 indices by importance
            sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
            top_features = sorted_idx[:5]

    # precompute nearest examples for each selected item
    rec_items: List[RecommendationItem] = []
    for r in selected:
        feats = r['features']
        # compute similarity to other cache items (exclude self)
        sims = []
        for other in _CACHE:
            if other['contentId'] == r['contentId']:
                continue
            sim = _cosine(feats, other['features'])
            sims.append({
                'contentId': other['contentId'],
                'title': other['title'],
                'similarity': round(sim, 4),
                'score': other['score'],
            })
        sims.sort(key=lambda d: d['similarity'], reverse=True)
        nearest_examples = sims[:3]
        explanation = {
            'score': r['score'],
            'model': _MODEL_META.get('version'),
            'nearest_examples': nearest_examples,
            'top_features': top_features,
        }
        rec_items.append(
            RecommendationItem(
                contentId=r['contentId'],
                title=r['title'],
                score=r['score'],
                explanation=explanation,
            )
        )
    return RecommendationResponse(recommendations=rec_items, modelVersion=_MODEL_META.get('version'))

@router.post('/recommendations', response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    return _compute_recommendations(req.creatorId or '', req.topK or 10)

@router.get('/recommendations', response_model=RecommendationResponse, summary='Convenience GET for recommendations')
async def recommend_get(creatorId: str = '', topK: int = 10):
    return _compute_recommendations(creatorId, topK)

def _refresh_cache() -> Dict[str, Any]:
    with _INIT_LOCK:
        global _CACHE
        _CACHE = []
    _initialize_cache()
    return {'status': 'refreshed', 'count': len(_CACHE)}

@router.post('/recommendations/refresh', summary='Refresh in-memory recommendation cache')
async def refresh_post():
    return _refresh_cache()

@router.get('/recommendations/refresh', summary='Refresh (GET alias)')
async def refresh_get():
    return _refresh_cache()

