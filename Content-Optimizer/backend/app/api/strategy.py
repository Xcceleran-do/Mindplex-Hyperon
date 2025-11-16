"""Strategy API: Recommend optimal format, platform, and length band.

Heuristic approach (data-light friendly):
- Aggregate content items (optionally for a creator) with engagement metrics.
- Compute a simple engagement score per item: views + avg_watch_time (normalized per group when possible).
- Group by (platform, format), compute mean score, pick top-K groups.
- For each group, compute a length band as [p25, p75] of lengthSec among top items in that group.
- Return a list of StrategyItem plus a simple summary.

If there is limited data, the endpoint returns best-effort suggestions.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.services.neo4j_client import run_read
from backend.app.models.schemas import StrategyRequest, StrategyResponse, StrategyItem

router = APIRouter()


def _fetch_content_rows(creator_id: str | None = None) -> List[Dict[str, Any]]:
    if creator_id:
        cypher = (
            "MATCH (cr:Creator {creatorId: $creatorId})-[:CREATED]->(c:Content) "
            "OPTIONAL MATCH (s:AudienceSegment)-[e:ENGAGED_WITH]->(c) "
            "WITH c, coalesce(sum(e.views),0) AS views, coalesce(sum(e.likes),0) AS likes, coalesce(avg(e.watch_time),0) AS avg_watch_time "
            "RETURN c.contentId AS contentId, c.title AS title, c.lengthSec AS lengthSec, c.format AS format, c.platform AS platform, views, likes, avg_watch_time "
        )
        params = {"creatorId": creator_id}
    else:
        cypher = (
            "MATCH (c:Content) "
            "OPTIONAL MATCH (s:AudienceSegment)-[e:ENGAGED_WITH]->(c) "
            "WITH c, coalesce(sum(e.views),0) AS views, coalesce(sum(e.likes),0) AS likes, coalesce(avg(e.watch_time),0) AS avg_watch_time "
            "RETURN c.contentId AS contentId, c.title AS title, c.lengthSec AS lengthSec, c.format AS format, c.platform AS platform, views, likes, avg_watch_time "
        )
        params = {}
    return run_read(cypher, params)


def _length_band(lengths: List[float]) -> Optional[list[int]]:
    if not lengths:
        return None
    a = np.asarray(lengths, dtype=float)
    p25 = int(np.nanpercentile(a, 25))
    p75 = int(np.nanpercentile(a, 75))
    if p25 == p75:
        return [max(p25 - 30, 0), p75 + 30]
    return [p25, p75]


@router.get('/strategy', response_model=StrategyResponse, summary='Suggest optimal format/platform/length band')
async def strategy_get(creatorId: str = '', topK: int = 3):
    return await strategy_post(StrategyRequest(creatorId=creatorId, topK=topK))


@router.post('/strategy', response_model=StrategyResponse)
async def strategy_post(req: StrategyRequest):
    rows = _fetch_content_rows(req.creatorId or None)
    if not rows:
        return StrategyResponse(items=[], summary='No data available to compute strategy')
    df = pd.DataFrame(rows)
    # Coerce types and fill missing
    for col in ['views', 'likes', 'avg_watch_time', 'lengthSec']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in ['format', 'platform']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # Simple engagement score: views + avg_watch_time (seconds)
    # For robustness, normalize per platform to avoid skew.
    if 'platform' in df.columns:
        df['_platform_mean'] = df.groupby('platform')['views'].transform('mean').replace(0, 1.0)
        df['score'] = (df['views'] / df['_platform_mean']) + df['avg_watch_time']
    else:
        df['score'] = df['views'] + df['avg_watch_time']

    if 'format' not in df.columns:
        df['format'] = 'unknown'
    if 'platform' not in df.columns:
        df['platform'] = 'unknown'

    # Group by platform+format and compute mean score
    grp = df.groupby(['platform', 'format']).agg(
        mean_score=('score', 'mean'),
        n=('contentId', 'count')
    ).reset_index()
    if grp.empty:
        return StrategyResponse(items=[], summary='Not enough data to compute strategy')

    grp = grp.sort_values('mean_score', ascending=False).head(max(1, req.topK))

    items: List[StrategyItem] = []
    for _, row in grp.iterrows():
        plat = str(row['platform'])
        fmt = str(row['format'])
        subset = df[(df['platform'] == plat) & (df['format'] == fmt)].copy()
        # Take top subset within group to estimate length band
        subset = subset.sort_values('score', ascending=False).head(20)
        length_band = _length_band(subset['lengthSec'].tolist())
        examples = subset['contentId'].head(3).tolist()
        items.append(StrategyItem(
            platform=plat,
            format=fmt,
            score=float(row['mean_score']),
            lengthRangeSec=length_band,
            examples=examples,
        ))

    summary = (
        f"Top suggestion: {items[0].format} on {items[0].platform} "
        + (f"around {items[0].lengthRangeSec[0]}-{items[0].lengthRangeSec[1]}s" if items[0].lengthRangeSec else "with typical length from peers")
    ) if items else 'No suggestions'

    return StrategyResponse(items=items, summary=summary)
