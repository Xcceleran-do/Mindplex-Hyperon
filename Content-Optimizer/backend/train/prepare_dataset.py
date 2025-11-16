"""Prepare dataset: fetch content embeddings and numeric features from Neo4j and assemble a
training dataset for the hybrid ranker.

Outputs a Parquet file with columns:
 - contentId (str)
 - title (str)
 - features (list[float])  # concatenated graph + text + numeric
 - target (float)          # engagement score computed from views/watch_time

Usage:
  python backend/train/prepare_dataset.py --out data/dataset.parquet
"""
from __future__ import annotations

import sys
import pathlib
import argparse
import json
from typing import List, Any, Dict
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so `backend` imports work when script is run from repo root
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.neo4j_client import run_read


def _coerce_emb(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    # Neo4j may return array-like
    return [float(v) for v in list(x)]


def assemble_dataset(limit: int | None = None) -> pd.DataFrame:
    params = {"limit": limit} if limit else {}
    cypher = (
        "MATCH (c:Content) OPTIONAL MATCH (s:AudienceSegment)-[e:ENGAGED_WITH]->(c) "
        "WITH c, coalesce(sum(e.views),0) AS views, coalesce(sum(e.likes),0) AS likes, coalesce(avg(e.watch_time),0) AS avg_watch_time "
        "RETURN c.contentId AS contentId, c.title AS title, c.lengthSec AS lengthSec, c.text_embedding AS text_embedding, c.embedding AS gsage_embedding, views, likes, avg_watch_time "
    )
    rows = run_read(cypher, params)
    records: List[Dict[str, Any]] = []
    # Determine embedding dims if present
    text_dim = None
    graph_dim = None
    for r in rows:
        te = _coerce_emb(r.get('text_embedding'))
        ge = _coerce_emb(r.get('gsage_embedding') or r.get('embedding'))
        if text_dim is None and te:
            text_dim = len(te)
        if graph_dim is None and ge:
            graph_dim = len(ge)
    # Fallback defaults
    if text_dim is None:
        text_dim = 384
    if graph_dim is None:
        graph_dim = 64

    for r in rows:
        contentId = r.get('contentId')
        title = r.get('title')
        lengthSec = r.get('lengthSec') or 0
        te = _coerce_emb(r.get('text_embedding'))
        if not te:
            te = [0.0] * text_dim
        ge = _coerce_emb(r.get('gsage_embedding') or r.get('embedding'))
        if not ge:
            ge = [0.0] * graph_dim
        views = float(r.get('views') or 0.0)
        likes = float(r.get('likes') or 0.0)
        watch_time = float(r.get('avg_watch_time') or 0.0)

        records.append({
            'contentId': contentId,
            'title': title,
            'lengthSec': float(lengthSec),
            'text_emb': te,
            'graph_emb': ge,
            'views': views,
            'likes': likes,
            'watch_time': watch_time,
        })

    df = pd.DataFrame(records)

    # Feature engineering: normalize views and watch_time to create a target
    if len(df) == 0:
        return df
    # simple normalization by max (avoid div by zero)
    max_views = df['views'].max() or 1.0
    max_watch = df['watch_time'].max() or 1.0
    df['views_norm'] = df['views'] / max_views
    df['watch_norm'] = df['watch_time'] / max_watch
    # target: simple weighted sum
    df['target'] = 0.5 * df['views_norm'] + 0.5 * df['watch_norm']

    # build concatenated feature vector per-row
    def _concat(row):
        return list(row['graph_emb']) + list(row['text_emb']) + [row['lengthSec'], row['views'], row['likes'], row['watch_time']]

    df['features'] = df.apply(_concat, axis=1)
    return df[['contentId', 'title', 'features', 'target']]


def main(out: str = 'data/dataset.parquet', limit: int | None = None):
    df = assemble_dataset(limit=limit)
    if df.empty:
        print('No rows fetched from Neo4j; dataset is empty')
        return
    # Use pandas to_parquet (requires pyarrow)
    df.to_parquet(out, index=False)
    print('Wrote dataset rows=', len(df), 'to', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/dataset.parquet')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    main(out=args.out, limit=args.limit)
