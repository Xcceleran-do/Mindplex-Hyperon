"""Embeddings utilities (Step 5).

Provides functions to compute text embeddings using sentence-transformers, persist them
to Neo4j, and combine different embedding sources into a unified vector.

Functions:
  compute_text_embedding(text_list, model_name='all-MiniLM-L6-v2') -> np.ndarray
  save_text_embeddings_to_neo4j(driver, content_embeddings: List[dict]) -> None
  combine_embeddings(graph_emb, text_emb, numeric_features) -> np.ndarray

Design notes:
  * Lazily loads SentenceTransformer model and caches it (global _MODEL).
  * Auto-detects GPU (CUDA or MPS) if torch is installed; falls back to CPU.
  * Neo4j write uses UNWIND + MERGE for efficient batch upsert.
  * Uses list[float] storage for text embeddings (sufficient for demo scale).
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Any, Optional
import numpy as np
import logging

try:  # heavy import guarded
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

_log = logging.getLogger(__name__)
_MODEL: Optional[SentenceTransformer] = None

def _select_device() -> str:
    """Select best available device for embeddings (cuda > mps > cpu)."""
    if torch is None:
        return 'cpu'
    if torch.cuda.is_available():  # pragma: no cover (depends on host GPU)
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # pragma: no cover
        return 'mps'
    return 'cpu'

def _get_model(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed. Please add to requirements.')
    global _MODEL
    if _MODEL is None or getattr(_MODEL, '_model_name', None) != model_name:
        device = _select_device()
        _log.info("Loading SentenceTransformer model=%s device=%s", model_name, device)
        _MODEL = SentenceTransformer(model_name, device=device)  # type: ignore[arg-type]
        _MODEL._model_name = model_name  # type: ignore[attr-defined]
    return _MODEL

def compute_text_embedding(texts: Sequence[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Compute embeddings for a list of texts.

    Parameters
    ----------
    texts : Sequence[str]
        List/sequence of input strings.
    model_name : str, default 'all-MiniLM-L6-v2'
        SentenceTransformer model identifier.

    Returns
    -------
    np.ndarray
        2D array shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, 0))
    model = _get_model(model_name)
    return model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)

def save_text_embeddings_to_neo4j(driver, content_embeddings: List[Dict[str, Any]]) -> None:
    """Persist text embeddings to Neo4j.

    Expects list of dicts: {contentId: str, embedding: np.ndarray or list[float]}.
    Writes property `text_embedding` on :Content nodes.
    """
    if not content_embeddings:
        _log.warning("No content embeddings provided; nothing to save.")
        return
    # Convert np arrays to Python lists for Neo4j driver JSON serialization.
    rows = []
    for item in content_embeddings:
        emb = item.get('embedding')
        if emb is None:
            continue
        if isinstance(emb, np.ndarray):
            emb_list = emb.tolist()
        else:
            emb_list = list(emb)
        rows.append({'contentId': item['contentId'], 'text_embedding': emb_list})
    if not rows:
        _log.warning("After filtering, no embeddings remained to save.")
        return
    cypher = (
        "UNWIND $rows AS row "
        "MERGE (c:Content {contentId: row.contentId}) "
        "SET c.text_embedding = row.text_embedding"
    )
    with driver.session() as session:
        session.run(cypher, {'rows': rows})
    _log.info("Persisted %d text embeddings to Neo4j", len(rows))

def combine_embeddings(graph_emb: Sequence[float], text_emb: Sequence[float], numeric_feats: Sequence[float]) -> np.ndarray:
    """Concatenate embeddings and numeric feature sequence into single numpy vector."""
    return np.concatenate([
        np.array(graph_emb, dtype=float),
        np.array(text_emb, dtype=float),
        np.array(numeric_feats, dtype=float),
    ])

def _demo_fetch_and_persist(limit: int = 5, model_name: str = 'all-MiniLM-L6-v2') -> None:  # pragma: no cover
    """Demonstration: fetch Content titles lacking text_embedding, compute & persist."""
    from .neo4j_client import get_driver, run_read
    driver = get_driver()
    rows = run_read(
        "MATCH (c:Content) WHERE c.text_embedding IS NULL RETURN c.contentId AS contentId, c.title AS title LIMIT $limit",
        {"limit": limit},
    )
    titles = [r['title'] for r in rows]
    emb = compute_text_embedding(titles, model_name=model_name)
    payload = [
        {"contentId": r['contentId'], "embedding": emb[i]} for i, r in enumerate(rows)
    ]
    save_text_embeddings_to_neo4j(get_driver(), payload)
    _log.info("Demo persisted %d embeddings", len(payload))

if __name__ == "__main__":  # Manual test
    logging.basicConfig(level=logging.INFO)
    _demo_fetch_and_persist(limit=3)
