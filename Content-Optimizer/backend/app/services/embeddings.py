from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def compute_text_embedding(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Compute embeddings for a list of texts using sentence-transformers."""
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True)
    return emb


def combine_embeddings(graph_emb: List[float], text_emb: List[float], numeric_feats: List[float]):
    return np.concatenate([np.array(graph_emb, dtype=float), np.array(text_emb, dtype=float), np.array(numeric_feats, dtype=float)])
