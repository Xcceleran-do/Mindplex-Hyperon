"""Small helper to compute and persist text embeddings for Content nodes.

Usage:
  python backend/scripts/persist_text_embeddings.py [--limit N]

This avoids complex PowerShell escaping when running one-liner -c commands.
"""
import argparse
import json
import sys
import pathlib

# Ensure repo root is on sys.path so `backend` package imports work when running the script
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.embeddings import compute_text_embedding, save_text_embeddings_to_neo4j
from backend.app.services.neo4j_client import get_driver, run_read


def main(limit: int = 3, model_name: str = "all-MiniLM-L6-v2"):
    rows = run_read("MATCH (c:Content) RETURN c.contentId AS contentId, c.title AS title LIMIT $limit", {"limit": limit})
    if not rows:
        print("No content rows found.")
        return
    titles = [r["title"] for r in rows]
    emb = compute_text_embedding(titles, model_name=model_name)
    payload = [{"contentId": r["contentId"], "embedding": emb[i]} for i, r in enumerate(rows)]
    save_text_embeddings_to_neo4j(get_driver(), payload)
    print("Persisted", len(payload))
    # Print verification
    ids = [r["contentId"] for r in rows]
    check = run_read("MATCH (c:Content) WHERE c.contentId IN $ids RETURN c.contentId AS cid, size(c.text_embedding) AS dim ORDER BY c.contentId", {"ids": ids})
    print("STEP5_VERIFY", json.dumps(check))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(limit=args.limit, model_name=args.model)
