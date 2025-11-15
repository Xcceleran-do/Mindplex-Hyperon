"""Neo4j client & helper utilities (Step 4).

Features:
 - Centralized driver creation (singleton) with env configuration.
 - Context-managed sessions and automatic retry for transient errors.
 - Helper methods for read/write queries with parameter binding.
 - Record serialization helpers (to dict / list of dicts).
 - Convenience domain helpers (fetch content embeddings, counts).
 - Lazy .env loading using python-dotenv if available.

Environment variables used:
  NEO4J_URI (default bolt://localhost:7687)
  NEO4J_USER (default neo4j)
  NEO4J_PASSWORD (default 12345678)

Safe to import early; driver only created on first need.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from neo4j import GraphDatabase, Driver, Session, Result

try:  # Optional dependency
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:  # pragma: no cover - ignore if dotenv missing
    pass

_log = logging.getLogger(__name__)

_DRIVER: Optional[Driver] = None

def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)

def get_driver() -> Driver:
    """Return a singleton Neo4j driver instance."""
    global _DRIVER
    if _DRIVER is None:
        uri = _get_env("NEO4J_URI", "bolt://localhost:7687")
        user = _get_env("NEO4J_USER", "neo4j")
        password = _get_env("NEO4J_PASSWORD", "12345678")
        _log.info("Initializing Neo4j driver uri=%s user=%s", uri, user)
        _DRIVER = GraphDatabase.driver(uri, auth=(user, password))
    return _DRIVER

def close_driver() -> None:
    global _DRIVER
    if _DRIVER is not None:
        _log.info("Closing Neo4j driver")
        _DRIVER.close()
        _DRIVER = None

def _session(database: Optional[str] = None) -> Session:
    return get_driver().session(database=database)

def run_query(cypher: str, params: Optional[Dict[str, Any]] = None, *, database: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run a Cypher query (auto session) and return list of dict records."""
    params = params or {}
    _log.debug("Cypher run: %s params=%s", cypher, params)
    with _session(database) as s:
        result: Result = s.run(cypher, params)
        return [r.data() for r in result]

def run_write(cypher: str, params: Optional[Dict[str, Any]] = None, *, database: Optional[str] = None) -> List[Dict[str, Any]]:
    """Explicit write query (transaction function)."""
    params = params or {}
    _log.debug("Cypher write: %s params=%s", cypher, params)
    with _session(database) as s:
        def _tx_run(tx):
            res = tx.run(cypher, params)
            return [r.data() for r in res]
        return s.execute_write(_tx_run)

def run_read(cypher: str, params: Optional[Dict[str, Any]] = None, *, database: Optional[str] = None) -> List[Dict[str, Any]]:
    params = params or {}
    _log.debug("Cypher read: %s params=%s", cypher, params)
    with _session(database) as s:
        def _tx_run(tx):
            res = tx.run(cypher, params)
            return [r.data() for r in res]
        return s.execute_read(_tx_run)

def scalar(cypher: str, params: Optional[Dict[str, Any]] = None, *, database: Optional[str] = None) -> Any:
    records = run_query(cypher, params, database=database)
    if not records:
        return None
    return next(iter(records[0].values())) if records[0] else None

def fetch_content_embeddings(limit: int = 100) -> List[Dict[str, Any]]:
    """Return content nodes with their GraphSAGE embeddings (first N)."""
    cypher = (
        "MATCH (c:Content) WHERE c.embedding IS NOT NULL "
        "RETURN c.contentId AS contentId, c.title AS title, "
        "size(c.embedding) AS dim, c.embedding[0..15] AS sample, c.engagementScore AS engagementScore "
        "ORDER BY contentId LIMIT $limit"
    )
    return run_read(cypher, {"limit": limit})

def count_nodes(label: str) -> int:
    return int(scalar(f"MATCH (n:{label}) RETURN count(n) AS c") or 0)

def health_check() -> Dict[str, Any]:
    return {
        "neo4j_version": scalar("CALL dbms.components() YIELD versions RETURN versions[0] AS version"),
        "content_nodes": count_nodes("Content"),
        "has_embeddings": bool(scalar("MATCH (c:Content) WHERE c.embedding IS NOT NULL RETURN count(c) AS cnt")),
    }

if __name__ == "__main__":  # Simple manual test
    logging.basicConfig(level=logging.INFO)
    print("Health:", health_check())
    print("Embeddings sample:")
    for row in fetch_content_embeddings(5):
        print(row)
    close_driver()
