from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import threading
from typing import Callable

from dotenv import load_dotenv

from experiments.ingestion.pipeline import resolve_output_path


DEFAULT_CACHE_TTL_DAYS = 3.0
_ingestion_lock = threading.Lock()


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cache_ttl() -> timedelta:
    raw_value = os.getenv("INGESTION_CACHE_TTL_DAYS", str(DEFAULT_CACHE_TTL_DAYS))
    try:
        days = max(0.0, float(raw_value))
    except ValueError:
        days = DEFAULT_CACHE_TTL_DAYS
    return timedelta(days=days)


def _metadata_path(output_path: str) -> str:
    return f"{output_path}.ingestion.json"


def _read_cache(
    *, username: str, source_name: str, limit: int, output_path: str
) -> dict | None:
    ttl = _cache_ttl()
    metadata_path = _metadata_path(output_path)
    if ttl <= timedelta(0) or not os.path.isfile(output_path) or not os.path.isfile(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        completed_at = datetime.fromisoformat(metadata["completed_at"])
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - completed_at.astimezone(timezone.utc)
        if age < timedelta(0) or age >= ttl:
            return None
        if metadata.get("source") != source_name:
            return None
        if str(metadata.get("username", "")).casefold() != username.casefold():
            return None
        if int(metadata.get("requested_limit", 0)) < limit:
            return None
        result = metadata.get("result")
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    expires_at = completed_at + ttl
    age_seconds = max(0, int(age.total_seconds()))
    return {
        **result,
        "cached": True,
        "message": (
            f"Reusing the existing dataset for @{username}; "
            f"it was ingested {age_seconds} seconds ago."
        ),
        "cache_age_seconds": age_seconds,
        "cache_expires_at": expires_at.isoformat(),
    }


def _write_cache(
    *, username: str, source_name: str, limit: int, output_path: str, result: dict
) -> None:
    metadata_path = _metadata_path(output_path)
    output_dir = os.path.dirname(metadata_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "version": 1,
        "source": source_name,
        "username": username,
        "requested_limit": limit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".ingestion-",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary_path, metadata_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def run_ingestion_request(
    run_ingestion: Callable[..., dict],
    *,
    username: str | None,
    source_name: str,
    limit: int,
    output_path: str | None,
    source_config: dict | None,
    force: bool = False,
) -> dict:
    """Run API ingestion with an authoritative switch and freshness cache."""
    load_dotenv()
    if not _env_enabled("INGESTION_ENABLED", True):
        return {
            "status": "error",
            "code": "ingestion_disabled",
            "message": "Ingestion is disabled by the server configuration.",
        }

    resolved_output_path = resolve_output_path(output_path)
    normalized_username = (username or "").strip()
    cacheable = source_name == "mindplex" and bool(normalized_username)

    # Check and ingest under one lock so concurrent requests cannot both spend tokens.
    with _ingestion_lock:
        if cacheable and not force:
            cached = _read_cache(
                username=normalized_username,
                source_name=source_name,
                limit=limit,
                output_path=resolved_output_path,
            )
            if cached is not None:
                return cached

        result = run_ingestion(
            username=username,
            source_name=source_name,
            limit=limit,
            output_path=resolved_output_path,
            source_config=source_config,
        )
        result = {**result, "cached": False}
        if cacheable and result.get("status") == "success":
            _write_cache(
                username=normalized_username,
                source_name=source_name,
                limit=limit,
                output_path=resolved_output_path,
                result=result,
            )
        return result
