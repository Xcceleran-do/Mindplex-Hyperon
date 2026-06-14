import datetime
import hashlib
import json
import os
import re
from typing import Any, Iterable, List, Mapping, Optional


PROPERTY_NAME_RE = re.compile(r"[^a-z0-9-]+")
METTA_PREDICATE_ALIASES = {
    "length": "length-bucket",
}
DEFAULT_EXCLUDED_PREDICATES = {
    "author",
    "authored-by",
    "author-username",
}
DEFAULT_PLANNER_EXCLUDED_PREDICATES = DEFAULT_EXCLUDED_PREDICATES | {
    "title",
    "id",
    "uuid",
    "slug",
    "username",
}


def normalize_property_name(value: str, fallback: str = "property") -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    normalized = PROPERTY_NAME_RE.sub("-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or fallback


def normalize_metadata_key(value: str, fallback: str = "property") -> str:
    return normalize_property_name(value, fallback=fallback).replace("-", "_")


def metta_predicate(value: str) -> str:
    normalized = normalize_property_name(value)
    return METTA_PREDICATE_ALIASES.get(normalized, normalized)


def excluded_predicates(include_display: bool = False) -> set[str]:
    raw = os.getenv("INGESTION_EXCLUDED_PREDICATES")
    if raw is None:
        return set(DEFAULT_PLANNER_EXCLUDED_PREDICATES if include_display else DEFAULT_EXCLUDED_PREDICATES)
    return {
        normalize_property_name(item)
        for item in raw.split(",")
        if normalize_property_name(item)
    }


def is_excluded_predicate(value: str, include_display: bool = False) -> bool:
    normalized = normalize_property_name(value)
    excluded = excluded_predicates(include_display=include_display)
    return normalized in excluded or normalized.replace("_", "-") in excluded


def stable_record_id(raw: Mapping[str, Any], id_fields: Iterable[str] = ()) -> str:
    for field in id_fields:
        value = get_path(raw, field)
        if value not in (None, ""):
            return sanitize_atom_id(str(value))

    payload = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def sanitize_atom_id(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("A_"):
        return value[2:]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


def get_path(data: Any, path: str) -> Any:
    """Read a dotted path from nested dict/list data.

    Supports simple list notation such as ``categories[].slug`` by returning
    the first non-empty value found in the list.
    """

    if not path:
        return None

    parts = path.split(".")

    def read(current: Any, index: int) -> Any:
        if current is None:
            return None
        if index >= len(parts):
            return first_scalar(current)

        part = parts[index]
        wants_list = part.endswith("[]")
        key = part[:-2] if wants_list else part

        if isinstance(current, Mapping):
            next_value = current.get(key)
        elif isinstance(current, list):
            next_value = [
                item.get(key) for item in current
                if isinstance(item, Mapping) and item.get(key) not in (None, "")
            ]
        else:
            return None

        if wants_list or isinstance(next_value, list):
            values = next_value if isinstance(next_value, list) else [next_value]
            for item in values:
                value = read(item, index + 1)
                if value not in (None, ""):
                    return value
            return None

        return read(next_value, index + 1)

    return read(data, 0)


def first_scalar(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            scalar = first_scalar(item)
            if scalar not in (None, ""):
                return scalar
        return None
    if isinstance(value, Mapping):
        for candidate_key in ("slug", "name", "title", "value", "content", "id"):
            if candidate_key in value and value[candidate_key] not in (None, ""):
                return value[candidate_key]
        return None
    return value


def text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, Mapping):
                chunks.append(str(item.get("content") or item.get("text") or item.get("title") or ""))
            else:
                chunks.append(text_from_value(item))
        return " ".join(chunk for chunk in chunks if chunk)
    if isinstance(value, Mapping):
        return " ".join(text_from_value(v) for v in value.values())
    return str(value)


def coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_datetime(value: Any) -> Optional[datetime.datetime]:
    if not value:
        return None
    text = str(value).strip()
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
    )
    for fmt in formats:
        try:
            return datetime.datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def sample_schema(records: Iterable[Mapping[str, Any]], max_depth: int = 2) -> List[str]:
    paths = set()

    def walk(value: Any, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                paths.add(child)
                walk(item, child, depth + 1)
        elif isinstance(value, list) and value:
            child = f"{prefix}[]" if prefix else "[]"
            paths.add(child)
            walk(value[0], child, depth + 1)

    for record in records:
        walk(record, "", 0)
    return sorted(paths)
