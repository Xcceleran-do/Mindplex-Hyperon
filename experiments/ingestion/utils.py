import re


PROPERTY_NAME_RE = re.compile(r"[^a-z0-9-]+")
METTA_PREDICATE_ALIASES = {"length": "length-bucket"}


def normalize_property_name(value: str, fallback: str = "property") -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    normalized = PROPERTY_NAME_RE.sub("-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return normalized or fallback


def metta_predicate(value: str) -> str:
    normalized = normalize_property_name(value)
    return METTA_PREDICATE_ALIASES.get(normalized, normalized)


def sanitize_atom_id(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("A_"):
        return value[2:]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"
