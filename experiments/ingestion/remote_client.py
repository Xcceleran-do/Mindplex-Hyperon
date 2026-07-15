from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Iterable
from urllib.parse import urlparse

import requests


ATOM_RE = re.compile(
    r'^\(([a-z][a-z0-9-]{0,63}) ([A-Za-z_][A-Za-z0-9_-]{0,127}) ("(?:[^"\\\r\n]|\\.)*")\)$'
)
DEFAULT_REQUIRED_PROPERTIES = ("engagement", "audience-expertise")


class MetadataExtractorError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteIngestionResult:
    dataset_lines: list[str]
    record_count: int
    fact_count: int
    plan_fingerprint: str
    planner: str
    model: str | None
    properties: list[str]
    usage: dict[str, int]


class MetadataExtractorClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        namespace: str = "A",
        timeout_seconds: float = 150.0,
        chunk_size: int = 10,
        required_properties: Iterable[str] = DEFAULT_REQUIRED_PROPERTIES,
        use_model: bool = True,
        session: requests.Session | None = None,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "METADATA_EXTRACTOR_BASE_URL must be an absolute HTTP(S) URL"
            )
        if not api_key:
            raise ValueError("METADATA_EXTRACTOR_API_KEY is required")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", namespace):
            raise ValueError("METADATA_EXTRACTOR_NAMESPACE is invalid")
        if not 1 <= chunk_size <= 100:
            raise ValueError("METADATA_EXTRACTOR_CHUNK_SIZE must be between 1 and 100")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        self.required_properties = list(
            dict.fromkeys(_property_name(item) for item in required_properties)
        )
        self.use_model = use_model
        self.session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "MetadataExtractorClient":
        required = [
            item.strip()
            for item in os.getenv(
                "METADATA_EXTRACTOR_REQUIRED_PROPERTIES",
                ",".join(DEFAULT_REQUIRED_PROPERTIES),
            ).split(",")
            if item.strip()
        ]
        return cls(
            base_url=os.getenv("METADATA_EXTRACTOR_BASE_URL", "http://127.0.0.1:8080"),
            api_key=os.getenv("METADATA_EXTRACTOR_API_KEY", ""),
            namespace=os.getenv("METADATA_EXTRACTOR_NAMESPACE", "A"),
            timeout_seconds=float(
                os.getenv("METADATA_EXTRACTOR_TIMEOUT_SECONDS", "150")
            ),
            chunk_size=int(os.getenv("METADATA_EXTRACTOR_CHUNK_SIZE", "10")),
            required_properties=required,
            use_model=os.getenv("METADATA_EXTRACTOR_USE_MODEL", "true").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def ingest(
        self, records: list[dict[str, Any]], *, source_name: str = "mindplex"
    ) -> RemoteIngestionResult:
        if not records:
            raise ValueError("records cannot be empty")
        plan_response = self._request(
            "/v1/plans/discover",
            {
                "source_name": source_name,
                "records": records[:20],
                "required_properties": self.required_properties,
                "use_model": self.use_model,
            },
        )
        plan = plan_response.get("plan")
        if not isinstance(plan, dict) or not plan.get("fingerprint"):
            raise MetadataExtractorError("metadata extractor returned an invalid plan")

        dataset_lines: list[str] = []
        usage = _usage(plan_response.get("usage"))
        processed_records = 0
        for start in range(0, len(records), self.chunk_size):
            chunk = records[start : start + self.chunk_size]
            response = self._request(
                "/v1/extract",
                {"namespace": self.namespace, "plan": plan, "records": chunk},
            )
            response_usage = _usage(response.get("usage"))
            usage = {
                "input_tokens": usage["input_tokens"] + response_usage["input_tokens"],
                "output_tokens": usage["output_tokens"]
                + response_usage["output_tokens"],
            }
            record_results = response.get("records")
            if not isinstance(record_results, list) or len(record_results) != len(
                chunk
            ):
                raise MetadataExtractorError(
                    "metadata extractor returned the wrong number of records"
                )
            for result in record_results:
                dataset_lines.extend(self._record_lines(result))
                processed_records += 1

        properties = plan.get("properties") or []
        property_names = [
            item["name"]
            for item in properties
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        return RemoteIngestionResult(
            dataset_lines=dataset_lines,
            record_count=processed_records,
            fact_count=len(dataset_lines),
            plan_fingerprint=plan["fingerprint"],
            planner=str(plan.get("planner") or "unknown"),
            model=plan_response.get("model"),
            properties=property_names,
            usage=usage,
        )

    def _record_lines(self, result: Any) -> list[str]:
        if not isinstance(result, dict):
            raise MetadataExtractorError(
                "metadata extractor returned an invalid record"
            )
        errors = result.get("errors")
        if errors:
            source_id = str(result.get("source_id") or "unknown")[:128]
            safe_errors = "; ".join(str(item)[:300] for item in errors[:5])
            raise MetadataExtractorError(
                f"metadata extraction failed for {source_id}: {safe_errors}"
            )
        facts = result.get("facts")
        if not isinstance(facts, list):
            raise MetadataExtractorError("metadata extractor omitted record facts")
        lines = []
        properties = set()
        for fact in facts:
            line, property_name = _dataset_line(fact)
            lines.append(line)
            properties.add(property_name)
        missing = set(self.required_properties) - properties
        if missing:
            raise MetadataExtractorError(
                "metadata extractor omitted required facts: "
                + ", ".join(sorted(missing))
            )
        return lines

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}{path}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise MetadataExtractorError(
                "metadata extractor could not be reached"
            ) from exc
        if response.status_code >= 400:
            message = "metadata extractor rejected the request"
            try:
                body = response.json()
                remote_error = body.get("error") if isinstance(body, dict) else None
                if isinstance(remote_error, dict) and remote_error.get("message"):
                    message = str(remote_error["message"])[:500]
            except (ValueError, TypeError):
                pass
            raise MetadataExtractorError(f"{message} (HTTP {response.status_code})")
        try:
            body = response.json()
        except ValueError as exc:
            raise MetadataExtractorError(
                "metadata extractor returned non-JSON output"
            ) from exc
        if not isinstance(body, dict):
            raise MetadataExtractorError("metadata extractor returned invalid JSON")
        return body


def _dataset_line(fact: Any) -> tuple[str, str]:
    if not isinstance(fact, dict):
        raise MetadataExtractorError("metadata extractor returned an invalid fact")
    atom = fact.get("atom")
    property_name = fact.get("property_name")
    match = ATOM_RE.fullmatch(atom) if isinstance(atom, str) else None
    if not match or property_name != match.group(1):
        raise MetadataExtractorError("metadata extractor returned an unsafe fact atom")
    try:
        value = json.loads(match.group(3))
        strength = float(fact["strength"])
        confidence = float(fact["confidence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetadataExtractorError(
            "metadata extractor returned an invalid truth value"
        ) from exc
    if not isinstance(value, str) or not 0 <= strength <= 1 or not 0 <= confidence <= 1:
        raise MetadataExtractorError("metadata extractor returned an invalid fact")
    return (
        f"({atom} (STV {_number(strength)} {_number(confidence)}))",
        property_name,
    )


def _usage(value: Any) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    return {
        "input_tokens": max(0, int(value.get("input_tokens") or 0)),
        "output_tokens": max(0, int(value.get("output_tokens") or 0)),
    }


def _number(value: float) -> str:
    return format(value, ".15g")


def _property_name(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", normalized):
        raise ValueError("METADATA_EXTRACTOR_REQUIRED_PROPERTIES contains an invalid name")
    return normalized
