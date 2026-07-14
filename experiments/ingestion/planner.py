import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .llm_client import LLMClient
from .models import ExtractionPlan, PropertySpec
from .utils import (
    coerce_float,
    excluded_predicates,
    get_path,
    is_excluded_predicate,
    normalize_property_name,
    sample_schema,
    text_from_value,
)


TEXT_FIELD_HINTS = ("content", "body", "text", "overview", "description", "summary", "abstract")
TITLE_FIELD_HINTS = ("title", "name", "headline")
ID_FIELD_HINTS = ("id", "uuid", "slug")
DATE_FIELD_HINTS = ("date", "time", "timestamp", "published", "created", "updated")
READING_TIME_HINTS = ("min_to_read", "read_time", "reading_time", "minutes_to_read")
ENGAGEMENT_FIELD_HINTS = ("view", "like", "comment", "reaction", "share", "clap")
CONTENT_ANALYSIS_TEXT_SPECS = (
    PropertySpec(
        "audience-expertise",
        "Required predicate: intended reader expertise level. Use Beginner, Intermediate, Advanced, or Expert when possible.",
        "text_llm",
        allowed_values=("Beginner", "Intermediate", "Advanced", "Expert"),
    ),
    PropertySpec("topic", "Dominant content topic or subject area.", "text_llm"),
    PropertySpec("tone", "Communication tone of the content.", "text_llm"),
    PropertySpec("content-type", "Content format or genre.", "text_llm"),
    PropertySpec("primary-goal", "Main goal of the content.", "text_llm"),
    PropertySpec("audience-sentiment", "Expected audience sentiment or stance.", "text_llm"),
    PropertySpec("complexity", "Conceptual complexity of the content.", "text_llm"),
    PropertySpec("actionability", "How directly actionable the content is.", "text_llm"),
)


class ExtractionPlanner:
    """Creates an extraction plan from a sample of raw source records."""

    def __init__(self, llm_client: Optional[LLMClient] = None, require_llm: bool = False):
        self.llm_client = llm_client
        self.require_llm = require_llm

    def build_plan(self, records: Iterable[Mapping[str, Any]], source_name: str = "json") -> ExtractionPlan:
        records = list(records)
        pinned_plan = os.getenv("INGESTION_PLAN_JSON")
        if pinned_plan:
            plan = ExtractionPlan.from_dict(json.loads(pinned_plan), source_name=source_name)
            return self._sanitize_plan(plan, records, planner="env")

        if self.llm_client and self.llm_client.available:
            try:
                plan = self._build_llm_plan(records, source_name)
                return self._sanitize_plan(plan, records, planner="llm")
            except Exception as exc:
                if self.require_llm:
                    raise RuntimeError(f"LLM extraction planning failed: {exc}") from exc
                print(f"Warning: LLM extraction planning failed, using schema heuristic: {exc}")
        elif self.require_llm:
            raise RuntimeError("ASI_API_KEY is required because INGESTION_REQUIRE_LLM=true.")

        return self._sanitize_plan(
            self._build_heuristic_plan(records, source_name),
            records,
            planner="heuristic",
        )

    def _build_llm_plan(self, records: List[Mapping[str, Any]], source_name: str) -> ExtractionPlan:
        sample_records = [compact_record(record) for record in records[:5]]
        schema_paths = sample_schema(sample_records, max_depth=2)
        prompt = {
            "task": "Create a source-agnostic property extraction plan for pattern mining.",
            "product_context": (
                "We are building a content analysis and pattern-mining tool. "
                "The facts should describe reusable content attributes, audience fit, "
                "format, intent, topic, complexity, quality signals, and engagement."
            ),
            "source_name": source_name,
            "requirements": [
                "Extract as many useful content-analysis predicates as the source can support.",
                "Choose categorical or bucketed properties that repeat across records.",
                "Do not assume a fixed article schema.",
                "Do not include identity/display predicates such as title, author, authored-by, username, id, uuid, or slug.",
                "audience-expertise is mandatory and must be extracted from text.",
                "engagement is mandatory. Always include it as a calculated_metric; aggregate views, likes, comments, reactions, shares, or similar counters.",
                "Use calculated_metric for length, reading-time, and engagement. These use legacy proportional STV buckets from the production code.",
                "Use structured_field for direct categorical fields, numeric_bucket for other numeric metrics, date_bucket for timestamps, and text_llm for semantic text attributes.",
                "Return JSON only.",
            ],
            "recommended_content_predicates": [
                "audience-expertise",
                "topic",
                "tone",
                "content-type",
                "primary-goal",
                "audience-sentiment",
                "complexity",
                "actionability",
                "length",
                "reading-time",
                "engagement",
                "date-period",
            ],
            "forbidden_predicates": sorted(excluded_predicates(include_display=True)),
            "json_schema": {
                "entity_type": "item",
                "id_fields": ["field.path"],
                "text_fields": ["field.path"],
                "properties": [
                    {
                        "name": "kebab-case-property",
                        "description": "what this extracts",
                        "agent": "structured_field | calculated_metric | numeric_bucket | date_bucket | text_llm",
                        "field_paths": ["field.path"],
                        "value_type": "category | string | number",
                        "allowed_values": [],
                        "parameters": {},
                        "include_in_metta": True,
                    }
                ],
            },
            "schema_paths": schema_paths,
            "sample_records": sample_records,
        }
        messages = [
            {
                "role": "system",
                "content": "You design robust data ingestion extraction plans. Return only valid JSON.",
            },
            {"role": "user", "content": json.dumps(prompt, default=str)[:16000]},
        ]
        response = self.llm_client.complete_json(messages)
        return ExtractionPlan.from_dict(response, source_name=source_name)

    def _build_heuristic_plan(self, records: List[Mapping[str, Any]], source_name: str) -> ExtractionPlan:
        schema_paths = sample_schema(records, max_depth=2)
        id_fields = choose_fields(schema_paths, ID_FIELD_HINTS, default=["id"])
        text_fields = choose_fields(schema_paths, TEXT_FIELD_HINTS)
        date_fields = choose_fields(schema_paths, DATE_FIELD_HINTS)

        properties: List[PropertySpec] = []

        properties.extend(self._calculated_content_specs(schema_paths))
        properties.extend(self._structured_low_cardinality_specs(records, schema_paths))
        properties.extend(self._numeric_specs(records, schema_paths))

        if date_fields:
            properties.append(
                PropertySpec(
                    name="date-period",
                    description="Relative age bucket for the record timestamp.",
                    agent="date_bucket",
                    field_paths=(date_fields[0],),
                    allowed_values=("Current", "Recent", "Older", "Archived"),
                )
            )

        if text_fields:
            properties.extend(CONTENT_ANALYSIS_TEXT_SPECS)

        return ExtractionPlan(
            source_name=source_name,
            entity_type="item",
            id_fields=tuple(id_fields),
            text_fields=tuple(text_fields),
            properties=tuple(dedupe_specs(properties)),
            planner="heuristic",
        )

    def _calculated_content_specs(self, schema_paths: List[str]) -> List[PropertySpec]:
        readable_paths = choose_fields(schema_paths, READING_TIME_HINTS)
        specs = [
            PropertySpec(
                "length",
                "Legacy word-count bucket for content length.",
                "calculated_metric",
                field_paths=(),
                allowed_values=("Short", "Medium", "Long"),
                parameters={"metric": "length"},
            ),
            PropertySpec(
                "reading-time",
                "Legacy reading-time bucket from source minutes or estimated words-per-minute.",
                "calculated_metric",
                field_paths=tuple(readable_paths),
                allowed_values=("Very_Short", "Short", "Medium", "Long"),
                parameters={"metric": "reading-time"},
            ),
        ]

        engagement_paths = engagement_field_paths(schema_paths)
        if engagement_paths:
            specs.append(
                PropertySpec(
                    "engagement",
                    "Required predicate: legacy aggregate engagement bucket from views, likes, comments, reactions, shares, and similar counters.",
                    "calculated_metric",
                    field_paths=tuple(engagement_paths),
                    allowed_values=("Low", "Medium", "High", "Very_High"),
                    parameters={"metric": "engagement"},
                )
            )
        return specs

    def _structured_low_cardinality_specs(
        self,
        records: List[Mapping[str, Any]],
        schema_paths: List[str],
    ) -> List[PropertySpec]:
        specs = []
        blocked = set(ID_FIELD_HINTS + TEXT_FIELD_HINTS + DATE_FIELD_HINTS)
        for path in schema_paths:
            name = path.replace("[]", "")
            lowered = name.lower()
            if any(token in lowered for token in blocked):
                continue
            if is_excluded_predicate(name, include_display=True):
                continue
            values = [get_path(record, path) for record in records]
            values = [text_from_value(value).strip() for value in values if text_from_value(value).strip()]
            if not values:
                continue
            avg_len = sum(len(value) for value in values) / len(values)
            unique_ratio = len(set(values)) / max(1, len(values))
            if avg_len <= 48 and unique_ratio <= 0.8:
                specs.append(
                    PropertySpec(
                        name=normalize_property_name(path.replace("[]", "")),
                        description=f"Structured source field {path}.",
                        agent="structured_field",
                        value_type="category",
                        field_paths=(path,),
                    )
                )
        return specs

    def _numeric_specs(self, records: List[Mapping[str, Any]], schema_paths: List[str]) -> List[PropertySpec]:
        specs = []
        for path in schema_paths:
            lowered = path.lower()
            if any(token in lowered for token in ID_FIELD_HINTS + DATE_FIELD_HINTS):
                continue
            numbers = [coerce_float(get_path(record, path)) for record in records]
            numbers = [number for number in numbers if number is not None]
            if len(numbers) < max(2, min(5, len(records))):
                continue
            specs.append(
                PropertySpec(
                    name=f"{normalize_property_name(path.replace('[]', ''))}-level",
                    description=f"Relative bucket for numeric source field {path}.",
                    agent="numeric_bucket",
                    value_type="category",
                    field_paths=(path,),
                    allowed_values=("Low", "Medium", "High", "Very_High"),
                )
            )
        return specs

    def _sanitize_plan(
        self,
        plan: ExtractionPlan,
        records: List[Mapping[str, Any]],
        planner: str,
    ) -> ExtractionPlan:
        properties = []
        for spec in plan.properties:
            name = normalize_property_name(spec.name)
            if not name or is_excluded_predicate(name, include_display=True):
                continue
            if name == "engagement":
                continue
            properties.append(
                PropertySpec(
                    name=name,
                    description=spec.description,
                    agent=normalize_agent(spec.agent),
                    value_type=spec.value_type,
                    field_paths=spec.field_paths,
                    allowed_values=spec.allowed_values,
                    parameters=spec.parameters,
                    include_in_metta=spec.include_in_metta,
                )
            )

        properties.extend(missing_required_specs(properties, records))
        text_fields = plan.text_fields or tuple(choose_fields(sample_schema(records, 2), TEXT_FIELD_HINTS))
        id_fields = plan.id_fields or ("id",)
        return ExtractionPlan(
            source_name=plan.source_name,
            entity_type=plan.entity_type or "item",
            id_fields=tuple(id_fields),
            text_fields=tuple(text_fields),
            properties=tuple(dedupe_specs(properties)),
            version=plan.version,
            planner=planner,
        )


def compact_record(record: Mapping[str, Any], max_chars: int = 500) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, default=str)[:max_chars]
            compact[key] = text
        else:
            compact[key] = str(value)[:max_chars]
    return compact


def choose_fields(paths: List[str], hints: Iterable[str], default: Optional[List[str]] = None) -> List[str]:
    selected = []
    for path in paths:
        lowered = path.lower()
        if any(hint in lowered for hint in hints):
            selected.append(path)
    return selected or list(default or [])


def dedupe_specs(specs: Iterable[PropertySpec]) -> List[PropertySpec]:
    seen = set()
    deduped = []
    for spec in specs:
        name = normalize_property_name(spec.name)
        if name in seen:
            continue
        seen.add(name)
        deduped.append(spec)
    return deduped


def normalize_agent(agent: str) -> str:
    aliases = {
        "structured": "structured_field",
        "field": "structured_field",
        "llm": "text_llm",
        "text": "text_llm",
        "semantic": "text_llm",
        "numeric": "numeric_bucket",
        "metric": "numeric_bucket",
        "date": "date_bucket",
        "datetime": "date_bucket",
        "calculated": "calculated_metric",
        "derived": "calculated_metric",
        "derived_metric": "calculated_metric",
    }
    key = str(agent or "").strip().lower().replace("-", "_")
    return aliases.get(key, key or "text_llm")


def missing_required_specs(specs: Iterable[PropertySpec], records: List[Mapping[str, Any]]) -> List[PropertySpec]:
    existing = {normalize_property_name(spec.name) for spec in specs}
    additions: List[PropertySpec] = []
    schema_paths = sample_schema(records, max_depth=2)
    has_text = bool(choose_fields(schema_paths, TEXT_FIELD_HINTS))
    if "length" not in existing and has_text:
        additions.append(
            PropertySpec(
                "length",
                "Legacy word-count bucket for content length.",
                "calculated_metric",
                field_paths=(),
                allowed_values=("Short", "Medium", "Long"),
                parameters={"metric": "length"},
            )
        )
    if "reading-time" not in existing and (has_text or choose_fields(schema_paths, READING_TIME_HINTS)):
        additions.append(
            PropertySpec(
                "reading-time",
                "Legacy reading-time bucket from source minutes or estimated words-per-minute.",
                "calculated_metric",
                field_paths=tuple(choose_fields(schema_paths, READING_TIME_HINTS)),
                allowed_values=("Very_Short", "Short", "Medium", "Long"),
                parameters={"metric": "reading-time"},
            )
        )
    if "audience-expertise" not in existing and has_text:
        additions.append(CONTENT_ANALYSIS_TEXT_SPECS[0])
    if "engagement" not in existing:
        additions.append(
            PropertySpec(
                "engagement",
                "Required predicate: legacy aggregate engagement bucket from views, likes, comments, reactions, shares, and similar counters.",
                "calculated_metric",
                field_paths=tuple(engagement_field_paths(schema_paths)),
                allowed_values=("Low", "Medium", "High", "Very_High"),
                parameters={"metric": "engagement"},
            )
        )
    return additions


def engagement_field_paths(schema_paths: Iterable[str]) -> List[str]:
    return [
        path
        for path in schema_paths
        if any(token in path.lower() for token in ENGAGEMENT_FIELD_HINTS)
    ]
