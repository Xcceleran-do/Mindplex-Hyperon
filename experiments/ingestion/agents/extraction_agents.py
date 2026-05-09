import datetime
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from experiments.ingestion.config import (
    DETERMINISTIC_STV,
    ENGAGEMENT_BUCKETS,
    LENGTH_BUCKETS,
    READING_TIME_BUCKETS,
    UNKNOWN_STV,
)
from experiments.ingestion.llm_client import LLMClient
from experiments.ingestion.models import AgentResult, PropertySpec
from experiments.ingestion.utils import (
    clamp01,
    coerce_float,
    get_path,
    normalize_metadata_key,
    parse_datetime,
    text_from_value,
)


class ExtractionAgent:
    agent_name = "base"

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        raise NotImplementedError


class StructuredFieldAgent(ExtractionAgent):
    agent_name = "structured_field"

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        results = {}
        for spec in specs:
            value = first_field_value(record, spec.field_paths)
            if value in (None, "", "Unknown"):
                continue
            results[normalize_metadata_key(spec.name)] = AgentResult(
                property_name=spec.name,
                value=text_from_value(value).strip(),
                stv=DETERMINISTIC_STV,
                evidence="structured_field",
                source_agent=self.agent_name,
            )
        return results


class NumericBucketAgent(ExtractionAgent):
    agent_name = "numeric_bucket"

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        results = {}
        corpus_stats = context.get("numeric_stats", {})
        for spec in specs:
            value = coerce_float(first_field_value(record, spec.field_paths))
            if value is None:
                continue
            stats = corpus_stats.get(tuple(spec.field_paths)) or {}
            bucket, strength = bucket_numeric(value, stats)
            results[normalize_metadata_key(spec.name)] = AgentResult(
                property_name=spec.name,
                value=bucket,
                stv=(round(strength, 3), 0.9),
                evidence=f"{value}",
                source_agent=self.agent_name,
            )
        return results


class CalculatedMetricAgent(ExtractionAgent):
    agent_name = "calculated_metric"

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        results = {}
        for spec in specs:
            metric = str(spec.parameters.get("metric") or spec.name).replace("_", "-")
            if metric == "engagement":
                value = calculate_engagement(record, spec.field_paths)
                if value is None:
                    continue
                label, stv = legacy_bucket_stv(value, ENGAGEMENT_BUCKETS, legacy_bounds("engagement"))
                evidence = f"aggregate={value}"
            elif metric == "length":
                text = build_text_context(record, context.get("text_fields") or ())
                value = len(text.split())
                label, stv = legacy_bucket_stv(value, LENGTH_BUCKETS, legacy_bounds("length"))
                evidence = f"word_count={value}"
            elif metric == "reading-time":
                value = parse_reading_minutes(first_field_value(record, spec.field_paths))
                if value is None:
                    text = build_text_context(record, context.get("text_fields") or ())
                    word_count = len(text.split())
                    value = max(1, math.ceil(word_count / 200)) if word_count else None
                if value is None:
                    continue
                label, stv = legacy_bucket_stv(value, READING_TIME_BUCKETS, legacy_bounds("reading-time"))
                evidence = f"minutes={value}"
            else:
                continue

            results[normalize_metadata_key(spec.name)] = AgentResult(
                property_name=spec.name,
                value=label,
                stv=stv,
                evidence=evidence,
                source_agent=self.agent_name,
            )
        return results


class DateBucketAgent(ExtractionAgent):
    agent_name = "date_bucket"

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        results = {}
        now = context.get("now") or datetime.datetime.now()
        for spec in specs:
            raw_value = first_field_value(record, spec.field_paths)
            parsed = parse_datetime(raw_value)
            if not parsed:
                continue
            days = max(0, (now - parsed).days)
            if days <= 7:
                bucket = "Current"
            elif days <= 30:
                bucket = "Recent"
            elif days <= 90:
                bucket = "Older"
            else:
                bucket = "Archived"
            results[normalize_metadata_key(spec.name)] = AgentResult(
                property_name=spec.name,
                value=bucket,
                stv=DETERMINISTIC_STV,
                evidence=str(raw_value),
                source_agent=self.agent_name,
            )
        return results


class TextLLMAgent(ExtractionAgent):
    agent_name = "text_llm"

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client

    def extract(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        specs = list(specs)
        if not specs:
            return {}
        if not self.llm_client or not self.llm_client.available:
            return {
                normalize_metadata_key(spec.name): AgentResult(
                    property_name=spec.name,
                    value="Unknown",
                    stv=UNKNOWN_STV,
                    evidence="llm_unavailable",
                    source_agent=self.agent_name,
                )
                for spec in specs
            }

        text = build_text_context(record, context.get("text_fields") or ())
        title = text_from_value(first_field_value(record, context.get("title_fields") or ("title", "post_title", "name")))
        request = {
            "task": "Extract the requested semantic properties from this record.",
            "instructions": [
                "Return JSON only.",
                "This is for a content analysis and pattern-mining tool.",
                "Use short categorical values that can repeat across many records.",
                "Extract all requested predicates from the content, not from identity/display metadata.",
                "For audience-expertise, infer who the content is written for: Beginner, Intermediate, Advanced, or Expert.",
                "For every property include value, strength, and confidence.",
                "Strength is the degree of truth of the label. Confidence is your certainty.",
            ],
            "properties": [spec.to_dict() for spec in specs],
            "record": {
                "title": title,
                "text": text[:5000],
            },
        }
        messages = [
            {
                "role": "system",
                "content": "You are a careful data extraction agent for a probabilistic knowledge graph.",
            },
            {"role": "user", "content": json.dumps(request, default=str)},
        ]
        try:
            payload = self.llm_client.complete_json(messages)
        except Exception as exc:
            print(f"Warning: text LLM extraction failed: {exc}")
            payload = {}

        results = {}
        for spec in specs:
            key = spec.name
            value = payload.get(key) or payload.get(normalize_metadata_key(key)) or payload.get(key.replace("-", "_"))
            parsed = normalize_llm_value(value)
            if not parsed:
                parsed = {"value": "Unknown", "strength": 0.5, "confidence": 0.5}
            results[normalize_metadata_key(spec.name)] = AgentResult(
                property_name=spec.name,
                value=parsed["value"],
                stv=(parsed["strength"], parsed["confidence"]),
                evidence="text_llm",
                source_agent=self.agent_name,
            )
        return results


class ExtractionAgentRegistry:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.agents = {
            StructuredFieldAgent.agent_name: StructuredFieldAgent(),
            NumericBucketAgent.agent_name: NumericBucketAgent(),
            CalculatedMetricAgent.agent_name: CalculatedMetricAgent(),
            DateBucketAgent.agent_name: DateBucketAgent(),
            TextLLMAgent.agent_name: TextLLMAgent(llm_client),
        }

    def run(
        self,
        record: Mapping[str, Any],
        specs: Iterable[PropertySpec],
        context: Mapping[str, Any],
    ) -> Dict[str, AgentResult]:
        grouped: Dict[str, List[PropertySpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.agent, []).append(spec)

        results: Dict[str, AgentResult] = {}
        for agent_name, agent_specs in grouped.items():
            agent = self.agents.get(agent_name) or self.agents[TextLLMAgent.agent_name]
            results.update(agent.extract(record, agent_specs, context))
        return results


def first_field_value(record: Mapping[str, Any], field_paths: Iterable[str]) -> Any:
    for path in field_paths:
        value = get_path(record, path)
        if value not in (None, ""):
            return value
    return None


def build_text_context(record: Mapping[str, Any], field_paths: Iterable[str]) -> str:
    chunks = []
    for path in field_paths:
        text = text_from_value(get_path(record, path)).strip()
        if text:
            chunks.append(text)
    if chunks:
        return "\n\n".join(chunks)
    return text_from_value(record)


def normalize_llm_value(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        label = value.get("value") or value.get("label") or value.get("category")
        strength = clamp01(value.get("strength"), default=0.5)
        confidence = clamp01(value.get("confidence"), default=0.5)
    else:
        label = value
        strength = 0.5
        confidence = 0.5
    if label in (None, ""):
        return None
    return {
        "value": str(label).strip(),
        "strength": strength,
        "confidence": confidence,
    }


def bucket_numeric(value: float, stats: Mapping[str, Any]) -> tuple[str, float]:
    values = stats.get("values") or []
    if not values:
        if value <= 0:
            return "Low", 0.2
        return "High", 0.8

    sorted_values = sorted(values)
    n = len(sorted_values)
    rank = sum(1 for item in sorted_values if item <= value) / max(1, n)
    if rank <= 0.25:
        bucket = "Low"
    elif rank <= 0.5:
        bucket = "Medium"
    elif rank <= 0.75:
        bucket = "High"
    else:
        bucket = "Very_High"
    strength = 0.15 + (0.8 * rank)
    return bucket, min(0.95, max(0.05, strength))


def calculate_engagement(record: Mapping[str, Any], field_paths: Iterable[str]) -> Optional[float]:
    paths = tuple(field_paths) or (
        "views",
        "likes",
        "comments",
        "comment_count",
        "shares",
        "reactions",
        "claps",
    )
    values = []
    for path in paths:
        number = coerce_float(get_path(record, path))
        if number is not None:
            values.append(number)
    if not values:
        return None
    return sum(values)


def parse_reading_minutes(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    number = coerce_float(value)
    if number is not None:
        return number
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return None
    return float(match.group(1))


def legacy_bucket_stv(value: float, buckets: Mapping[str, Any], bounds: Mapping[str, tuple[float, float]]) -> tuple[str, tuple[float, float]]:
    for label, (condition, stv_range) in buckets.items():
        try:
            if condition(value):
                return label, proportional_stv(value, stv_range, bounds.get(label))
        except Exception:
            continue
    return "Unknown", UNKNOWN_STV


def proportional_stv(value: float, stv_range: tuple[float, float], bucket_bounds: Optional[tuple[float, float]]) -> tuple[float, float]:
    if stv_range[0] == stv_range[1]:
        return stv_range
    if bucket_bounds:
        min_val, max_val = bucket_bounds
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
    else:
        normalized = 0.5
    strength = stv_range[0] + normalized * (stv_range[1] - stv_range[0])
    return round(strength, 3), 0.9


def legacy_bounds(metric: str) -> Mapping[str, tuple[float, float]]:
    if metric == "length":
        return {
            "Short": (0, 500),
            "Medium": (500, 1500),
            "Long": (1500, 3000),
        }
    if metric == "reading-time":
        return {
            "Very_Short": (0, 2),
            "Short": (2, 5),
            "Medium": (5, 10),
            "Long": (10, 20),
        }
    if metric == "engagement":
        return {
            "Low": (0, 30),
            "Medium": (30, 50),
            "High": (50, 100),
            "Very_High": (100, 1000),
        }
    return {}
