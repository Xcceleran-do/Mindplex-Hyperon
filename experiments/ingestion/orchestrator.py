from typing import Any, Dict, Iterable, Mapping, Optional

from .agents import ExtractionAgentRegistry
from .llm_client import LLMClient
from .models import ExtractionPlan
from .utils import coerce_float, stable_record_id


class IngestionOrchestrator:
    """Coordinates extractor agents for one planned source schema."""

    def __init__(
        self,
        plan: ExtractionPlan,
        llm_client: Optional[LLMClient] = None,
        corpus_records: Optional[Iterable[Mapping[str, Any]]] = None,
    ):
        self.plan = plan
        self.registry = ExtractionAgentRegistry(llm_client=llm_client)
        self.numeric_stats = build_numeric_stats(corpus_records or [], plan)

    def process(self, record: Mapping[str, Any], rank_stats: Optional[Dict[Any, int]] = None) -> Dict[str, Any]:
        context = {
            "text_fields": self.plan.text_fields,
            "title_fields": ("title", "post_title", "name", "headline"),
            "numeric_stats": self.numeric_stats,
            "rank_stats": rank_stats or {},
        }
        results = self.registry.run(record, self.plan.metta_properties, context)
        metadata = {
            key: result.as_metadata()
            for key, result in results.items()
            if result.value not in (None, "", "Unknown")
        }

        enriched = dict(record)
        enriched.setdefault("id", stable_record_id(record, self.plan.id_fields))
        enriched["enriched_metadata"] = metadata
        enriched["_ingestion_plan"] = self.plan.to_dict()
        return enriched


def build_numeric_stats(records: Iterable[Mapping[str, Any]], plan: ExtractionPlan) -> Dict[tuple, Dict[str, Any]]:
    stats: Dict[tuple, Dict[str, Any]] = {}
    numeric_specs = [spec for spec in plan.properties if spec.agent == "numeric_bucket"]
    for spec in numeric_specs:
        values = []
        for record in records:
            for path in spec.field_paths:
                value = coerce_float(read_path(record, path))
                if value is not None:
                    values.append(value)
                    break
        stats[tuple(spec.field_paths)] = {"values": values}
    return stats


def read_path(record: Mapping[str, Any], path: str) -> Any:
    from .utils import get_path

    return get_path(record, path)
