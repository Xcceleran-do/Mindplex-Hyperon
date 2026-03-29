"""Triple construction agent."""

from __future__ import annotations

from typing import List

from ..models import Fact, IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class TripleConstructionAgent(Agent):
    name = "triple-construction"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        facts: List[Fact] = []

        for record in state.records:
            subject = tools.normalize_subject(record.get("id"), config.subject_prefix)
            source_reliability = float(record.get("_source_reliability", config.source_reliability))

            for raw_property in state.selected_properties:
                if raw_property in {"id", "_source", "_source_reliability"}:
                    continue
                if raw_property not in record:
                    continue

                value = record.get(raw_property)
                if value is None or value == "":
                    continue

                raw_value = value
                if isinstance(value, dict) and "value" in value:
                    value = value.get("value")
                    if value is None or value == "":
                        continue

                predicate = tools.normalize_predicate(raw_property)
                object_value = tools.normalize_object(value)
                confidence, strength = tools.compute_stv(
                    record=record,
                    raw_property=raw_property,
                    value=value,
                    raw_value=raw_value,
                    source_reliability=source_reliability,
                )

                facts.append(
                    Fact(
                        predicate=predicate,
                        subject=subject,
                        object_value=object_value,
                        confidence=confidence,
                        strength=strength,
                    )
                )

                if predicate == "author":
                    facts.append(
                        Fact(
                            predicate="authored-by",
                            subject=subject,
                            object_value=object_value,
                            confidence=confidence,
                            strength=strength,
                        )
                    )

        state.facts = sorted(facts, key=lambda f: (f.subject, f.predicate, f.object_value))
