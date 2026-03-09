"""Pluggable registry for ingestion agents.

This makes future extension simple: register new agents without editing
orchestrator internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Type

from .agents import (
    Agent,
    ContentClassificationAgent,
    DiscretizationAgent,
    FactPersistenceAgent,
    FactValidationAgent,
    PropertySelectionAgent,
    RecommendationSignalAgent,
    RecordExtractionAgent,
    SchemaProfilerAgent,
    SemanticParsingAgent,
    SentimentAnalysisAgent,
    SourceResolutionAgent,
    TripleConstructionAgent,
)


@dataclass
class AgentRegistry:
    """Ordered registry of agent classes."""

    agent_types: List[Type[Agent]] = field(default_factory=list)

    def register(self, agent_type: Type[Agent]) -> None:
        self.agent_types.append(agent_type)

    def insert_after(self, existing_agent_name: str, new_agent_type: Type[Agent]) -> None:
        for idx, agent_type in enumerate(self.agent_types):
            if getattr(agent_type, "name", "") == existing_agent_name:
                self.agent_types.insert(idx + 1, new_agent_type)
                return
        self.agent_types.append(new_agent_type)

    def insert_before(self, existing_agent_name: str, new_agent_type: Type[Agent]) -> None:
        for idx, agent_type in enumerate(self.agent_types):
            if getattr(agent_type, "name", "") == existing_agent_name:
                self.agent_types.insert(idx, new_agent_type)
                return
        self.agent_types.insert(0, new_agent_type)

    def build_instances(self) -> List[Agent]:
        return [agent_type() for agent_type in self.agent_types]

    @classmethod
    def from_sequence(cls, agent_types: Sequence[Type[Agent]]) -> "AgentRegistry":
        return cls(agent_types=list(agent_types))


def build_default_agent_registry() -> AgentRegistry:
    """Default ordered pipeline registry."""

    return AgentRegistry.from_sequence(
        [
            SourceResolutionAgent,
            RecordExtractionAgent,
            SentimentAnalysisAgent,
            ContentClassificationAgent,
            SemanticParsingAgent,
            RecommendationSignalAgent,
            SchemaProfilerAgent,
            PropertySelectionAgent,
            DiscretizationAgent,
            TripleConstructionAgent,
            FactValidationAgent,
            FactPersistenceAgent,
        ]
    )
