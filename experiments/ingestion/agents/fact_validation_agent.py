"""Fact validation agent."""

from __future__ import annotations

from typing import List

from ..models import Fact, IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class FactValidationAgent(Agent):
    name = "fact-validation"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        valid_facts: List[Fact] = []
        for fact in state.facts:
            if not fact.predicate or not fact.subject:
                state.warnings.append("Dropped fact with empty predicate/subject")
                continue
            if not (0.0 <= fact.confidence <= 1.0 and 0.0 <= fact.strength <= 1.0):
                state.warnings.append(f"Dropped out-of-range STV fact: {fact.predicate} {fact.subject}")
                continue
            valid_facts.append(fact)
        state.facts = valid_facts
