"""Schema profiling agent."""

from __future__ import annotations

from ..models import IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class SchemaProfilerAgent(Agent):
    name = "schema-profiler"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        state.schema_profile = tools.profile_schema(state.records)
