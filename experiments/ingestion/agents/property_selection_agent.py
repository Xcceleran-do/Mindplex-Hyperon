"""Property selection agent."""

from __future__ import annotations

from typing import List

from ..constants import IGNORED_PROPERTY_SUFFIXES
from ..models import IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class PropertySelectionAgent(Agent):
    name = "property-selection"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        selected: List[str] = []
        for key, stats in state.schema_profile.items():
            if self._ignore_key(key):
                continue
            if stats.get("coverage", 0.0) >= config.min_property_coverage:
                selected.append(key)

        if "id" not in selected:
            selected.append("id")

        state.selected_properties = sorted(set(selected))

    def _ignore_key(self, key: str) -> bool:
        lowered = key.lower()
        if lowered.startswith("_"):
            return True
        return any(lowered.endswith(suffix) for suffix in IGNORED_PROPERTY_SUFFIXES)
