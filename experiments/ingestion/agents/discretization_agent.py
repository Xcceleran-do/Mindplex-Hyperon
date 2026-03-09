"""Value discretization agent."""

from __future__ import annotations

from typing import Any, Dict, List

from ..models import IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class DiscretizationAgent(Agent):
    name = "value-discretization"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        numeric_values_by_property: Dict[str, List[float]] = {}

        for prop in state.selected_properties:
            stats = state.schema_profile.get(prop, {})
            values = stats.get("numeric_values", [])
            if values:
                numeric_values_by_property[prop] = list(values)

        transformed: List[Dict[str, Any]] = []
        for record in state.records:
            item = dict(record)
            for prop, values in numeric_values_by_property.items():
                if prop not in item:
                    continue
                value = item.get(prop)
                if not isinstance(value, (int, float)):
                    continue
                item[prop] = tools.discretize_numeric(float(value), values)
            transformed.append(item)

        state.records = transformed
