"""Base agent interface for ingestion orchestration."""

from __future__ import annotations

from ..models import IngestionConfig, IngestionState
from ..tool_router import ToolRouter


class Agent:
    """Base agent interface."""

    name = "agent"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        raise NotImplementedError()
