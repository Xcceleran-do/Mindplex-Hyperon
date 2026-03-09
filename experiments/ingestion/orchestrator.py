"""Ingestion orchestrator that coordinates specialized agents."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .agent_registry import AgentRegistry, build_default_agent_registry
from .agents import (
    Agent,
)
from .models import AgentReport, IngestionConfig, IngestionState
from .tool_router import ToolRouter


class MultiAgentIngestionOrchestrator:
    """Coordinates specialized ingestion agents with guard rails."""

    def __init__(self, tools: Optional[ToolRouter] = None, registry: Optional[AgentRegistry] = None) -> None:
        self.tools = tools or ToolRouter()
        self.registry = registry or build_default_agent_registry()
        self.agents: List[Agent] = self.registry.build_instances()

    def execute(self, config: IngestionConfig) -> Tuple[IngestionState, List[AgentReport]]:
        state = IngestionState()
        reports: List[AgentReport] = []

        for agent in self.agents:
            report = self._run_agent(agent, state, config)
            reports.append(report)

            if report.status == "error":
                break

            if agent.name == "source-resolution" and not state.documents:
                reports.append(
                    AgentReport(
                        name="pipeline-guard",
                        status="error",
                        duration_ms=0,
                        details={},
                        error="No ingestible sources were resolved",
                    )
                )
                break

            if agent.name == "record-extraction" and not state.records:
                reports.append(
                    AgentReport(
                        name="pipeline-guard",
                        status="error",
                        duration_ms=0,
                        details={},
                        error="No records could be extracted from sources",
                    )
                )
                break

        return state, reports

    def _run_agent(self, agent: Agent, state: IngestionState, config: IngestionConfig) -> AgentReport:
        started = time.perf_counter()
        try:
            agent.run(state, config, self.tools)
            details = {
                "documents": len(state.documents),
                "records": len(state.records),
                "selected_properties": len(state.selected_properties),
                "facts": len(state.facts),
                "warnings": len(state.warnings),
            }
            return AgentReport(
                name=agent.name,
                status="success",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details=details,
            )
        except Exception as exc:  # defensive guard
            return AgentReport(
                name=agent.name,
                status="error",
                duration_ms=int((time.perf_counter() - started) * 1000),
                details={},
                error=str(exc),
            )
