#!/usr/bin/env python3
"""Public ingestion entrypoint.

This file intentionally stays thin and delegates implementation details to
modular components in `models.py`, `tool_router.py`, `agents/`, and
`orchestrator.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .models import IngestionConfig, IngestionResult
from .orchestrator import MultiAgentIngestionOrchestrator


def run_ingestion(
    username: Optional[str] = None,
    sources: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    subject_prefix: str = "A",
    source_reliability: float = 0.9,
    min_property_coverage: float = 0.25,
) -> Dict[str, Any]:
    """Standalone ingestion entrypoint used by both API and CLI."""

    if source_reliability < 0 or source_reliability > 1:
        return IngestionResult(
            status="error",
            message="source_reliability must be in [0, 1]",
        ).__dict__

    if min_property_coverage < 0 or min_property_coverage > 1:
        return IngestionResult(
            status="error",
            message="min_property_coverage must be in [0, 1]",
        ).__dict__

    config = IngestionConfig(
        username=username,
        sources=list(sources or []),
        output_path=output_path,
        subject_prefix=subject_prefix,
        source_reliability=source_reliability,
        min_property_coverage=min_property_coverage,
    )

    orchestrator = MultiAgentIngestionOrchestrator()
    state, reports = orchestrator.execute(config)

    failed_report = next((report for report in reports if report.status == "error"), None)
    if failed_report is not None:
        return IngestionResult(
            status="error",
            message=failed_report.error or f"Agent failed: {failed_report.name}",
            skipped_sources=state.skipped_sources,
            agent_reports=[report.__dict__ for report in reports],
        ).__dict__

    unique_subjects = sorted({fact.subject for fact in state.facts})
    unique_properties = sorted({fact.predicate for fact in state.facts})

    message = "Ingestion completed"
    if state.warnings:
        message = f"Ingestion completed with {len(state.warnings)} warning(s)"

    return IngestionResult(
        status="success",
        message=message,
        output_path=state.output_path,
        facts_count=len(state.facts),
        subjects=unique_subjects,
        properties=unique_properties,
        skipped_sources=state.skipped_sources,
        agent_reports=[report.__dict__ for report in reports],
    ).__dict__
