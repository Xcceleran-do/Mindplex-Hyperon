"""Data models for the ingestion multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceDocument:
    """Normalized source document before record extraction."""

    source: str
    source_type: str
    payload: Any
    source_reliability: float


@dataclass
class Fact:
    """MeTTa triple represented with STV metadata."""

    predicate: str
    subject: str
    object_value: str
    confidence: float
    strength: float


@dataclass
class IngestionResult:
    """Serializable result of the ingestion pipeline."""

    status: str
    message: str
    output_path: Optional[str] = None
    facts_count: int = 0
    subjects: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    skipped_sources: List[str] = field(default_factory=list)
    agent_reports: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """Runtime config shared by all agents."""

    username: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    subject_prefix: str = "A"
    source_reliability: float = 0.9
    min_property_coverage: float = 0.25


@dataclass
class AgentReport:
    """Per-agent execution telemetry."""

    name: str
    status: str
    duration_ms: int
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class IngestionState:
    """Mutable ingestion state passed between agents."""

    documents: List[SourceDocument] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    schema_profile: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selected_properties: List[str] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    skipped_sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
