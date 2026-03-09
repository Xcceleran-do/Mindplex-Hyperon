"""I/O-centric ingestion agents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..models import IngestionConfig, IngestionState, SourceDocument
from ..tool_router import ToolRouter
from .base import Agent


class SourceResolutionAgent(Agent):
    name = "source-resolution"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        normalized_sources = list(config.sources)
        if not normalized_sources and config.username:
            normalized_sources = [f"mindplex:user:{config.username}"]

        for source in normalized_sources:
            if source.startswith("mindplex:user:"):
                username = source.split(":", 2)[-1]
                pseudo_record = {
                    "author": username,
                    "title": f"Ingested profile for {username}",
                    "content_type": "Profile",
                }
                state.documents.append(
                    SourceDocument(
                        source=source,
                        source_type="pseudo-json",
                        payload=[pseudo_record],
                        source_reliability=config.source_reliability,
                    )
                )
                continue

            path = Path(source)
            if path.is_dir():
                for child in tools.expand_directory(path):
                    document, error = tools.load_source(str(child), config.source_reliability)
                    if document is not None:
                        state.documents.append(document)
                    elif error:
                        state.skipped_sources.append(f"{child}: {error}")
                continue

            document, error = tools.load_source(source, config.source_reliability)
            if document is not None:
                state.documents.append(document)
            elif error:
                state.skipped_sources.append(f"{source}: {error}")


class RecordExtractionAgent(Agent):
    name = "record-extraction"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        for document in state.documents:
            extracted = tools.extract_records_from_payload(document.payload)
            for index, record in enumerate(extracted):
                if not isinstance(record, dict):
                    continue
                normalized = dict(record)
                normalized["_source"] = document.source
                normalized["_source_reliability"] = document.source_reliability
                normalized.setdefault("id", self._build_subject_id(document.source, index))
                state.records.append(normalized)

    def _build_subject_id(self, source: str, index: int) -> str:
        digest = hashlib.md5(f"{source}:{index}".encode("utf-8")).hexdigest()[:8]
        return f"A_{digest}"


class FactPersistenceAgent(Agent):
    name = "fact-persistence"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        default_output = Path(__file__).resolve().parents[1] / "outputs" / "data.metta"
        final_output_path = config.output_path or str(default_output)
        state.output_path = tools.write_metta(final_output_path, state.facts)
