"""Placeholder multimedia ingestion module.

This module is intentionally minimal for now. It keeps the extension point for
image/audio/video extraction while the current pipeline focuses on text and
structured resources.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MultimediaResult:
    """Result returned by the multimedia ingester placeholder."""

    status: str
    message: str
    extracted_records: List[Dict[str, str]]


class MultimediaIngester:
    """Reserved extension point for future multimedia ingestion support."""

    def ingest(self, source: str) -> MultimediaResult:
        return MultimediaResult(
            status="not_implemented",
            message=(
                "Multimedia ingestion is reserved for the next phase. "
                "Provide text, JSON, CSV, NDJSON, URL, or directory sources for now."
            ),
            extracted_records=[],
        )
