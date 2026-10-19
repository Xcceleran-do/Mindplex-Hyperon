from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv

from experiments.ingestion.config import DEFAULT_USERNAME
from experiments.ingestion.fetcher import MindplexFetcher
from experiments.ingestion.remote_client import (
    MetadataExtractorClient,
    MetadataExtractorError,
)


DEFAULT_OUTPUT_PATH = "experiments/atomspace_visualizer/public/data.metta"
DEFAULT_LIMIT = 50


def resolve_output_path(output_path: str | None = None) -> str:
    return output_path or os.getenv("METTA_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)


def run_ingestion(
    username: str | None = None,
    source_name: str = "mindplex",
    limit: int = DEFAULT_LIMIT,
    output_path: str | None = None,
    source_config: dict | None = None,
) -> dict:
    """Fetch Mindplex records and delegate all enrichment to metadata-extractor2PLN."""
    load_dotenv()
    if source_name != "mindplex" or source_config:
        return {
            "status": "error",
            "code": "unsupported_source",
            "message": "Mindplex-Hyperon only fetches the Mindplex source.",
        }
    username = username or os.getenv("MINDPLEX_USERNAME") or DEFAULT_USERNAME
    output_path = resolve_output_path(output_path)
    try:
        records = MindplexFetcher(username=username).fetch_all(limit=limit)
        if not records:
            return {
                "status": "error",
                "code": "no_articles",
                "message": "No Mindplex articles were found.",
            }
        remote = MetadataExtractorClient.from_env().ingest(
            records, source_name="mindplex"
        )
        _write_dataset_atomically(output_path, remote.dataset_lines)
    except (MetadataExtractorError, OSError, ValueError) as exc:
        return {
            "status": "error",
            "code": "remote_ingestion_failed",
            "message": str(exc),
        }

    return {
        "status": "success",
        "message": f"Ingested {remote.record_count} Mindplex articles",
        "source": "mindplex",
        "username": username,
        "records": remote.record_count,
        "facts": remote.fact_count,
        "output_path": output_path,
        "planner": remote.planner,
        "model": remote.model,
        "plan_fingerprint": remote.plan_fingerprint,
        "property_count": len(remote.properties),
        "properties": remote.properties,
        "usage": remote.usage,
    }


def _write_dataset_atomically(output_path: str, lines: list[str]) -> None:
    if not lines:
        raise ValueError("metadata extractor returned no facts")
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".data-",
            suffix=".metta.tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write("\n".join(lines))
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


if __name__ == "__main__":
    print(run_ingestion())
