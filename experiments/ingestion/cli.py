#!/usr/bin/env python3
"""Command-line interface for the standalone ingestion tool."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .pipeline import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest heterogeneous resources, identify recommendation properties, "
            "and export MeTTa facts with STV values."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Input source path/URL/directory. Repeat to provide multiple sources.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Optional username used for backward-compatible ingestion flow.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for .metta file. Defaults to experiments/ingestion/outputs/data.metta.",
    )
    parser.add_argument(
        "--subject-prefix",
        default="A",
        help="Subject prefix for generated IDs (default: A).",
    )
    parser.add_argument(
        "--source-reliability",
        type=float,
        default=0.9,
        help="Base STV strength in [0,1] for source reliability.",
    )
    parser.add_argument(
        "--min-property-coverage",
        type=float,
        default=0.25,
        help="Minimum field coverage required for the property-selection agent.",
    )
    parser.add_argument(
        "--include-agent-reports",
        action="store_true",
        help="Include full agent runtime reports in CLI output JSON.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write ingestion summary JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_ingestion(
        username=args.username,
        sources=args.inputs,
        output_path=args.output,
        subject_prefix=args.subject_prefix,
        source_reliability=args.source_reliability,
        min_property_coverage=args.min_property_coverage,
    )

    if not args.include_agent_reports and "agent_reports" in result:
        result = dict(result)
        result.pop("agent_reports", None)

    print(json.dumps(result, indent=2))

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
