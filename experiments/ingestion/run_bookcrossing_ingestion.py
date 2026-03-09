#!/usr/bin/env python3
"""Run ingestion on Kaggle Book-Crossing dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.ingestion.adapters.bookcrossing_adapter import (
    build_bookcrossing_records,
    write_records_jsonl,
)
from experiments.ingestion.pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Book-Crossing Kaggle dataset into MeTTa facts.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing BX-Books.csv, BX-Users.csv, BX-Book-Ratings.csv")
    parser.add_argument(
        "--prepared-jsonl",
        default="experiments/ingestion/outputs/bookcrossing_prepared.jsonl",
        help="Intermediate JSONL output path",
    )
    parser.add_argument(
        "--output-metta",
        default="experiments/ingestion/outputs/bookcrossing_data.metta",
        help="Final .metta output path",
    )
    parser.add_argument("--limit-books", type=int, default=10000, help="Cap number of books processed")
    parser.add_argument("--min-property-coverage", type=float, default=0.2)
    parser.add_argument("--source-reliability", type=float, default=0.92)
    args = parser.parse_args()

    records = build_bookcrossing_records(args.dataset_dir, limit_books=args.limit_books)
    prepared_path = write_records_jsonl(records, args.prepared_jsonl)

    result = run_ingestion(
        sources=[prepared_path],
        output_path=args.output_metta,
        min_property_coverage=args.min_property_coverage,
        source_reliability=args.source_reliability,
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
