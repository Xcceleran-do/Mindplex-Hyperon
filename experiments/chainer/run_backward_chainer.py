import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_API_BASE = "http://localhost:5000"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "chainer" / "rules"


def run_backward_query(api_base: str, query: str, depth: int, mode: str) -> dict:
    endpoint = "/api/chainer/raw" if mode == "raw" else "/api/chainer/query"
    response = requests.post(
        f"{api_base.rstrip('/')}{endpoint}",
        json={"whatToCheck": query, "depth": depth},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def save_result(result: dict, output_dir: Path, query: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = output_dir / f"chainer_query_{timestamp}.json"

    payload = {
        "query": query,
        "saved_at_utc": timestamp,
        "result": result,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "chainer_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backward chainer query via API and save result.")
    parser.add_argument("--query", required=True, help='MeTTa query, e.g. "(engagement A_N123 \"Low\")"')
    parser.add_argument("--depth", type=int, default=5, help="Backward chaining depth")
    parser.add_argument(
        "--mode",
        choices=["raw", "analyzed"],
        default="raw",
        help="raw: deterministic proofs only, analyzed: includes LLM justification",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Mining API base URL")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save backward chaining results",
    )
    args = parser.parse_args()

    result = run_backward_query(api_base=args.api_base, query=args.query, depth=args.depth, mode=args.mode)
    out_file = save_result(result=result, output_dir=Path(args.output_dir), query=args.query)

    print("Backward chaining completed.")
    print(f"- status: {result.get('status')}")
    if result.get("proof_count") is not None:
        print(f"- proof_count: {result.get('proof_count')}")
    print(f"- saved: {out_file}")

    justification = result.get("justification")
    if justification:
        print("\nJustification:\n")
        print(justification)


if __name__ == "__main__":
    main()
