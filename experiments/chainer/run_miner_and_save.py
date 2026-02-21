import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_API_BASE = "http://localhost:5000"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "chainer" / "rules"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def run_mining(api_base: str, conjunction_count: int) -> dict:
    response = requests.post(
        f"{api_base.rstrip('/')}/api/mine",
        json={"conjunction_count": conjunction_count},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def save_outputs(payload: dict, output_dir: Path, conjunction_count: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_prefix = f"mine_c{conjunction_count}_{timestamp}"

    run_json = output_dir / f"{run_prefix}.json"
    latest_json = output_dir / "mined_latest.json"
    latest_metta = output_dir / "mined_latest.metta"

    run_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    patterns = payload.get("result", []) if isinstance(payload, dict) else []
    metta_lines = []
    for item in patterns:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).replace('"', '\\"')
        support = str(item.get("support", "")).replace('"', '\\"')
        if pattern:
            metta_lines.append(f'(mined-pattern "{pattern}" "{support}")')
    latest_metta.write_text("\n".join(metta_lines), encoding="utf-8")

    return {
        "run_json": str(run_json),
        "latest_json": str(latest_json),
        "latest_metta": str(latest_metta),
        "pattern_count": len(patterns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run miner via API and persist outputs for chaining.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Mining API base URL")
    parser.add_argument("--conjunction-count", type=int, default=5, help="Conjunction count for /api/mine")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save mined patterns/rules snapshots",
    )
    args = parser.parse_args()

    payload = run_mining(api_base=args.api_base, conjunction_count=args.conjunction_count)
    paths = save_outputs(payload=payload, output_dir=Path(args.output_dir), conjunction_count=args.conjunction_count)

    print("Mining completed and saved.")
    print(f"- patterns: {paths['pattern_count']}")
    print(f"- run snapshot: {paths['run_json']}")
    print(f"- latest snapshot: {paths['latest_json']}")
    print(f"- metta export: {paths['latest_metta']}")


if __name__ == "__main__":
    main()
