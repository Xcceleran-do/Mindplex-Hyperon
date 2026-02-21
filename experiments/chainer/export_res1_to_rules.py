import argparse
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_BASE = "http://localhost:5000"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "experiments" / "chainer" / "rules.metta"


def export_rules(api_base: str, output_file: Path) -> dict:
    response = requests.post(
        f"{api_base.rstrip('/')}/api/chainer/export-rules",
        json={"outputPath": str(output_file)},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export current &res1 atoms to experiments/chainer/rules.metta")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Mining API base URL")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE), help="Output .metta file path")
    args = parser.parse_args()

    result = export_rules(api_base=args.api_base, output_file=Path(args.output_file))

    print("Export completed.")
    print(f"- status: {result.get('status')}")
    print(f"- atom_count: {result.get('atomCount')}")
    print(f"- output: {result.get('outputPath')}")


if __name__ == "__main__":
    main()
