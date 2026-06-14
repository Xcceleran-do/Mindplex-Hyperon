from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated PeTTa mining job.")
    parser.add_argument("--conjunction-count", type=int, required=True)
    parser.add_argument("--min-support", type=int, required=True)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()

    try:
        from experiments import mining_api

        result = mining_api.run_mining_task_inprocess(
            args.conjunction_count,
            args.min_support,
        )
        Path(args.result_path).write_text(json.dumps(result), encoding="utf-8")
        return 0
    except Exception as exc:
        error_result = {
            "status": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }
        try:
            Path(args.result_path).write_text(json.dumps(error_result), encoding="utf-8")
        except Exception:
            pass
        print(error_result["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
