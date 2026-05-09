"""Fail-fast PeTTa runtime diagnostic.

Run with:

    python -m experiments.diagnostics.petta_check
"""

from __future__ import annotations

import json

from experiments.mining_api import create_app, get_petta_service


def main() -> int:
    create_app()
    print(json.dumps(get_petta_service().health(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
