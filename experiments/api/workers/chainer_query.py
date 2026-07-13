from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    payload = json.loads(sys.stdin.read())
    sys.path.append(payload["project_root"])

    try:
        from experiments.services.petta_service import PeTTaService

        service = PeTTaService.create_required(
            payload["project_root"],
            payload["setup_metta"],
            verbose=False,
        )

        for fact in payload["facts"]:
            service.add_atom(fact)

        for rule in payload["rules"]:
            service.add_forward_only_rule(rule)

        result = service.query(payload["query"], depth=int(payload["depth"]))
        print(json.dumps({"status": "success", "proofs": result}))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
