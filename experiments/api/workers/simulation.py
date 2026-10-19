from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())
    sys.path.append(payload["project_root"])

    from experiments.services.petta_service import PeTTaService

    service = PeTTaService.create_required(
        payload["project_root"],
        payload["setup_metta"],
        verbose=False,
    )

    for fact in payload["base_facts"]:
        service.add_atom(fact)

    for fact in payload["hypothetical_facts"]:
        service.add_atom(fact)

    for rule in payload["rules"]:
        service.add_forward_only_rule(rule)

    results = {}
    for level in payload["engagement_levels"]:
        query = f'(engagement {payload["article_id"]} "{level}")'
        results[level] = service.query(query, depth=payload["depth"])

    print(json.dumps({"status": "success", "proofs_by_level": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
