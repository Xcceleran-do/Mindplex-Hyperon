"""Boot PeTTaChainer and prove that runtime dataset facts compile into the KB.

Run with:

    python -m experiments.diagnostics.pettachainer_runtime_smoke
"""

from __future__ import annotations

import re

from experiments.mining_api import backWardChainer, create_app, getAllFactsAndRules


def first_queryable_fact(facts: list[str]) -> str:
    for fact in facts:
        match = re.match(r'^\(:\s+\(fact:-\s+(.+)\)\s+\1\s+\(STV\s+', fact)
        if match:
            return match.group(1)

        match = re.match(r'^\(:\s+fact[\w\-]*\s+(.+)\s+\(STV\s+', fact)
        if match:
            return match.group(1)

    raise AssertionError("No queryable facts were parsed from &res1.")


def main() -> int:
    create_app()
    facts_result = getAllFactsAndRules()
    assert facts_result["status"] == "success", facts_result
    facts = facts_result.get("facts", [])
    assert facts, facts_result

    query = first_queryable_fact(facts)
    proofs = backWardChainer(query, depth=2)
    assert proofs, {"query": query, "facts_result": facts_result}
    assert not any("(partial " in proof for proof in proofs), proofs

    print(f"PeTTaChainer runtime smoke test passed with {len(facts)} facts.")
    print(f"Direct query: {query}")
    print(f"Proof sample: {proofs[:1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
