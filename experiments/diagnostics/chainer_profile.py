"""Profile the PeTTaChainer hot path without calling the LLM.

Run with:

    python -m experiments.diagnostics.chainer_profile
"""

from __future__ import annotations

import re
import time
import argparse
from contextlib import contextmanager

from experiments.mining_api import (
    create_app,
    getAllFactsAndRules,
    get_chainer_service,
    insert_mined_rules_into_chainer,
    mine_pattern,
)


@contextmanager
def timed(label: str, timings: list[tuple[str, float]]):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings.append((label, elapsed))
    print(f"{label}: {elapsed:.4f}s")


def query_from_fact(fact: str) -> str:
    match = re.match(r'^\(:\s+\(fact:-\s+(.+)\)\s+\1\s+\(STV\s+', fact)
    if match:
        return match.group(1)

    match = re.match(r'^\(:\s+fact[\w\-]*\s+(.+)\s+\(STV\s+', fact)
    if match:
        return match.group(1)

    raise ValueError(f"Cannot derive query from fact: {fact}")


def kb_atom_count(service) -> str:
    lines = service.query_lines("!(size-atom (collapse (get-atoms &kb)))")
    return lines[0] if lines else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mine-conjunction", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--direct-depth", type=int, default=2)
    parser.add_argument("--engagement-depth", type=int, default=3)
    parser.add_argument("--broad-depth", type=int, default=2)
    parser.add_argument("--skip-broad", action="store_true")
    parser.add_argument("--show-proofs", action="store_true")
    args = parser.parse_args()

    timings: list[tuple[str, float]] = []

    with timed("create_app/bootstrap", timings):
        create_app()

    service = get_chainer_service()

    with timed("ensure dataset facts in KB", timings):
        facts_result = getAllFactsAndRules()
    if facts_result.get("status") != "success":
        raise RuntimeError(f"Failed to hydrate chainer facts: {facts_result}")
    facts = facts_result.get("facts", [])
    compiled = facts_result.get("compiled_new_count", 0)

    kb_after_compile = kb_atom_count(service)
    mined_result = None
    if args.mine_conjunction > 0:
        with timed(
            f"mine patterns conjunction={args.mine_conjunction} min_support={args.min_support}",
            timings,
        ):
            mined_result = mine_pattern(args.mine_conjunction, args.min_support)

        with timed("compile mined rules", timings):
            mined_result = insert_mined_rules_into_chainer(mined_result)

    kb_after_rules = kb_atom_count(service)

    direct_query = query_from_fact(facts[0])
    engagement_fact = next((fact for fact in facts if "(engagement " in fact), facts[0])
    engagement_query = query_from_fact(engagement_fact)

    with timed(f"direct query depth={args.direct_depth} first", timings):
        direct_first = service.query(direct_query, depth=args.direct_depth)
    kb_after_direct_first = kb_atom_count(service)

    with timed(f"direct query depth={args.direct_depth} repeat", timings):
        direct_repeat = service.query(direct_query, depth=args.direct_depth)
    kb_after_direct_repeat = kb_atom_count(service)

    with timed(f"engagement query depth={args.engagement_depth}", timings):
        engagement_proofs = service.query(engagement_query, depth=args.engagement_depth)
    kb_after_engagement = kb_atom_count(service)

    broad_proofs = []
    kb_after_broad = "skipped"
    if not args.skip_broad:
        with timed(f"broad engagement query depth={args.broad_depth}", timings):
            broad_proofs = service.query("(engagement $article $level)", depth=args.broad_depth)
        kb_after_broad = kb_atom_count(service)

    print()
    print("summary")
    print(f"facts={len(facts)} compiled_dataset_facts={compiled}")
    if mined_result is not None:
        print(
            "mining="
            f"status:{mined_result.get('status')}, "
            f"patterns:{len(mined_result.get('patterns', []))}, "
            f"rules:{mined_result.get('inserted_rule_count')}"
        )
    print(f"direct_query={direct_query}")
    print(f"direct_first_proofs={len(direct_first)} direct_repeat_proofs={len(direct_repeat)}")
    print(f"engagement_query={engagement_query} proofs={len(engagement_proofs)}")
    if args.show_proofs:
        print("engagement_proofs:")
        for proof in engagement_proofs:
            print(f"  {proof}")
        if mined_result is not None:
            print("inserted_rules:")
            for rule in mined_result.get("rule_insertion", {}).get("rules", []):
                print(f"  {rule}")
    print(f"broad_engagement_proofs={len(broad_proofs)}")
    print(
        "kb_atom_counts="
        f"after_compile:{kb_after_compile}, "
        f"after_rules:{kb_after_rules}, "
        f"after_direct_first:{kb_after_direct_first}, "
        f"after_direct_repeat:{kb_after_direct_repeat}, "
        f"after_engagement:{kb_after_engagement}, "
        f"after_broad:{kb_after_broad}"
    )
    print(f"service_health={service.health()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
