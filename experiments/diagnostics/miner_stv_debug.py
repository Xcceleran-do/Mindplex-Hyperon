#!/usr/bin/env python3
"""Inspect empirical STV support math for a mined conjunction."""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.services.petta_service import PeTTaService


def expect(label: str, actual: list[str], expected: list[str]) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> int:
    root = PROJECT_ROOT.replace("\\", "/")
    setup_metta = f"""
!(import! &self {root}/PeTTa/lib/lib_import.metta)
!(import! &self {root}/experiments/utils/common-utils)
!(import! &self {root}/experiments/frequent-pattern-miner/etv-utils)
!(import! &self {root}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {root}/experiments/pattern-miner/pattern-miner)
!(import! &tempo {root}/experiments/atomspace_visualizer/public/data)
!(let $atom (match &tempo ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))
"""
    service = PeTTaService.create_required(PROJECT_ROOT, setup_metta, verbose=False)

    pattern = '(, (author $x "Hruy") (authored-by $x "Hruy") (date-period $x "Archived") (audience-expertise $x "Beginner") (engagement $x "Low"))'
    antecedent = '(, (author $x "Hruy") (authored-by $x "Hruy") (date-period $x "Archived") (audience-expertise $x "Beginner"))'

    queries = {
        "pattern": f"!(counter &purifiedDbSpace {pattern})",
        "antecedent-direct": f"!(counter &purifiedDbSpace {antecedent})",
        "antecedent-fn": f"!(antecedent {pattern})",
        "antecedent-fn-counter": f"!(let $a (antecedent {pattern}) (counter &purifiedDbSpace $a))",
        "emp-tv": f"!(emp-tv {pattern} &purifiedDbSpace)",
    }
    results = {}
    for label, query in queries.items():
        results[label] = service.query_lines(query)
        print(f"{label}: {results[label]}")

    expect("pattern support", results["pattern"], ["3"])
    expect("direct antecedent support", results["antecedent-direct"], ["3"])
    expect("antecedent function support", results["antecedent-fn-counter"], ["3"])
    expect("empirical truth value", results["emp-tv"], ["(STV 1.0 0.75)"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
