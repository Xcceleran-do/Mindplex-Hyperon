#!/usr/bin/env python3
"""Smoke-check the PeTTaChainer STV formulas.

This is intentionally small and deterministic. It loads the production
PeTTaChainer library, evaluates the arithmetic formulas directly, and fails
if they drift from the audited equations.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from experiments.services.petta_service import PeTTaService


def normalize_result(result: object) -> str:
    if isinstance(result, (list, tuple)):
        return "\n".join(str(item) for item in result)
    return str(result)


def numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.I)]


def assert_close(actual: Iterable[float], expected: Iterable[float], label: str) -> None:
    actual_values = list(actual)
    expected_values = list(expected)
    if len(actual_values) < len(expected_values):
        raise AssertionError(f"{label}: expected at least {len(expected_values)} numbers, got {actual_values}")

    actual_tail = actual_values[-len(expected_values):]
    for got, want in zip(actual_tail, expected_values):
        if abs(got - want) > 1e-9:
            raise AssertionError(f"{label}: expected {expected_values}, got {actual_tail}")


def main() -> int:
    project_root_metta = PROJECT_ROOT.replace("\\", "/")
    setup_metta = f"""
!(import! &self {project_root_metta}/experiments/utils/common-utils)
!(import! &self {project_root_metta}/experiments/frequent-pattern-miner/etv-utils)
"""
    service = PeTTaService.create_required(PROJECT_ROOT, setup_metta, verbose=False)
    cases = [
        ("count_to_confidence", "!(count_to_confidence 3)", [0.75]),
        ("MpFormula", "!(MpFormula (STV 0.8 0.75) (STV 0.5 0.9))", [0.4, 0.675]),
        ("AndFormula", "!(AndFormula (STV 0.8 0.7) (STV 0.5 0.9))", [0.5, 0.7]),
        ("OrFormula", "!(OrFormula (STV 0.8 0.7) (STV 0.5 0.9))", [0.8, 0.7]),
        (
            "InversionFormula",
            "!(InversionFormula (STV 0.2 0.9) (STV 0.5 0.8) (STV 0.75 0.7))",
            [0.3, 0.7],
        ),
    ]

    for label, expr, expected in cases:
        result = normalize_result(service.process_metta_string(expr))
        assert_close(numbers(result), expected, label)
        print(f"{label}: {result}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
