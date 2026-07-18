from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Optional
from experiments.services.petta_service import PeTTaService
from experiments.api.config import  MINING_METTA_SETUP,PROJECT_ROOT
DATASET_FACT_LINE_RE = re.compile(
    r'^\s*\(\s*(\(.+\))\s+(\(STV\s+[0-9eE\.\-]+\s+[0-9eE\.\-]+\))\s*\)\s*$'
)
MINED_PATTERN_ATOM_RE = re.compile(r'\([A-Za-z_][\w\-]*\s+\$[_\w\d]+\s+"[^"]*"\)')
MINED_PATTERN_STV_RE = re.compile(r"\(STV\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)")


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = (item or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def parse_petta_output(output: str) -> list[str]:
    """Parse the PeTTa stdout stream and keep only result lines."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)

    result_lines = []
    for line in clean_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-->") or line.startswith("prolog goal") or line.startswith("metta runnable"):
            continue
        if line.startswith("^^^^^"):
            continue
        result_lines.append(line)

    return result_lines


def parse_pattern_string(pattern_text: str) -> Optional[dict[str, str]]:
    """Parse `(supportOf ...)` output into a JSON-safe dict."""
    match = re.match(r'^\(supportOf\s+(.+)\s+(\d+)\)$', pattern_text.strip(), re.DOTALL)
    if not match:
        return None

    pattern_body = match.group(1).strip()
    support = match.group(2)

    if pattern_body.startswith("(") and pattern_body.endswith(")"):
        pattern_body = pattern_body[1:-1].strip()

    return {
        "pattern": pattern_body,
        "support": support,
    }


def patterns_to_chainer_rules(patterns: list[Any]) -> list[str]:

    query = f"""
    !(patterns->rules 
        {patterns}
    )
    """
    service = PeTTaService.create_required(
        PROJECT_ROOT,
        MINING_METTA_SETUP,
        verbose=False,
        load_chainer=False,
    )
    return service.query_lines(query)

def _balanced_expression_at(text: str, idx: int) -> Optional[str]:
    if idx < 0 or idx >= len(text) or text[idx] != "(":
        return None

    depth = 0
    in_string = False
    escaped = False

    for pos in range(idx, len(text)):
        char = text[pos]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[idx:pos + 1]

    return None


def extract_prefixed_expressions(text: str, prefix: str) -> list[str]:
    expressions = []
    start = 0
    while True:
        idx = text.find(prefix, start)
        if idx == -1:
            break
        expr = _balanced_expression_at(text, idx)
        if expr is None:
            break
        expressions.append(expr)
        start = idx + len(expr)
    return expressions


def extract_support_of_expressions(text: str) -> list[str]:
    return extract_prefixed_expressions(text, "(supportOf")


def is_pettachainer_fact_atom(expr: str) -> bool:
    stripped = (expr or "").strip()
    if not stripped.startswith("(:"):
        return False
    if "(STV " not in stripped:
        return False
    return True


def parse_facts_for_pettachainer(facts_output: Any) -> list[str]:
    if not facts_output:
        return []

    nested_facts = "\n".join(str(item) for item in facts_output) if isinstance(facts_output, list) else str(facts_output)
    matches = [expr for expr in extract_prefixed_expressions(nested_facts, "(:") if is_pettachainer_fact_atom(expr)]
    return unique_preserve_order(matches)


def load_dataset_facts_for_chainer(file_path: str) -> list[str]:
    facts: list[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = DATASET_FACT_LINE_RE.match(line)
            if not match:
                continue
            fact, stv = match.groups()
            proof_id = hashlib.sha256(fact.encode("utf-8")).hexdigest()[:24]
            facts.append(f'(: fact_{proof_id} {fact} {stv})')
    return unique_preserve_order(facts)


def select_facts_for_prompt(facts: list[str], query: str, limit: int = 80) -> list[str]:
    """Keep the LLM explanation prompt focused without starving the chainer."""
    if len(facts) <= limit:
        return facts

    terms = re.findall(r'A_[A-Za-z0-9_-]+|"[^"]+"', query or "")
    predicate_match = re.match(r"\s*\(\s*([A-Za-z_][\w\-]*)", query or "")
    if predicate_match:
        terms.append(f"({predicate_match.group(1)} ")
    terms = unique_preserve_order(terms)

    if not terms:
        return facts[:limit]

    scored = []
    for idx, fact in enumerate(facts):
        score = sum(1 for term in terms if term in fact)
        if score:
            scored.append((-score, idx, fact))

    selected = [fact for _, _, fact in sorted(scored)[:limit]]
    seen = set(selected)
    for fact in facts:
        if len(selected) >= limit:
            break
        if fact not in seen:
            selected.append(fact)
            seen.add(fact)

    return selected


def extract_parenthesized_expressions(text: str) -> list[str]:
    results = []
    start = 0
    while True:
        idx = text.find("(", start)
        if idx == -1:
            break
        expr = _balanced_expression_at(text, idx)
        if expr is None:
            break
        results.append(expr)
        start = idx + len(expr)
    return results


def make_json_safe(obj: Any) -> Any:
    """Recursively convert common non-JSON-serializable objects into JSON-safe types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Mapping):
        safe = {}
        for key, value in obj.items():
            safe_key = key if isinstance(key, (str, int, float, bool)) else str(key)
            safe[safe_key] = make_json_safe(value)
        return safe

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(item) for item in obj]

    try:
        data = getattr(obj, "__dict__", None)
        if isinstance(data, Mapping):
            return {str(key): make_json_safe(value) for key, value in data.items()}
    except Exception:
        pass

    try:
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            return make_json_safe(obj.to_dict())
        if hasattr(obj, "as_dict") and callable(getattr(obj, "as_dict")):
            return make_json_safe(obj.as_dict())
    except Exception:
        pass

    try:
        return str(obj)
    except Exception:
        return repr(obj)
