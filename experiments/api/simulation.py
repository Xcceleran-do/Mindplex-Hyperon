from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from typing import Any, Optional

from experiments.ingestion.utils import metta_predicate, normalize_property_name, sanitize_atom_id
from experiments.api.config import CHAINER_METTA_SETUP, DEFAULT_CHAIN_DEPTH, PROJECT_ROOT
from experiments.api.runtime import (
    dataset_facts,
    ordered_chainer_rules,
    reload_petta_dataset_if_ready,
)
from experiments.api.support import unique_preserve_order

ENGAGEMENT_FACT_RE = re.compile(
    r'^\(:\s+\(fact:-\s+\(engagement\s+[^\s()]+\s+"([^"]+)"\)\)'
)
STV_CAPTURE_RE = re.compile(r"\(STV\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)")
SIMULATION_FACT_RE = re.compile(
    r'^\(:\s+([^\s()]+)\s+'
    r'(\([A-Za-z_][\w\-]*\s+[^\s()]+\s+"[^"]*"\))\s+'
    r'\(STV\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)\s*\)$'
)
SIMULATION_RULE_ID_RE = re.compile(r"\brule_[A-Za-z0-9_\-]+\b")
SIMULATION_FACT_ID_RE = re.compile(r"\bsim_fact_\d+\b")
SIMULATION_ATOM_RE = re.compile(r'\(([A-Za-z_][\w\-]*)\s+([^\s()]+)\s+"([^"]*)"\)')
SIMULATION_ENGAGEMENT_LEVELS = ("High", "Medium", "Low")
SIMULATION_PREDICATE_ALIASES = {
    "length": ("length-bucket", "length"),
    "length-bucket": ("length-bucket", "length"),
}


def _predicate_is_active(predicate: str) -> bool:
    token = f"({predicate} "
    return any(token in fact for fact in dataset_facts()) or any(
        token in rule for rule in ordered_chainer_rules()
    )


def simulation_predicates_for(raw_predicate: str) -> list[str]:
    normalized = normalize_property_name(raw_predicate)
    aliases = SIMULATION_PREDICATE_ALIASES.get(normalized)
    if aliases:
        active = [predicate for predicate in aliases if _predicate_is_active(predicate)]
        if active:
            return list(unique_preserve_order(active))
        return [aliases[0]]
    return [metta_predicate(raw_predicate)]


def _escape_metta_string(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def build_simulation_fact_atoms(payload: dict, article_id: str) -> list[str]:
    attributes = payload.get("attributes")
    facts = payload.get("facts")
    if not attributes and not facts:
        raise ValueError("Simulation requires either 'attributes' or 'facts'.")

    fact_atoms: list[str] = []
    fact_counter = 0
    if isinstance(attributes, dict):
        for raw_predicate, raw_value in attributes.items():
            if isinstance(raw_value, dict):
                value = raw_value.get("value")
                strength = raw_value.get("strength", 1.0)
                confidence = raw_value.get("confidence", 1.0)
            else:
                value = raw_value
                strength = 1.0
                confidence = 1.0

            if value in (None, ""):
                continue
            for predicate in simulation_predicates_for(raw_predicate):
                fact_counter += 1
                fact_atoms.append(
                    f'(: sim_fact_{fact_counter} ({predicate} {article_id} "{_escape_metta_string(value)}") '
                    f'(STV {float(strength)} {float(confidence)}))'
                )

    if isinstance(facts, list):
        for item in facts:
            if not isinstance(item, dict):
                raise ValueError("Simulation facts must be objects with predicate and value.")
            raw_predicate = item.get("predicate")
            value = item.get("value")
            if not raw_predicate or value in (None, ""):
                continue
            strength = item.get("strength", 1.0)
            confidence = item.get("confidence", 1.0)
            for predicate in simulation_predicates_for(raw_predicate):
                fact_counter += 1
                fact_atoms.append(
                    f'(: sim_fact_{fact_counter} ({predicate} {article_id} "{_escape_metta_string(value)}") '
                    f'(STV {float(strength)} {float(confidence)}))'
                )

    return unique_preserve_order(fact_atoms)


def extract_stv_from_proof(proof: str) -> Optional[tuple[float, float]]:
    matches = STV_CAPTURE_RE.findall(proof or "")
    if not matches:
        return None
    strength, confidence = matches[-1]
    return float(strength), float(confidence)


# Temporary simulator bridge: exact proofs come from PeTTaChainer, but aggregate
# presentation still happens here until the simulator query is moved into MeTTa.
def truth_revision(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    def confidence_to_weight(confidence: float) -> float:
        if confidence >= 0.999999:
            return 1_000_000.0
        if confidence <= 0.0:
            return 0.0
        return confidence / max(1.0 - confidence, 1e-9)

    left_strength, left_confidence = left
    right_strength, right_confidence = right
    left_weight = confidence_to_weight(left_confidence)
    right_weight = confidence_to_weight(right_confidence)
    total_weight = left_weight + right_weight
    if total_weight <= 0.0:
        return 0.0, max(left_confidence, right_confidence)

    strength = ((left_weight * left_strength) + (right_weight * right_strength)) / total_weight
    derived_confidence = total_weight / (total_weight + 1.0)
    confidence = min(1.0, max(derived_confidence, left_confidence, right_confidence))
    return min(1.0, strength), confidence


def aggregate_proof_stvs(proofs: list[str]) -> Optional[tuple[float, float]]:
    aggregate: Optional[tuple[float, float]] = None
    for proof in proofs:
        stv = extract_stv_from_proof(proof)
        if stv is None:
            continue
        aggregate = stv if aggregate is None else truth_revision(aggregate, stv)
    return aggregate


def engagement_priors() -> dict[str, float]:
    counts = {level: 0 for level in SIMULATION_ENGAGEMENT_LEVELS}
    total = 0
    for fact in dataset_facts():
        match = ENGAGEMENT_FACT_RE.match(fact)
        if not match:
            continue
        level = match.group(1)
        if level in counts:
            counts[level] += 1
            total += 1

    if total <= 0:
        uniform = 1.0 / len(SIMULATION_ENGAGEMENT_LEVELS)
        return {level: uniform for level in SIMULATION_ENGAGEMENT_LEVELS}
    return {level: counts[level] / total for level in SIMULATION_ENGAGEMENT_LEVELS}


def normalize_bucket_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(score, 0.0) for score in scores.values())
    priors = engagement_priors()
    if total <= 0.0:
        return priors

    evidence_distribution = {
        bucket: max(score, 0.0) / total for bucket, score in scores.items()
    }
    evidence_mass = min(1.0, total)
    prior_mass = 1.0 - evidence_mass
    return {
        bucket: (evidence_distribution[bucket] * evidence_mass) + (priors[bucket] * prior_mass)
        for bucket in scores
    }


def _parse_simulation_fact(fact: str) -> dict[str, Any]:
    match = SIMULATION_FACT_RE.match(fact.strip())
    if not match:
        return {"id": "", "atom": fact, "strength": None, "confidence": None}
    fact_id, atom, strength, confidence = match.groups()
    return {
        "id": fact_id,
        "atom": atom,
        "strength": float(strength),
        "confidence": float(confidence),
    }


def _parse_simulation_rule(rule_atom: str) -> dict[str, Any]:
    rule_id_match = re.match(r"^\(:\s+([^\s()]+)", rule_atom.strip())
    stv = extract_stv_from_proof(rule_atom)
    atoms = [
        {
            "atom": match.group(0),
            "predicate": match.group(1),
            "subject": match.group(2),
            "value": match.group(3),
        }
        for match in SIMULATION_ATOM_RE.finditer(rule_atom)
    ]
    consequent = next((atom for atom in atoms if atom["predicate"] == "engagement"), None)
    antecedents = [atom for atom in atoms if atom is not consequent]
    return {
        "id": rule_id_match.group(1) if rule_id_match else "",
        "atom": rule_atom,
        "antecedents": antecedents,
        "consequent": consequent,
        "stv": (
            {"strength": stv[0], "confidence": stv[1]}
            if stv is not None
            else None
        ),
    }


def _ground_rule_atom(atom: dict[str, str], article_id: str) -> str:
    return f'({atom["predicate"]} {article_id} "{atom["value"]}")'


def _fact_key_from_atom_text(atom_text: str) -> Optional[tuple[str, str]]:
    match = SIMULATION_ATOM_RE.search(atom_text or "")
    if not match:
        return None
    return match.group(1), match.group(3)


def _rule_atom_key(atom: dict[str, str]) -> tuple[str, str]:
    return atom["predicate"], atom["value"]


def _score_stv(stv: Optional[tuple[float, float]]) -> float:
    if stv is None:
        return 0.0
    return max(0.0, min(1.0, stv[0])) * max(0.0, min(1.0, stv[1]))


def _combine_rule_and_fact_stvs(
    rule_stv: Optional[tuple[float, float]],
    matched_facts: list[dict[str, Any]],
) -> tuple[float, float]:
    strength, confidence = rule_stv if rule_stv is not None else (1.0, 1.0)
    for fact in matched_facts:
        fact_strength = fact.get("strength")
        fact_confidence = fact.get("confidence")
        strength *= float(fact_strength if fact_strength is not None else 1.0)
        confidence *= float(fact_confidence if fact_confidence is not None else 1.0)
    return max(0.0, min(1.0, strength)), max(0.0, min(1.0, confidence))


def build_conditional_rule_matches(*,
    article_id: str,
    hypothetical_facts: list[str],
    rule_atoms: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Optional[tuple[float, float]]]]:
    fact_details = [_parse_simulation_fact(fact) for fact in hypothetical_facts]
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    selected_values_by_predicate: dict[str, set[str]] = {}
    for fact in fact_details:
        key = _fact_key_from_atom_text(fact.get("atom", ""))
        if not key:
            continue
        selected_by_key[key] = fact
        selected_values_by_predicate.setdefault(key[0], set()).add(key[1])

    suggestions_by_level: dict[str, list[dict[str, Any]]] = {
        level: [] for level in SIMULATION_ENGAGEMENT_LEVELS
    }
    stv_by_level: dict[str, Optional[tuple[float, float]]] = {
        level: None for level in SIMULATION_ENGAGEMENT_LEVELS
    }

    for rule_atom in rule_atoms:
        rule = _parse_simulation_rule(rule_atom)
        consequent = rule.get("consequent")
        if not consequent or consequent.get("predicate") != "engagement":
            continue
        level = consequent.get("value")
        if level not in suggestions_by_level:
            continue

        matched = []
        missing = []
        conflict = False
        for antecedent in rule.get("antecedents", []):
            antecedent_key = _rule_atom_key(antecedent)
            if antecedent_key in selected_by_key:
                matched.append({
                    "required": _ground_rule_atom(antecedent, article_id),
                    "fact": selected_by_key[antecedent_key],
                })
                continue

            selected_values = selected_values_by_predicate.get(antecedent_key[0], set())
            if selected_values and antecedent_key[1] not in selected_values:
                conflict = True
                break
            missing.append(_ground_rule_atom(antecedent, article_id))

        if conflict or not matched:
            continue

        rule_stv = extract_stv_from_proof(rule_atom)
        conditional_stv = _combine_rule_and_fact_stvs(
            rule_stv,
            [item["fact"] for item in matched],
        )
        suggestion = {
            "rule_id": rule.get("id"),
            "consequent": consequent,
            "matched_antecedents": matched,
            "missing_antecedents": missing,
            "assumed_antecedents": missing,
            "rule": rule,
            "conditional_stv": {
                "strength": conditional_stv[0],
                "confidence": conditional_stv[1],
            },
            "conditional_score": _score_stv(conditional_stv),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "summary": (
                f"If you also use {len(missing)} missing attribute"
                f"{'' if len(missing) == 1 else 's'}, this rule predicts {level}."
            ),
        }
        suggestions_by_level[level].append(suggestion)
        stv_by_level[level] = (
            conditional_stv
            if stv_by_level[level] is None
            else truth_revision(stv_by_level[level], conditional_stv)
        )

    for level in SIMULATION_ENGAGEMENT_LEVELS:
        suggestions_by_level[level].sort(
            key=lambda item: (
                item["conditional_score"],
                item["matched_count"],
                -item["missing_count"],
            ),
            reverse=True,
        )

    return suggestions_by_level, stv_by_level


def build_simulation_explanation(
    *,
    article_id: str,
    hypothetical_facts: list[str],
    rule_atoms: list[str],
    proofs_by_level: dict[str, list[str]],
    used_prior_fallback: bool,
    conditional_suggestions_by_level: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    fact_details = [_parse_simulation_fact(fact) for fact in hypothetical_facts]
    facts_by_id = {fact["id"]: fact for fact in fact_details if fact.get("id")}
    facts_by_atom = {fact["atom"]: fact for fact in fact_details if fact.get("atom")}
    rule_details = [_parse_simulation_rule(rule) for rule in rule_atoms]
    rules_by_id = {rule["id"]: rule for rule in rule_details if rule.get("id")}
    conditional_suggestions_by_level = conditional_suggestions_by_level or {
        level: [] for level in SIMULATION_ENGAGEMENT_LEVELS
    }

    chains_by_level: dict[str, list[dict[str, Any]]] = {}
    for level in SIMULATION_ENGAGEMENT_LEVELS:
        chains = []
        for proof in proofs_by_level.get(level, []):
            rule_id_match = SIMULATION_RULE_ID_RE.search(proof)
            rule_id = rule_id_match.group(0) if rule_id_match else ""
            fact_ids = unique_preserve_order(SIMULATION_FACT_ID_RE.findall(proof))
            stv = extract_stv_from_proof(proof)
            chains.append({
                "proof": proof,
                "rule_id": rule_id,
                "rule": rules_by_id.get(rule_id),
                "facts": [facts_by_id[fact_id] for fact_id in fact_ids if fact_id in facts_by_id],
                "stv": (
                    {"strength": stv[0], "confidence": stv[1]}
                    if stv is not None
                    else None
                ),
            })
        chains_by_level[level] = chains

    unmatched_rules = []
    for rule in rule_details:
        matched = []
        missing = []
        for antecedent in rule.get("antecedents", []):
            grounded = _ground_rule_atom(antecedent, article_id)
            if grounded in facts_by_atom:
                matched.append({"required": grounded, "fact": facts_by_atom[grounded]})
            else:
                missing.append(grounded)
        unmatched_rules.append({
            "rule_id": rule.get("id"),
            "consequent": rule.get("consequent"),
            "matched_antecedents": matched,
            "missing_antecedents": missing,
            "rule": rule,
        })

    has_exact_chains = any(proofs_by_level.get(level) for level in SIMULATION_ENGAGEMENT_LEVELS)
    has_conditional_suggestions = any(
        conditional_suggestions_by_level.get(level)
        for level in SIMULATION_ENGAGEMENT_LEVELS
    )

    if used_prior_fallback:
        summary = (
            "No mined rule matched the selected hypothetical facts. "
            "The probabilities are historical engagement priors, not a rule proof."
        )
    elif has_exact_chains:
        summary = "At least one mined rule fired; probabilities were normalized from proof STVs."
    elif has_conditional_suggestions:
        summary = (
            "No complete rule fired yet, but compatible partial rules were scored as conditional "
            "what-if suggestions using the selected fact STVs."
        )
    else:
        summary = "Probabilities were normalized from mined rule STV scores."

    return {
        "summary": summary,
        "input_facts": fact_details,
        "rules": rule_details,
        "chains_by_level": chains_by_level,
        "unmatched_rules": unmatched_rules,
        "conditional_suggestions_by_level": conditional_suggestions_by_level,
        "conditional_suggestions": [
            suggestion
            for level in SIMULATION_ENGAGEMENT_LEVELS
            for suggestion in conditional_suggestions_by_level.get(level, [])[:4]
        ],
    }


def run_simulation_worker(payload: dict[str, Any]) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "experiments.api.workers.simulation"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
    )
    if process.returncode != 0:
        return {
            "status": "error",
            "message": "Simulation worker failed.",
            "stderr": process.stderr.strip(),
            "stdout": process.stdout.strip(),
        }

    stdout = (process.stdout or "").strip()
    if not stdout:
        return {"status": "error", "message": "Simulation worker produced no output."}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Simulation worker returned invalid JSON.",
            "stdout": stdout,
            "stderr": process.stderr.strip(),
        }


# Architecture debt: conditional what-if scoring should become a PeTTaChainer
# simulation query. Python should only build hypothetical facts/rules, call
# PeTTa, and serialize the returned proof/STV explanation.
def simulate_engagement(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_reload = reload_petta_dataset_if_ready(force=False)
    rules = ordered_chainer_rules()
    if not rules:
        return {
            "status": "error",
            "message": "No mined rules are loaded. Run /api/mine first.",
            "dataset": dataset_reload,
        }

    raw_article_id = str(payload.get("article_id") or f"sim_{uuid.uuid4().hex[:8]}")
    if raw_article_id.startswith("H_"):
        raw_article_id = raw_article_id[2:]
    article_id = f'H_{sanitize_atom_id(raw_article_id)}'
    depth = int(payload.get("depth", DEFAULT_CHAIN_DEPTH))
    if depth < 1:
        raise ValueError("depth must be a positive integer")

    hypothetical_facts = build_simulation_fact_atoms(payload, article_id)
    if not hypothetical_facts:
        raise ValueError("No valid hypothetical facts were provided for simulation.")

    worker_payload = {
        "project_root": PROJECT_ROOT,
        "setup_metta": CHAINER_METTA_SETUP,
        "base_facts": dataset_facts(),
        "rules": rules,
        "hypothetical_facts": hypothetical_facts,
        "article_id": article_id,
        "depth": depth,
        "engagement_levels": list(SIMULATION_ENGAGEMENT_LEVELS),
    }
    worker_result = run_simulation_worker(worker_payload)
    if worker_result.get("status") != "success":
        worker_result["dataset"] = dataset_reload
        return worker_result

    proofs_by_level = worker_result.get("proofs_by_level", {})
    conditional_suggestions_by_level, conditional_stv_by_level = build_conditional_rule_matches(
        article_id=article_id,
        hypothetical_facts=hypothetical_facts,
        rule_atoms=rules,
    )
    bucket_results: dict[str, Any] = {}
    raw_scores: dict[str, float] = {}
    proof_count = 0
    for level in SIMULATION_ENGAGEMENT_LEVELS:
        proofs = proofs_by_level.get(level, [])
        aggregated_stv = aggregate_proof_stvs(proofs)
        exact_score = _score_stv(aggregated_stv)
        conditional_stv = conditional_stv_by_level.get(level)
        conditional_score = _score_stv(conditional_stv)
        score = max(exact_score, conditional_score)
        raw_scores[level] = score
        proof_count += len(proofs)
        bucket_results[level] = {
            "proofs": proofs,
            "proof_count": len(proofs),
            "aggregated_stv": (
                {"strength": aggregated_stv[0], "confidence": aggregated_stv[1]}
                if aggregated_stv is not None
                else None
            ),
            "conditional_stv": (
                {"strength": conditional_stv[0], "confidence": conditional_stv[1]}
                if conditional_stv is not None
                else None
            ),
            "conditional_score": conditional_score,
            "conditional_suggestions": conditional_suggestions_by_level.get(level, [])[:6],
            "exact_score": exact_score,
            "raw_score": score,
        }

    probabilities = normalize_bucket_scores(raw_scores)
    used_prior_fallback = sum(raw_scores.values()) <= 0.0
    best_level = max(probabilities.items(), key=lambda item: item[1])[0]
    for level, probability in probabilities.items():
        bucket_results[level]["probability"] = probability
    explanation = build_simulation_explanation(
        article_id=article_id,
        hypothetical_facts=hypothetical_facts,
        rule_atoms=rules,
        proofs_by_level=proofs_by_level,
        used_prior_fallback=used_prior_fallback,
        conditional_suggestions_by_level=conditional_suggestions_by_level,
    )

    return {
        "status": "success",
        "article_id": article_id,
        "depth_used": depth,
        "dataset": dataset_reload,
        "rules_used": len(rules),
        "used_prior_fallback": used_prior_fallback,
        "proof_count": proof_count,
        "input_facts": hypothetical_facts,
        "probabilities": probabilities,
        "predicted_engagement": best_level,
        "buckets": bucket_results,
        "explanation": explanation,
    }
