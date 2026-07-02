#!/usr/bin/env python3
"""
Unified API Server
A Flask-based API server that exposes pattern mining and AI chat functionality
"""

import os
import sys
import json
import time
import re
import signal
import subprocess
import tempfile
import requests
from flask import Flask
from flask_cors import CORS
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Add workspace root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from experiments.ingestion.pipeline import resolve_output_path, run_ingestion
from experiments.ingestion.utils import metta_predicate, normalize_property_name, sanitize_atom_id
from experiments.services.petta_service import (
    PeTTaService,
    PeTTaStartupError,
    format_proofs_for_prompt,
    unique_preserve_order,
)
from experiments.chat_api_prompts import (
    SYSTEM_INSTRUCTION,
    build_chainer_analysis_prompt,
    build_tools_schema,
)
from experiments.chat_api_support import (
    analyze_pattern as analyze_pattern_impl,
    build_rule_grounded_summary,
    handle_backward_chain_for_message as handle_backward_chain_for_message_impl,
    handle_mining_for_message as handle_mining_for_message_impl,
    is_backward_chain_intent,
    parse_chat_mining_intent as parse_chat_mining_intent_impl,
    parse_pattern as parse_pattern_impl,
    register_chat_routes,
    summarize_patterns as summarize_patterns_impl,
)
from experiments.mining_api_routes import register_core_routes
from experiments.mining_api_support import (
    compile_facts_into_chainer,
    extract_parenthesized_expressions,
    extract_support_of_expressions,
    load_dataset_facts_for_chainer,
    make_json_safe,
    parse_facts_for_pettachainer,
    parse_pattern_string,
    parse_petta_output,
    select_facts_for_prompt,
)
from experiments.omegaclaw_bridge import (
    is_omegaclaw_chat_enabled,
    send_chat_to_omegaclaw,
)
load_dotenv()

# Configure ASI API
ASI_API_KEY = os.getenv("ASI_API_KEY")
if not ASI_API_KEY:
    logger.warning("ASI_API_KEY environment variable is not set. AI features will fail.")
ASI_BASE_URL = "https://api.asi1.ai/v1/chat/completions"
ASI_MODEL = "asi1-mini"
ASI_TIMEOUT_SECONDS = float(os.getenv("ASI_TIMEOUT_SECONDS", "45"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_METTA = os.path.abspath(PROJECT_ROOT).replace('\\', '/')

DEFAULT_CONJUNCTION_COUNT = 2
DEFAULT_MIN_SUPPORT = 3
DEFAULT_MAX_CONJUNCTION_COUNT = int(os.getenv("PETTA_MAX_CONJUNCTION_COUNT", "10"))
DEFAULT_CHAIN_DEPTH = int(os.getenv("PETTA_CHAIN_DEPTH", "3"))
PETTA_MINING_TIMEOUT_SECONDS = int(os.getenv("PETTA_MINING_TIMEOUT_SECONDS", "90"))
PETTA_MINING_MAX_OUTPUT_BYTES = int(os.getenv("PETTA_MINING_MAX_OUTPUT_BYTES", str(8 * 1024 * 1024)))
PETTA_CHAIN_TIMEOUT_SECONDS = int(os.getenv("PETTA_CHAIN_TIMEOUT_SECONDS", "30"))


def dataset_file_path() -> str:
    output_path = resolve_output_path()
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    return os.path.abspath(output_path)


def dataset_module_path() -> str:
    path = dataset_file_path()
    if path.endswith(".metta"):
        path = path[:-6]
    return path.replace("\\", "/")

MINING_METTA_SETUP = f"""
!(import! &self {PROJECT_ROOT_METTA}/PeTTa/lib/lib_import.metta)
!(import! &self {PROJECT_ROOT_METTA}/PeTTa/lib/lib_spaces)
!(import_prolog_functions_from_file "{PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/conj_exp.pl" (unique_combinations_star cut-first-char promote_engagement_conj))
!(import! &self {PROJECT_ROOT_METTA}/experiments/utils/common-utils)
!(import! &self {PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/etv-utils)
!(import! &self {PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/frequent-pattern-miner)
!(import! &self {PROJECT_ROOT_METTA}/experiments/pattern-miner/pattern-miner)
"""

CHAINER_METTA_SETUP = f"""
!(import! &self {PROJECT_ROOT_METTA}/experiments/utils/common-utils)
!(import! &self {PROJECT_ROOT_METTA}/experiments/frequent-pattern-miner/etv-utils)
"""

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

chainer_service: Optional[PeTTaService] = None
chainer_dataset_path: Optional[str] = None
chainer_dataset_mtime: Optional[float] = None
chainer_dataset_facts: list[str] = []
chainer_dataset_compile_errors: list[dict[str, str]] = []
chainer_dataset_compiled_count: int = 0
chainer_rule_atoms: list[str] = []
runtime_lock = threading.Lock()
tools_schema = build_tools_schema(DEFAULT_CHAIN_DEPTH)


def ordered_chainer_rules() -> list[str]:
    """PeTTaChainer rule aggregation is order-sensitive; keep ordering stable."""
    return sorted(unique_preserve_order(chainer_rule_atoms))


def _dataset_runtime_payload(status: str, dataset_path: str, dataset_mtime: float) -> dict[str, Any]:
    service = get_chainer_service()
    dataset_errors = list(chainer_dataset_compile_errors)
    dataset_facts = list(chainer_dataset_facts)
    return {
        "status": status,
        "dataset_path": dataset_path,
        "dataset_mtime": dataset_mtime,
        "fact_count": len(dataset_facts),
        "compiled_fact_count": chainer_dataset_compiled_count,
        "compile_error_count": len(dataset_errors),
        "compile_errors": dataset_errors[:5],
        "chainer": service.health(),
        "mining": {"mode": "subprocess"},
    }

def _create_chainer_service() -> PeTTaService:
    return PeTTaService.create_required(
        project_root=PROJECT_ROOT,
        setup_metta=CHAINER_METTA_SETUP,
        verbose=False,
    )


def bootstrap_runtime() -> PeTTaService:
    """Initialize the lean chainer runtime and preload the current dataset once."""
    reload_petta_dataset_if_ready(force=False)
    return get_chainer_service()


def get_chainer_service() -> PeTTaService:
    if chainer_service is None:
        raise PeTTaStartupError(
            "Chainer runtime has not been initialized. Start the app through create_app() "
            "or call bootstrap_runtime() before serving requests."
        )
    return chainer_service


def get_petta_service() -> PeTTaService:
    return get_chainer_service()


def reload_petta_dataset_if_ready(force: bool = False) -> dict:
    """Ensure the lean chainer runtime matches the current dataset on disk."""
    global chainer_service, chainer_dataset_path, chainer_dataset_mtime
    global chainer_dataset_facts, chainer_dataset_compile_errors, chainer_dataset_compiled_count
    global chainer_rule_atoms
    file_path = dataset_file_path()
    current_mtime = os.path.getmtime(file_path)
    changed = (
        force
        or chainer_service is None
        or chainer_dataset_path != file_path
        or chainer_dataset_mtime != current_mtime
    )
    if changed:
        with runtime_lock:
            file_path = dataset_file_path()
            current_mtime = os.path.getmtime(file_path)
            changed = (
                force
                or chainer_service is None
                or chainer_dataset_path != file_path
                or chainer_dataset_mtime != current_mtime
            )
            if changed:
                status = "initialized" if chainer_service is None and not force else "reinitialized"
                logger.info("Syncing chainer runtime from dataset: %s", file_path)
                service = _create_chainer_service()
                facts = load_dataset_facts_for_chainer(file_path)
                compiled_count, compile_errors = compile_facts_into_chainer(service, facts)
                service.set_dataset_metadata(
                    dataset_module_path=dataset_module_path(),
                    dataset_file_path=file_path,
                    dataset_mtime=current_mtime,
                )

                chainer_service = service
                chainer_dataset_path = file_path
                chainer_dataset_mtime = current_mtime
                chainer_dataset_facts = facts
                chainer_dataset_compile_errors = compile_errors
                chainer_dataset_compiled_count = compiled_count
                chainer_rule_atoms = []
                logger.info(
                    "Chainer runtime %s: facts=%s compiled=%s errors=%s health=%s",
                    status,
                    len(chainer_dataset_facts),
                    chainer_dataset_compiled_count,
                    len(chainer_dataset_compile_errors),
                    chainer_service.health(),
                )
            else:
                status = "unchanged"
    else:
        status = "unchanged"

    return _dataset_runtime_payload(status, file_path, current_mtime)
def call_asi_api(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Calls the ASI API with the given messages and tools."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ASI_API_KEY}"
    }
    payload = {
        "model": ASI_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(ASI_BASE_URL, headers=headers, json=payload, timeout=ASI_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        if hasattr(exc, "response") and exc.response is not None:
            logger.warning("ASI API request failed: %s | response=%s", exc, exc.response.text)
        else:
            logger.warning("ASI API request failed: %s", exc)
        return {"error": str(exc)}


def run_metta_with_petta(metta_code: str) -> str:
    """
    Run mining MeTTa in a fresh worker process so heavy mining imports do not
    poison the persistent chainer runtime.
    """
    worker_code = f"""
import sys
from experiments.services.petta_service import PeTTaService

service = PeTTaService.create_required(
    project_root={PROJECT_ROOT!r},
    setup_metta={MINING_METTA_SETUP!r},
    verbose=False,
)
service.reload_dataset({dataset_module_path()!r}, {dataset_file_path()!r})
result = service.run_metta_string({metta_code!r})
sys.stdout.write(result)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    stdout_path = ""
    stderr_path = ""
    process: Optional[subprocess.Popen] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="petta-mine-out-", delete=False) as stdout_file:
            stdout_path = stdout_file.name
        with tempfile.NamedTemporaryFile(prefix="petta-mine-err-", delete=False) as stderr_file:
            stderr_path = stderr_file.name

        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            process = subprocess.Popen(
                [sys.executable, "-c", worker_code],
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
            )

            started_at = time.monotonic()
            killed_reason: Optional[str] = None
            while process.poll() is None:
                elapsed = time.monotonic() - started_at
                output_size = os.path.getsize(stdout_path)
                error_size = os.path.getsize(stderr_path)

                if elapsed > PETTA_MINING_TIMEOUT_SECONDS:
                    killed_reason = f"exceeded {PETTA_MINING_TIMEOUT_SECONDS}s timeout"
                    break
                if output_size > PETTA_MINING_MAX_OUTPUT_BYTES:
                    killed_reason = (
                        f"produced more than {PETTA_MINING_MAX_OUTPUT_BYTES} bytes of output"
                    )
                    break
                if error_size > PETTA_MINING_MAX_OUTPUT_BYTES:
                    killed_reason = (
                        f"produced more than {PETTA_MINING_MAX_OUTPUT_BYTES} bytes of stderr"
                    )
                    break

                time.sleep(0.25)

            if killed_reason:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                raise TimeoutError(f"Mining worker {killed_reason}.")

            process.wait()

        with open(stdout_path, "rb") as handle:
            stdout = handle.read().decode("utf-8", errors="replace")
        with open(stderr_path, "rb") as handle:
            stderr = handle.read().decode("utf-8", errors="replace")

        if process.returncode != 0:
            stderr = stderr.strip() or "unknown mining worker failure"
            raise RuntimeError(f"Mining worker failed: {stderr}")
        return stdout
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for path in (stdout_path, stderr_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


def _predicate_is_active(predicate: str) -> bool:
    token = f"({predicate} "
    return any(token in fact for fact in chainer_dataset_facts) or any(token in rule for rule in chainer_rule_atoms)


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
        for idx, item in enumerate(facts, start=1):
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
    for fact in chainer_dataset_facts:
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


def build_conditional_rule_matches(
    *,
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
        summary = "At least one mined rule fired; probabilities were normalized from proof and STV scores."
    elif has_conditional_suggestions:
        summary = (
            "No complete rule fired yet, but compatible partial rules were scored as conditional "
            "what-if suggestions using the selected fact STVs."
        )
    else:
        summary = "Probabilities were normalized from mined rule STV scores."

    conditional_suggestions_by_level = conditional_suggestions_by_level or {
        level: [] for level in SIMULATION_ENGAGEMENT_LEVELS
    }

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
        [sys.executable, "-m", "experiments.simulation_worker"],
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
        "base_facts": list(chainer_dataset_facts),
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

def mine_pattern(numberOfConjunction: int, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    """
    Mines patterns with a specified number of conjunctions using PeTTa.

    Args:
        numberOfConjunction: The number of conjunctions to use in pattern mining.
        min_support: Minimum support threshold used by the PeTTa miner.

    Returns:
        A dictionary containing the mining results with parsed patterns.
    """
    try:
        dataset_reload = reload_petta_dataset_if_ready(force=False)
        numberOfConjunction = int(numberOfConjunction)
        min_support = int(min_support)
        if numberOfConjunction < 1:
            return {
                "status": "error",
                "message": "numberOfConjunction must be a positive integer",
            }
        if numberOfConjunction > DEFAULT_MAX_CONJUNCTION_COUNT:
            return {
                "status": "error",
                "message": (
                    f"numberOfConjunction must be <= {DEFAULT_MAX_CONJUNCTION_COUNT}. "
                    "Use PETTA_MAX_CONJUNCTION_COUNT to raise this limit after validating the dataset."
                ),
            }
        if min_support < 1:
            return {
                "status": "error",
                "message": "min_support must be a positive integer",
            }

        query = f"!(pattern-miner &purifiedDbSpace {min_support} {numberOfConjunction})"
        petta_output = run_metta_with_petta(query)
        normalized_query = query.strip().lstrip("!").strip()
        if petta_output.strip() == normalized_query:
            return {
                "status": "error",
                "message": "PeTTa returned the unevaluated expression. The runnable may not have executed.",
                "raw_result": petta_output
            }
        result_lines = parse_petta_output(petta_output)
        
        patterns = []
        full_answer_str = " ".join(result_lines)
        support_matches = extract_support_of_expressions(full_answer_str)

        for match in support_matches:
            parsed = parse_pattern_string(match)
            if parsed:
                patterns.append(parsed)
        
        if not patterns:
            return {"status": "no_results", "patterns": [], "dataset": dataset_reload}
        
        return {
            "answer": full_answer_str,
            "status": "success",
            "conjunction_count": numberOfConjunction,
            "min_support": min_support,
            "dataset": dataset_reload,
            "patterns": patterns,
            "total_count": len(patterns)
        }
        
    except Exception as exc:
        logger.exception("mine_pattern failed")
        return {
            "status": "error",
            "message": f"Failed to run pattern mining or parse result: {str(exc)}",
            "raw_result": locals().get('petta_output', 'Command failed before output')
        }


app = Flask(__name__)
# Enable CORS for all domains on all routes with all methods
CORS(app, resources={r"/api/*": {
    "origins": "*",  # Allow all origins
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "expose_headers": ["Content-Type"],
    "supports_credentials": False,
    "max_age": 3600
}})

def getAllFactsAndRules():
    """Return current facts and rules from the MeTTa knowledge base.

    The assistant should call this before attempting backward chaining so it
    can rewrite a user's natural-language question into a canonical MeTTa
    query that matches predicates/constants present in the KB. Example:
    user: "What is article 1's engagement level?"
    assistant: call getAllFactsAndRules(), notice atoms like `(engagement 1 "high")`,
    rewrite as `(engagement 1 $whatIsIt)`, then call getChainerResult.
    """
    try:
        dataset_reload = reload_petta_dataset_if_ready(force=False)
        aligned_facts = list(chainer_dataset_facts)
        compile_errors = list(chainer_dataset_compile_errors)
        compiled_count = chainer_dataset_compiled_count

        if aligned_facts and len(compile_errors) == len(aligned_facts):
            return {
                "status": "error",
                "error": "Failed to compile every fact into PeTTaChainer.",
                "dataset": dataset_reload,
                "fact_count": len(aligned_facts),
                "compile_errors": compile_errors[:5],
            }

        return {
            "status": "success",
            "dataset": dataset_reload,
            "facts": aligned_facts,
            "fact_count": len(aligned_facts),
            "compiled_new_count": compiled_count,
            "compile_error_count": len(compile_errors),
            "compile_errors": compile_errors[:5],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def parse_chat_mining_intent(message: str) -> Optional[Dict[str, int]]:
    return parse_chat_mining_intent_impl(
        message,
        default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
        default_min_support=DEFAULT_MIN_SUPPORT,
    )


def handle_mining_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    return handle_mining_for_message_impl(
        message,
        default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
        default_min_support=DEFAULT_MIN_SUPPORT,
        start_mining_job=start_mining_job,
        summarize_patterns=summarize_patterns,
    )


def handle_backward_chain_for_message(message: str) -> tuple[Optional[str], Optional[list]]:
    return handle_backward_chain_for_message_impl(
        message,
        get_all_facts_and_rules=getAllFactsAndRules,
        select_facts_for_prompt=select_facts_for_prompt,
        call_asi_api=call_asi_api,
        system_instruction=SYSTEM_INSTRUCTION,
        get_chainer_result=getChainerResult,
        logger=logger,
    )


# Define available functions for the AI with proper docstrings for automatic function calling
def get_mining_results() -> dict:
    """Retrieves the latest pattern mining results from the system.
    
    Use this when the user asks about mining results, patterns found, or says "Mine rules with X patterns".
    
    Returns:
        A dictionary containing all patterns with their indices, support values, and properties.
    """
    jobs = list(mining_jobs.values())
    if not jobs:
        return {"status": "no_results", "message": "No mining jobs have been run yet."}
    
    latest_job = max(jobs, key=lambda j: j.start_time)
    if latest_job.status != 'completed':
        return {"status": "not_ready", "message": f"Latest job is {latest_job.status}"}
    
    # Parse all patterns to extract detailed information
    patterns_data = []
    if latest_job.result and isinstance(latest_job.result, dict):
        # Get patterns from the dict returned by mine_pattern()
        patterns = latest_job.result.get('patterns', [])
        for idx, item in enumerate(patterns, 1):
            pattern = item.get('pattern', '')
            support = item.get('support', '0')
            properties = parse_pattern(pattern)
            patterns_data.append({
                "index": idx,
                "pattern": pattern,
                "support": support,
                "properties": properties
            })
    
    return {
        "status": "success",
        "patterns": patterns_data,
        "total_count": len(patterns_data),
        "conjunction_size": latest_job.conjunction_count,
        "min_support": latest_job.min_support,
        "rules": latest_job.result.get("rules", []) if isinstance(latest_job.result, dict) else [],
        "rule_insertion": latest_job.result.get("rule_insertion") if isinstance(latest_job.result, dict) else None,
    }

def analyze_specific_pattern(pattern: str) -> dict:
    """Analyzes a specific pattern in detail, extracting properties and values.
    
    Args:
        pattern: The pattern string to analyze, e.g., '((length-bucket $x "low") (engagement $x "high"))'
        
    Returns:
        A dictionary with pattern analysis including properties and their values.
    """
    properties = parse_pattern(pattern)
    return {
        "pattern": pattern,
        "properties": properties,
        "property_count": len(properties),
        "description": f"Pattern with {len(properties)} properties: {', '.join(properties.keys())}"
    }

def get_pattern_statistics() -> dict:
    """Gets statistics about all mining results including total jobs and patterns.
    
    Returns:
        A dictionary with statistics about all completed mining jobs.
    """
    jobs = [j for j in mining_jobs.values() if j.status == 'completed']
    if not jobs:
        return {"status": "no_data", "message": "No completed mining jobs"}
    
    # Each job.result is expected to be the dict returned by mine_pattern().
    # Count how many patterns are stored under the 'patterns' key for each job.
    total_patterns = sum(
        (len(j.result.get('patterns', [])) if isinstance(j.result, dict) else 0)
        for j in jobs
    )

    return {
        "total_jobs": len(jobs),
        "total_patterns": total_patterns,
        "average_patterns_per_job": total_patterns / len(jobs) if jobs else 0
    }

def visualize_pattern_request(pattern: str) -> dict:
    """Requests visualization of a specific pattern on the graph canvas.
    
    Args:
        pattern: The pattern string to visualize
        
    Returns:
        A confirmation message that the pattern will be visualized.
    """
    return {
        "action": "visualize",
        "pattern": pattern,
        "message": "Pattern visualization requested. The frontend will display this pattern."
    }

def insert_mined_rules_into_chainer(mining_result: dict) -> dict:
    """Compile mined patterns into PeTTa rules before returning mining results."""
    global chainer_rule_atoms
    if not isinstance(mining_result, dict):
        return mining_result
    if mining_result.get("status") != "success":
        return mining_result

    patterns = mining_result.get("patterns", [])
    if not patterns:
        mining_result["rule_insertion"] = {
            "status": "no_rules",
            "insertedRuleCount": 0,
            "rules": [],
        }
        mining_result["rules"] = []
        mining_result["inserted_rule_count"] = 0
        return mining_result

    insertion_result = get_chainer_service().formatter({"patterns": patterns})
    mining_result["rule_insertion"] = insertion_result
    mining_result["rules"] = insertion_result.get("rules", []) if isinstance(insertion_result, dict) else []
    mining_result["inserted_rule_count"] = (
        insertion_result.get("insertedRuleCount", 0)
        if isinstance(insertion_result, dict)
        else 0
    )
    if isinstance(insertion_result, dict) and insertion_result.get("status") == "success":
        chainer_rule_atoms = unique_preserve_order(
            [*chainer_rule_atoms, *insertion_result.get("rules", [])]
        )
    return mining_result

def run_mining_task_inprocess(conjunction_count: int, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    result = mine_pattern(conjunction_count, min_support)
    return insert_mined_rules_into_chainer(result)


def run_mining_task(job_id: str, conjunction_count: int, min_support: int = DEFAULT_MIN_SUPPORT):
    """
    Run the mining task for a given job.
    Args:
        job_id (str): Unique identifier for the mining job.
        conjunction_count (int): Number of conjunctions to use in the mining process.
        min_support (int): Minimum support threshold for returned patterns.
    Returns:
        dict: A dictionary containing the job status, result, error (if any), and timestamps.
    """
    global chainer_rule_atoms
    job = mining_jobs[job_id]
    job.start_time = time.time()
    job.min_support = min_support

    result_path = ""
    stdout_path = ""
    stderr_path = ""
    process: Optional[subprocess.Popen] = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-", suffix=".json", delete=False) as result_file:
            result_path = result_file.name
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-out-", delete=False) as stdout_file:
            stdout_path = stdout_file.name
        with tempfile.NamedTemporaryFile(prefix=f"petta-job-{job_id}-err-", delete=False) as stderr_file:
            stderr_path = stderr_file.name

        with open(stdout_path, "wb") as stdout_file, open(stderr_path, "wb") as stderr_file:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.mining_job_worker",
                    "--conjunction-count",
                    str(conjunction_count),
                    "--min-support",
                    str(min_support),
                    "--result-path",
                    result_path,
                ],
                cwd=PROJECT_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                env={
                    **os.environ,
                    "PYTHONPATH": PROJECT_ROOT
                    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                },
                start_new_session=True,
            )

            started_at = time.monotonic()
            while process.poll() is None:
                elapsed = time.monotonic() - started_at
                stdout_size = os.path.getsize(stdout_path)
                stderr_size = os.path.getsize(stderr_path)
                result_size = os.path.getsize(result_path)

                if elapsed > PETTA_MINING_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Mining worker exceeded {PETTA_MINING_TIMEOUT_SECONDS}s timeout.")
                if max(stdout_size, stderr_size, result_size) > PETTA_MINING_MAX_OUTPUT_BYTES:
                    raise RuntimeError(
                        f"Mining worker output exceeded {PETTA_MINING_MAX_OUTPUT_BYTES} bytes."
                    )
                time.sleep(0.25)

            if process.returncode != 0:
                with open(stderr_path, "rb") as handle:
                    stderr = handle.read(4096).decode("utf-8", errors="replace").strip()
                raise RuntimeError(stderr or f"Mining worker exited with code {process.returncode}.")

        with open(result_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)

        job.result = result
        if isinstance(result, dict) and result.get("status") == "error":
            job.status = 'error'
            job.error = result.get("message", "Mining failed")
        else:
            job.status = 'completed'
            if isinstance(result, dict) and result.get("status") == "success":
                chainer_rule_atoms = unique_preserve_order(
                    [*chainer_rule_atoms, *result.get("rules", [])]
                )
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'result': job.result,
            'min_support': job.min_support,
            'start_time': job.start_time,
            'end_time': job.end_time
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        job.status = 'error'
        job.error = str(exc)
        job.end_time = time.time()
        return {
            'jobId': job_id,
            'status': job.status,
            'error': job.error,
            'min_support': job.min_support,
            'start_time': job.start_time,
            'end_time': job.end_time
        }
    finally:
        for path in (result_path, stdout_path, stderr_path):
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

def start_mining_job(conjunction_count: int = DEFAULT_CONJUNCTION_COUNT, min_support: int = DEFAULT_MIN_SUPPORT):
    """
    Wrapper that creates a MiningJob and runs the mining task synchronously
    so that function-calls from the LLM go through the same code path as the
    HTTP `/api/mine` endpoint.
    Returns a dict similar to the `/api/mine` response (jobId, status, result).
    """
    try:
        if not isinstance(conjunction_count, int):
            conjunction_count = int(conjunction_count)
    except Exception:
        return { 'error': 'conjunction_count must be an integer' }
    try:
        if not isinstance(min_support, int):
            min_support = int(min_support)
    except Exception:
        return { 'error': 'min_support must be an integer' }
    if conjunction_count < 1:
        return { 'error': 'conjunction_count must be a positive integer' }
    if conjunction_count > DEFAULT_MAX_CONJUNCTION_COUNT:
        return {
            'error': (
                f'conjunction_count must be <= {DEFAULT_MAX_CONJUNCTION_COUNT}. '
                'Use PETTA_MAX_CONJUNCTION_COUNT to raise this limit after validating the dataset.'
            )
        }
    if min_support < 1:
        return { 'error': 'min_support must be a positive integer' }

    job_id = str(uuid.uuid4())
    job = MiningJob(
        job_id=job_id,
        status='running',
        conjunction_count=conjunction_count,
        min_support=min_support,
    )
    mining_jobs[job_id] = job

    result = run_mining_task(job_id, conjunction_count, min_support)

    return {
        'jobId': job_id,
        'status': mining_jobs[job_id].status,
        'conjunction_count': conjunction_count,
        'min_support': min_support,
        'result': mining_jobs[job_id].result
    }

def formatter(mined_patterns):
    return get_chainer_service().formatter(mined_patterns)


def run_chainer_query_worker(what_to_check: str, depth: int = DEFAULT_CHAIN_DEPTH) -> list[str]:
    reload_petta_dataset_if_ready(force=False)
    payload = {
        "project_root": PROJECT_ROOT,
        "setup_metta": CHAINER_METTA_SETUP,
        "facts": list(chainer_dataset_facts),
        "rules": ordered_chainer_rules(),
        "query": what_to_check,
        "depth": int(depth),
    }
    process = subprocess.run(
        [sys.executable, "-m", "experiments.chainer_query_worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=PETTA_CHAIN_TIMEOUT_SECONDS,
        env={
            **os.environ,
            "PYTHONPATH": PROJECT_ROOT
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
    )
    stdout = (process.stdout or "").strip()
    stderr = (process.stderr or "").strip()
    if not stdout:
        raise RuntimeError(stderr or "Backward chainer worker produced no output.")

    try:
        worker_result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Backward chainer worker returned invalid JSON: {stdout[:500]}") from exc

    if process.returncode != 0 or worker_result.get("status") != "success":
        message = worker_result.get("message") if isinstance(worker_result, dict) else None
        raise RuntimeError(message or stderr or f"Backward chainer worker exited with code {process.returncode}.")

    proofs = worker_result.get("proofs", [])
    return proofs if isinstance(proofs, list) else [str(proofs)]


def safe_chainer_failure(what_to_check: str, depth: int, exc: Exception) -> dict[str, Any]:
    logger.exception("Backward chainer query failed")
    return {
        "query": what_to_check,
        "status": "error",
        "justification": (
            "The backward chainer could not safely complete this proof "
            f"within depth {depth}. The query may be too broad, recursive, or expensive for the current "
            "rule set. Try a more concrete article/engagement query or lower the depth."
        ),
        "technical_error": str(exc)[:1000],
        "depth_used": depth,
    }


def backWardChainer(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    return run_chainer_query_worker(whatToCheck.strip(), depth=depth)

def getChainerResult(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    """ Get the result of backward chaining for a specific query. 
    Args:
        whatToCheck (str): The query to check, e.g., '(engagement 0 "High")'
        depth (int): The depth limit for backward chaining.
    Returns:
        The justification of the backward chaining operation.
    """
    facts_res = getAllFactsAndRules()
    try:
        chainAnswer = backWardChainer(whatToCheck, depth)
    except subprocess.TimeoutExpired as exc:
        return safe_chainer_failure(whatToCheck, depth, exc)
    except Exception as exc:
        return safe_chainer_failure(whatToCheck, depth, exc)

    proof_text = format_proofs_for_prompt(chainAnswer)
    all_facts = facts_res.get("facts", []) if isinstance(facts_res, dict) else []
    prompt_facts = select_facts_for_prompt(all_facts, whatToCheck)
    fact_text = format_proofs_for_prompt(prompt_facts) if prompt_facts else str(facts_res)
    if len(all_facts) > len(prompt_facts):
        fact_text += f"\n\nShowing {len(prompt_facts)} of {len(all_facts)} facts most relevant to the query."
    if not chainAnswer or len(chainAnswer) == 0:
        return {
            "query": whatToCheck,
            "status": "no_proof",
            "justification": f"No logical proof could be found for the query '{whatToCheck}' within depth {depth}. This means the query cannot be deduced from the available rules and facts in the knowledge base."
        }
    prompt = build_chainer_analysis_prompt(whatToCheck, proof_text, fact_text)
    try:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
        response_data = call_asi_api(messages)
        justification = "Unable to generate justification analysis."
        if 'choices' in response_data and response_data['choices']:
             justification = response_data['choices'][0]['message'].get('content', '')
        
        return {
            "query": whatToCheck,
            "status": "success",
            "raw_proofs": chainAnswer,
            "proof_count": len(chainAnswer),
            "justification": justification,
            "depth_used": depth
        }
        
    except Exception as e:
        # Fallback to basic analysis if LLM fails
        proof_count = len(chainAnswer)
        basic_justification = f"""
        **Query Analysis:** {whatToCheck}

        **Result:** Found {proof_count} logical proof(s) supporting this query.

        **Raw Evidence:** {proof_text}

        **Basic Interpretation:** The backward chaining system discovered {proof_count} different logical path(s) that support the query "{whatToCheck}". Each proof represents a combination of rules and facts from the knowledge base that logically leads to this conclusion.

        **Note:** Advanced analysis unavailable due to processing error: {str(e)}
        """
        
        return {
            "query": whatToCheck,
            "status": "partial_success",
            "raw_proofs": chainAnswer,
            "proof_count": proof_count,
            "justification": basic_justification,
            "depth_used": depth,
            "error": str(e)
        }

def summarize_patterns(patterns: list) -> str:
    return summarize_patterns_impl(
        patterns,
        call_asi_api=call_asi_api,
        system_instruction=SYSTEM_INSTRUCTION,
        logger=logger,
    )

# Function name to actual function mapping (for execution)
available_functions = {
    "mine_pattern": start_mining_job,
    # Aliases so the model can call either the wrapper or the original-style name
    "start_mining_job": start_mining_job,
    "startMiningJob": start_mining_job,
    "minePattern": start_mining_job,
    "get_mining_results": get_mining_results,
    "analyze_specific_pattern": analyze_specific_pattern,
    "get_pattern_statistics": get_pattern_statistics,
    "visualize_pattern_request": visualize_pattern_request,
    "getChainerResult": getChainerResult
}
conversations = {}
@dataclass
class MiningJob:
    job_id: str
    status: str  # 'running', 'completed', 'error'
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: float = 0
    end_time: Optional[float] = None
    conjunction_count: int = 0
    min_support: int = DEFAULT_MIN_SUPPORT
# In-memory storage for mining jobs
mining_jobs: Dict[str, MiningJob] = {}


register_core_routes(
    app,
    logger=logger,
    run_ingestion=run_ingestion,
    reload_petta_dataset_if_ready=reload_petta_dataset_if_ready,
    dataset_file_path=dataset_file_path,
    get_chainer_service=get_chainer_service,
    petta_startup_error_type=PeTTaStartupError,
    default_conjunction_count=DEFAULT_CONJUNCTION_COUNT,
    default_min_support=DEFAULT_MIN_SUPPORT,
    max_conjunction_count=DEFAULT_MAX_CONJUNCTION_COUNT,
    mining_jobs=mining_jobs,
    mining_job_type=MiningJob,
    run_mining_task=run_mining_task,
    simulate_engagement=simulate_engagement,
    make_json_safe=make_json_safe,
)

# ============= CHAT API ENDPOINTS =============

def parse_pattern(pattern: str) -> dict:
    return parse_pattern_impl(pattern)

def analyze_pattern(pattern: str, support: str) -> str:
    return analyze_pattern_impl(pattern, support)


register_chat_routes(
    app,
    logger=logger,
    conversations=conversations,
    call_asi_api=call_asi_api,
    system_instruction=SYSTEM_INSTRUCTION,
    tools_schema=tools_schema,
    handle_mining_for_message=handle_mining_for_message,
    is_backward_chain_intent=is_backward_chain_intent,
    handle_backward_chain_for_message=handle_backward_chain_for_message,
    available_functions=available_functions,
    summarize_patterns=summarize_patterns,
    analyze_pattern=analyze_pattern,
    make_json_safe=make_json_safe,
    omegaclaw_chat_handler=send_chat_to_omegaclaw if is_omegaclaw_chat_enabled() else None,
)

def create_app():
    """Application factory used by production WSGI servers.

    Startup is intentionally fail-fast: if Janus, SWI-Prolog, PeTTa, or the
    required MeTTa libraries cannot be loaded, this function raises and the
    server process does not start.
    """
    bootstrap_runtime()
    return app

if __name__ == '__main__':
    logger.info("Starting Unified API Server (Mining + Chat)")
    create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        threaded=True
    )
