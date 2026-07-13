from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

from experiments.api.chat.prompts import SYSTEM_INSTRUCTION, build_chainer_analysis_prompt
from experiments.api.config import (
    CHAINER_METTA_SETUP,
    DEFAULT_CHAIN_DEPTH,
    PETTA_CHAIN_TIMEOUT_SECONDS,
    PROJECT_ROOT,
)
from experiments.api.support import select_facts_for_prompt
from experiments.api.runtime import (
    dataset_compile_errors,
    dataset_facts,
    ordered_chainer_rules,
    reload_petta_dataset_if_ready,
)
from experiments.services.petta_service import format_proofs_for_prompt

logger = logging.getLogger(__name__)


def select_facts_for_query(facts: list[str], query: str) -> list[str]:
    """Limit concrete article queries to facts about the named entities."""
    query_tokens = (query or "").replace("(", " ").replace(")", " ").split()
    entity_ids = {
        token
        for token in query_tokens
        if token.startswith(("A_", "H_")) and not token.startswith("$_")
    }
    if not entity_ids:
        return list(facts)

    return [
        fact
        for fact in facts
        if any(f" {entity_id} " in fact for entity_id in entity_ids)
    ]


def getAllFactsAndRules():
    """Return current facts and rules from the MeTTa knowledge base."""
    try:
        dataset_reload = reload_petta_dataset_if_ready(force=False)
        aligned_facts = dataset_facts()
        compile_errors = dataset_compile_errors()

        if aligned_facts and len(compile_errors) == len(aligned_facts):
            return {
                "status": "error",
                "error": "Failed to compile every fact into PeTTaChainer.",
                "dataset": dataset_reload,
                "fact_count": len(aligned_facts),
                "compile_error_count": len(compile_errors),
            }

        return {
            "status": "success",
            "dataset": dataset_reload,
            "facts": aligned_facts,
            "fact_count": len(aligned_facts),
            "compiled_new_count": dataset_reload.get("compiled_fact_count", 0),
            "compile_error_count": len(compile_errors),
        }
    except Exception:
        logger.exception("Failed to read facts and rules")
        return {"status": "error", "error": "The knowledge base is unavailable."}


def run_chainer_query_worker(what_to_check: str, depth: int = DEFAULT_CHAIN_DEPTH) -> list[str]:
    reload_petta_dataset_if_ready(force=False)
    all_facts = dataset_facts()
    query_facts = select_facts_for_query(all_facts, what_to_check)
    rules = ordered_chainer_rules()
    logger.info(
        "Starting backward chainer worker: depth=%s facts=%s/%s rules=%s",
        depth,
        len(query_facts),
        len(all_facts),
        len(rules),
    )
    payload = {
        "project_root": PROJECT_ROOT,
        "setup_metta": CHAINER_METTA_SETUP,
        "facts": query_facts,
        "rules": rules,
        "query": what_to_check,
        "depth": int(depth),
    }
    started_at = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-m", "experiments.api.workers.chainer_query"],
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
    normalized_proofs = proofs if isinstance(proofs, list) else [str(proofs)]
    logger.info(
        "Backward chainer worker completed: depth=%s proofs=%s seconds=%.3f",
        depth,
        len(normalized_proofs),
        time.monotonic() - started_at,
    )
    return normalized_proofs


def safe_chainer_failure(what_to_check: str, depth: int, *, timed_out: bool = False) -> dict[str, Any]:
    if timed_out:
        logger.warning("Backward chainer query timed out: depth=%s query=%s", depth, what_to_check)
    else:
        logger.exception("Backward chainer query failed")
    return {
        "query": what_to_check,
        "status": "error",
        "justification": (
            "The backward chainer could not safely complete this proof "
            f"within depth {depth}. The query may be too broad, recursive, or expensive for the current "
            "rule set. Verify the query specificity and the number of matching rules."
        ),
        "depth_used": depth,
    }


def backWardChainer(whatToCheck, depth=DEFAULT_CHAIN_DEPTH):
    return run_chainer_query_worker(whatToCheck.strip(), depth=depth)


def getChainerResult(whatToCheck, *, call_asi_api, depth=DEFAULT_CHAIN_DEPTH):
    facts_res = getAllFactsAndRules()
    try:
        chainAnswer = backWardChainer(whatToCheck, depth)
    except subprocess.TimeoutExpired:
        return safe_chainer_failure(whatToCheck, depth, timed_out=True)
    except Exception:
        return safe_chainer_failure(whatToCheck, depth)

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

    except Exception:
        logger.exception("Failed to generate backward-chainer explanation")
        proof_count = len(chainAnswer)
        basic_justification = f"""
        **Query Analysis:** {whatToCheck}

        **Result:** Found {proof_count} logical proof(s) supporting this query.

        **Raw Evidence:** {proof_text}

        **Basic Interpretation:** The backward chaining system discovered {proof_count} different logical path(s) that support the query "{whatToCheck}". Each proof represents a combination of rules and facts from the knowledge base that logically leads to this conclusion.

        **Note:** The proof was found, but the advanced language explanation is temporarily unavailable.
        """

        return {
            "query": whatToCheck,
            "status": "partial_success",
            "raw_proofs": chainAnswer,
            "proof_count": proof_count,
            "justification": basic_justification,
            "depth_used": depth,
        }
