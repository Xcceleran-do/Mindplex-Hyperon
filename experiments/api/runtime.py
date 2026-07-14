from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from experiments.api.config import (
    PETTA_CHAIN_TIMEOUT_SECONDS,
    PETTACHAINER_API_KEY,
    PETTACHAINER_BASE_URL,
    PETTACHAINER_KB_PREFIX,
    dataset_file_path,
)
from experiments.api.support import load_dataset_facts_for_chainer, unique_preserve_order
from experiments.services.pettachainer_client import PeTTaChainerClient

logger = logging.getLogger(__name__)

chainer_client = PeTTaChainerClient(
    PETTACHAINER_BASE_URL,
    PETTACHAINER_API_KEY,
    PETTA_CHAIN_TIMEOUT_SECONDS,
)
chainer_dataset_path: Optional[str] = None
chainer_dataset_mtime: Optional[float] = None
chainer_dataset_digest: Optional[str] = None
chainer_dataset_facts: list[str] = []
chainer_dataset_compile_errors: list[dict[str, str]] = []
chainer_dataset_compiled_count: int = 0
chainer_rule_atoms: list[str] = []
chainer_kb_id: Optional[str] = None
chainer_kb_signature: Optional[str] = None
runtime_lock = threading.Lock()


def ordered_chainer_rules() -> list[str]:
    return sorted(unique_preserve_order(chainer_rule_atoms))


def dataset_facts() -> list[str]:
    return list(chainer_dataset_facts)


def dataset_compile_errors() -> list[dict[str, str]]:
    return list(chainer_dataset_compile_errors)


def chainer_rules_path(file_path: str | None = None) -> str:
    return f"{file_path or dataset_file_path()}.rules.json"


def _load_persisted_rules(file_path: str, dataset_digest: str) -> list[str]:
    try:
        with open(chainer_rules_path(file_path), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("dataset_digest") != dataset_digest:
            return []
        rules = payload.get("rules", [])
        if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
            return []
        return unique_preserve_order(rules)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return []


def _persist_chainer_rules(file_path: str, dataset_digest: str, rules: list[str]) -> None:
    store_path = chainer_rules_path(file_path)
    output_dir = os.path.dirname(store_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = None
    payload = {
        "version": 1,
        "dataset_digest": dataset_digest,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rules": unique_preserve_order(rules),
    }
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".chainer-rules-",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary_path, store_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def record_chainer_rules(rules: list[str]) -> None:
    global chainer_rule_atoms
    normalized_rules = unique_preserve_order(rules)
    if not normalized_rules:
        return

    # Mining runs in an isolated worker. Align the parent runtime with the
    # dataset before recording its returned rules, otherwise the first query's
    # initial dataset load would clear those rules.
    reload_petta_dataset_if_ready(force=False)
    with runtime_lock:
        chainer_rule_atoms = unique_preserve_order([*chainer_rule_atoms, *normalized_rules])
        if chainer_dataset_path and chainer_dataset_digest:
            try:
                _persist_chainer_rules(
                    chainer_dataset_path,
                    chainer_dataset_digest,
                    chainer_rule_atoms,
                )
            except OSError:
                logger.exception("Failed to persist mined chainer rules")


def get_chainer_service() -> PeTTaChainerClient:
    return chainer_client


def invalidate_chainer_dataset() -> None:
    global chainer_dataset_mtime, chainer_dataset_digest, chainer_kb_id, chainer_kb_signature
    global chainer_rule_atoms
    with runtime_lock:
        chainer_dataset_mtime = None
        chainer_dataset_digest = None
        chainer_kb_id = None
        chainer_kb_signature = None
        chainer_rule_atoms = []
        try:
            os.unlink(chainer_rules_path())
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to remove stale persisted chainer rules")


def reload_petta_dataset_if_ready(force: bool = False) -> dict[str, Any]:
    """Read chainer facts locally; remote synchronization remains query-driven."""
    global chainer_dataset_path, chainer_dataset_mtime, chainer_dataset_digest
    global chainer_dataset_facts, chainer_dataset_compiled_count, chainer_kb_id
    global chainer_kb_signature
    global chainer_rule_atoms
    file_path = dataset_file_path()
    current_mtime = os.path.getmtime(file_path)
    changed = force or chainer_dataset_path != file_path or chainer_dataset_mtime != current_mtime
    if changed:
        with runtime_lock:
            file_path = dataset_file_path()
            current_mtime = os.path.getmtime(file_path)
            changed = force or chainer_dataset_path != file_path or chainer_dataset_mtime != current_mtime
            if changed:
                with open(file_path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                chainer_dataset_facts = load_dataset_facts_for_chainer(file_path)
                chainer_dataset_path = file_path
                chainer_dataset_mtime = current_mtime
                chainer_dataset_digest = digest
                chainer_dataset_compiled_count = 0
                chainer_kb_id = None
                chainer_kb_signature = None
                chainer_rule_atoms = _load_persisted_rules(file_path, digest)
                status = "loaded"
            else:
                status = "unchanged"
    else:
        status = "unchanged"
    return {
        "status": status,
        "fact_count": len(chainer_dataset_facts),
        "compiled_fact_count": chainer_dataset_compiled_count,
        "compile_error_count": 0,
        "chainer": {"status": "deferred", "added_atoms": chainer_dataset_compiled_count},
        "mining": {"mode": "subprocess"},
    }


def ensure_remote_chainer(facts: list[str] | None = None) -> tuple[PeTTaChainerClient, str]:
    global chainer_kb_id, chainer_kb_signature, chainer_dataset_compiled_count
    reload_petta_dataset_if_ready(force=False)
    with runtime_lock:
        rules = ordered_chainer_rules()
        selected_facts = chainer_dataset_facts if facts is None else facts
        rule_digest = hashlib.sha256("\n".join(rules).encode("utf-8")).hexdigest()
        fact_digest = hashlib.sha256("\n".join(selected_facts).encode("utf-8")).hexdigest()
        signature = f"{chainer_dataset_digest}:{rule_digest}:{fact_digest}"
        if chainer_kb_id is None or chainer_kb_signature != signature:
            name = (
                f"{PETTACHAINER_KB_PREFIX}-{chainer_dataset_digest[:12]}-"
                f"{rule_digest[:12]}-{fact_digest[:16]}"
            )
            chainer_kb_id = chainer_client.ensure_knowledge_base(name)
            chainer_kb_signature = signature
        statements = [*selected_facts, *rules]
        chainer_client.add_statements(chainer_kb_id, statements)
        chainer_dataset_compiled_count = len(selected_facts)
        return chainer_client, chainer_kb_id
