from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from experiments.api.config import CHAINER_METTA_SETUP, PROJECT_ROOT, dataset_file_path
from experiments.api.support import compile_facts_into_chainer, load_dataset_facts_for_chainer
from experiments.services.petta_service import PeTTaService, PeTTaStartupError, unique_preserve_order

logger = logging.getLogger(__name__)

chainer_service: Optional[PeTTaService] = None
chainer_dataset_path: Optional[str] = None
chainer_dataset_mtime: Optional[float] = None
chainer_dataset_facts: list[str] = []
chainer_dataset_compile_errors: list[dict[str, str]] = []
chainer_dataset_compiled_count: int = 0
chainer_rule_atoms: list[str] = []
runtime_lock = threading.Lock()


def ordered_chainer_rules() -> list[str]:
    """PeTTaChainer rule aggregation is order-sensitive; keep ordering stable."""
    return sorted(unique_preserve_order(chainer_rule_atoms))


def dataset_facts() -> list[str]:
    return list(chainer_dataset_facts)


def dataset_compile_errors() -> list[dict[str, str]]:
    return list(chainer_dataset_compile_errors)


def record_chainer_rules(rules: list[str]) -> None:
    global chainer_rule_atoms
    chainer_rule_atoms = unique_preserve_order([*chainer_rule_atoms, *rules])


def _dataset_runtime_payload(status: str) -> dict[str, Any]:
    service = get_chainer_service()
    dataset_errors = dataset_compile_errors()
    facts = dataset_facts()
    service_health = service.health() if service else {}
    return {
        "status": status,
        "fact_count": len(facts),
        "compiled_fact_count": chainer_dataset_compiled_count,
        "compile_error_count": len(dataset_errors),
        "chainer": {
            "status": service_health.get("status", "unknown"),
            "added_atoms": service_health.get("added_atoms", 0),
        },
        "mining": {"mode": "subprocess"},
    }


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
                service = PeTTaService.create_required(
                    project_root=PROJECT_ROOT,
                    setup_metta=CHAINER_METTA_SETUP,
                    verbose=False,
                )
                facts = load_dataset_facts_for_chainer(file_path)
                compiled_count, compile_errors = compile_facts_into_chainer(service, facts)
                service.set_dataset_metadata(
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

    return _dataset_runtime_payload(status)
