"""Production PeTTa integration boundary.

This module owns all direct interaction with Janus, SWI-Prolog, and PeTTa.
The Flask layer should call this service instead of issuing raw MeTTa helper
commands directly.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from petta import PeTTa


class PeTTaStartupError(RuntimeError):
    """Raised when the required PeTTa runtime cannot be initialized."""


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        item = (item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def chainer_result_lines(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, str):
        value = result.strip()
        return unique_preserve_order(value.splitlines() or [value]) if value else []
    if isinstance(result, (list, tuple, set)):
        flattened: list[str] = []
        for item in result:
            flattened.extend(chainer_result_lines(item))
        return unique_preserve_order(flattened)
    return unique_preserve_order([str(result)])


def parse_petta_output(output: Any) -> list[str]:
    """Clean PeTTa textual output into meaningful result lines."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    if isinstance(output, (list, tuple)):
        output = "\n".join(str(item) for item in output)
    clean_output = ansi_escape.sub("", str(output or ""))

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


def format_proofs_for_prompt(proofs: list[str]) -> str:
    if not proofs:
        return "No proofs found."
    return "\n".join(f"{idx}. {proof}" for idx, proof in enumerate(proofs, start=1))


@dataclass(frozen=True)
class PeTTaHealth:
    status: str
    swi_prolog_version: str
    petta_path: str
    setup_loaded: bool
    chainer_loaded: bool
    kb: str
    added_atoms: int
    dataset_path: Optional[str]
    dataset_mtime: Optional[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "swi_prolog_version": self.swi_prolog_version,
            "petta_path": self.petta_path,
            "setup_loaded": self.setup_loaded,
            "chainer_loaded": self.chainer_loaded,
            "kb": self.kb,
            "added_atoms": self.added_atoms,
            "dataset_path": self.dataset_path,
            "dataset_mtime": self.dataset_mtime,
        }


class PeTTaService:
    """Fail-fast, mandatory PeTTa runtime service."""

    MIN_SWI_VERSION = (9, 3, 0)

    def __init__(self, project_root: str, setup_metta: str, *, verbose: bool = False, min_swi_version: tuple[int, int, int] = MIN_SWI_VERSION,) -> None:
        self.project_root = os.path.abspath(project_root)
        self.setup_metta = setup_metta
        self.verbose = verbose
        self.min_swi_version = min_swi_version
        self.kb = "kb" + uuid.uuid4().hex
        self._engine = None
        self._lock = threading.RLock()
        self._atoms_lock = threading.Lock()
        self._added_atoms: set[str] = set()
        self._setup_loaded = False
        self._chainer_loaded = False
        self._swi_version = "unknown"
        self._dataset_file_path: Optional[str] = None
        self._dataset_mtime: Optional[float] = None
        self.atom_re = re.compile(r'\([A-Za-z_][\w\-]*\s+\$[_\w\d]+\s+"[^"]*"\)')
        self.stv_re = re.compile(r"\(STV\s+([0-9eE\.\-]+)\s+([0-9eE\.\-]+)\)")

    @classmethod
    def create_required(cls, project_root: str, setup_metta: str, *, verbose: bool = False,) -> "PeTTaService":
        service = cls(project_root=project_root, setup_metta=setup_metta, verbose=verbose)
        service.start()
        return service

    def start(self) -> None:
        """Initialize Janus/SWI, create PeTTa, and load all required libraries."""
        self._engine = PeTTa(verbose=self.verbose)
        self._load_setup()
        self._load_chainer()
        self._smoke_test()

    def health(self) -> dict[str, Any]:
        return PeTTaHealth(
            status="ok" if self._engine is not None and self._setup_loaded and self._chainer_loaded else "error",
            swi_prolog_version=self._swi_version,
            petta_path=os.path.join(self.project_root, "PeTTa"),
            setup_loaded=self._setup_loaded,
            chainer_loaded=self._chainer_loaded,
            kb=self.kb,
            added_atoms=len(self._added_atoms),
            dataset_path=self._dataset_file_path,
            dataset_mtime=self._dataset_mtime,
        ).as_dict()

    def process_metta_string(self, metta_code: str) -> Any:
        with self._lock:
            return self._engine.process_metta_string(metta_code)

    def load_metta_file(self, metta_path: str) -> Any:
        with self._lock:
            return self._engine.load_metta_file(metta_path)

    def query_lines(self, metta_code: str) -> list[str]:
        return [line for line in parse_petta_output(self.process_metta_string(metta_code)) if line and line.lower() != "true"]

    def reload_dataset(self, dataset_file_path: Optional[str] = None) -> dict[str, Any]:
        """Reload facts used by mining and chainer from the current dataset file.
        The miner reads `&purifiedDbSpace`; the chainer/fact listing reads
        `&res1`. Both spaces must be cleared and repopulated together or mined
        rules can drift away from the dataset currently shown by the frontend.
        """
        dataset_file_path = os.path.abspath(dataset_file_path)
        dataset_module_path = dataset_file_path.removesuffix(".metta")

        if not os.path.exists(dataset_file_path):
            raise FileNotFoundError(f"Dataset file does not exist: {dataset_file_path}")

        temp_space = "&tempoReload" + uuid.uuid4().hex
        reload_code = f"""
            !(import! {temp_space} {dataset_module_path})
            !(remove-all-atoms &res1)
            !(remove-all-atoms &purifiedDbSpace)
            !(let $atom (match {temp_space} ($fact $stv) (: (fact:- $fact) $fact $stv)) (add-atom &res1 $atom))
            !(remove-atom &res1 (: (fact:- (engagement $sub $obj)) (engagement $sub $obj) $stv))
            !(let $atom (match {temp_space} ($fact $stv) $fact) (add-atom &purifiedDbSpace $atom))
        """

        with self._lock:
            self._load_temp_metta_file(reload_code)
            self.kb = "kb" + uuid.uuid4().hex
            with self._atoms_lock:
                self._added_atoms.clear()
            self.set_dataset_metadata(
                dataset_file_path=dataset_file_path,
                dataset_mtime=os.path.getmtime(dataset_file_path),
            )

        return {
            "status": "success",
            "dataset_path": self._dataset_file_path,
            "dataset_mtime": self._dataset_mtime,
            "kb": self.kb,
        }


    def add_atom(self, atom: str) -> Optional[Any]:
        return self._compile_atom(atom, "compileadd")

    def add_forward_only_rule(self, atom: str) -> Optional[Any]:
        return self._compile_atom(atom, "compileadd-forward-only")

    def reset_kb(self) -> str:
        self.kb = "kb" + uuid.uuid4().hex
        with self._atoms_lock:
            self._added_atoms.clear()
        return self.kb

    def set_dataset_metadata(self, *, dataset_file_path: Optional[str] = None, dataset_mtime: Optional[float] = None,) -> None:
        self._dataset_file_path = dataset_file_path
        self._dataset_mtime = dataset_mtime

    def _compile_atom(self, atom: str, compiler: str) -> Optional[Any]:
        atom = (atom or "").strip()
        if not atom:
            return None

        cache_key = f"{compiler}:{atom}"
        with self._atoms_lock:
            if cache_key in self._added_atoms:
                return None
            self._added_atoms.add(cache_key)

        try:
            return self.process_metta_string(f"!({compiler} {self.kb} {atom})")
        except Exception:
            with self._atoms_lock:
                self._added_atoms.discard(cache_key)
            raise

    def query(self, atom: str, depth: int = 10) -> list[str]:
        atom = (atom or "").strip()
        if not atom.startswith("(:"):
            atom = f"(: $prf {atom} $tv)"
        result = self.process_metta_string(f"!(query (fromNumber {depth}) {self.kb} {atom})")
        return chainer_result_lines(result)

    def formatter(self, mined_patterns: Any) -> dict[str, Any]:
        """Insert mined patterns into the chainer KB as PeTTa rules."""
        try:
            if isinstance(mined_patterns, str):
                try:
                    payload = json.loads(mined_patterns)
                except json.JSONDecodeError:
                    payload = ast.literal_eval(mined_patterns)
            else:
                payload = mined_patterns
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "message": "Mined patterns payload must be a dictionary.",
                    "insertedRuleCount": 0,
                }

            patterns = payload.get("patterns", [])
            inserted_rules = []
            for idx, pattern in enumerate(patterns, start=1):
                pattern_text = str(pattern.get("pattern", "")) if isinstance(pattern, dict) else str(pattern)
                rule_atom = self.pattern_to_rule(pattern_text, idx)
                if rule_atom:
                    self.add_forward_only_rule(rule_atom)
                    inserted_rules.append(rule_atom)

            return {
                "status": "success",
                "insertedRuleCount": len(inserted_rules),
                "rules": inserted_rules,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "insertedRuleCount": 0,
            }

    def pattern_to_rule(self, pattern_text: str, idx: int) -> Optional[str]:
        atoms = [self._normalize_var(atom) for atom in self.atom_re.findall(pattern_text or "")]
        if not atoms:
            return None

        stv_match = self.stv_re.search(pattern_text or "")
        strength, confidence = (stv_match.group(1), stv_match.group(2)) if stv_match else ("1.0", "1.0")
        strength_value = min(max(float(strength), 0.0), 1.0)
        confidence_value = min(max(float(confidence), 0.0), 1.0)

        consequent = next((atom for atom in atoms if atom.startswith("(engagement ")), atoms[-1])
        antecedents = [atom for atom in atoms if atom != consequent]
        lhs = antecedents[0] if len(antecedents) == 1 else f"(And {' '.join(antecedents)})"

        return f'(: rule_{idx} (-> {lhs} {consequent}) (STV {strength_value} {confidence_value}))'

    @staticmethod
    def _normalize_var(atom: str) -> str:
        return re.sub(r"\$_\d+", "$x", atom)


    def _load_setup(self) -> None:
        try:
            self._load_temp_metta_file(self.setup_metta)
            self._setup_loaded = True
        except Exception as exc:
            raise PeTTaStartupError("Failed to load required mining MeTTa libraries into PeTTa.") from exc

    def _load_chainer(self) -> None:
        chainer_path = os.path.join(self.project_root, "experiments", "chainer", "petta_chainer.metta")
        try:
            self.load_metta_file(chainer_path)
            self._chainer_loaded = True
        except Exception as exc:
            raise PeTTaStartupError(f"Failed to load PeTTa chainer library: {chainer_path}") from exc

    def _smoke_test(self) -> None:
        try:
            self.process_metta_string("!(+ 1 1)")
        except Exception as exc:
            raise PeTTaStartupError("PeTTa started, but failed a minimal MeTTa smoke test.") from exc

    def _run_metta_file(self, metta_code: str) -> str:
        results = self._load_temp_metta_file(metta_code)
        if isinstance(results, (list, tuple)):
            return "\n".join(str(r) for r in results)
        return str(results)

    def _load_temp_metta_file(self, metta_code: str) -> Any:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".metta", encoding="utf-8", delete=False) as handle:
                handle.write(metta_code)
                temp_file_path = handle.name
            return self.load_metta_file(temp_file_path)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
