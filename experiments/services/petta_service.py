"""Local PeTTa integration boundary for mining and the legacy simulator.

This module owns all direct interaction with Janus, SWI-Prolog, and PeTTa.
The Flask layer should call this service instead of issuing raw MeTTa helper
commands directly.
"""

from __future__ import annotations

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

    def __init__(self, project_root: str, setup_metta: str, *, verbose: bool = False, load_chainer: bool = True, min_swi_version: tuple[int, int, int] = MIN_SWI_VERSION,) -> None:
        self.project_root = os.path.abspath(project_root)
        self.setup_metta = setup_metta
        self.verbose = verbose
        self.min_swi_version = min_swi_version
        self._requires_chainer = load_chainer
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

    @classmethod
    def create_required(cls, project_root: str, setup_metta: str, *, verbose: bool = False, load_chainer: bool = True,) -> "PeTTaService":
        service = cls(
            project_root=project_root,
            setup_metta=setup_metta,
            verbose=verbose,
            load_chainer=load_chainer,
        )
        service.start()
        return service

    def start(self) -> None:
        """Initialize Janus/SWI, create PeTTa, and load all required libraries."""
        self._engine = PeTTa(verbose=self.verbose)
        self._load_setup()
        if self._requires_chainer:
            self._load_chainer()
        self._smoke_test()

    def health(self) -> dict[str, Any]:
        return PeTTaHealth(
            status=(
                "ok"
                if self._engine is not None
                and self._setup_loaded
                and (self._chainer_loaded or not self._requires_chainer)
                else "error"
            ),
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
