import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from experiments import mining_api
    MINING_API_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific optional deps
    mining_api = None
    MINING_API_IMPORT_ERROR = exc


class FakeChainerService:
    def __init__(self, kb: str) -> None:
        self.kb = kb
        self.added_atoms: list[str] = []
        self.dataset_module_path = None
        self.dataset_file_path = None
        self.dataset_mtime = None

    def add_atom(self, atom: str):
        self.added_atoms.append(atom)
        return ["true"]

    def set_dataset_metadata(self, *, dataset_module_path=None, dataset_file_path=None, dataset_mtime=None):
        self.dataset_module_path = dataset_module_path
        self.dataset_file_path = dataset_file_path
        self.dataset_mtime = dataset_mtime

    def health(self) -> dict:
        return {
            "status": "ok",
            "kb": self.kb,
            "added_atoms": len(self.added_atoms),
            "dataset_path": self.dataset_file_path,
            "dataset_mtime": self.dataset_mtime,
        }


@unittest.skipUnless(mining_api is not None, f"mining_api dependencies unavailable: {MINING_API_IMPORT_ERROR}")
class TestMiningApiDatasetRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_state = {
            "chainer_service": mining_api.chainer_service,
            "chainer_dataset_path": mining_api.chainer_dataset_path,
            "chainer_dataset_mtime": mining_api.chainer_dataset_mtime,
            "chainer_dataset_facts": list(mining_api.chainer_dataset_facts),
            "chainer_dataset_compile_errors": list(mining_api.chainer_dataset_compile_errors),
            "chainer_dataset_compiled_count": mining_api.chainer_dataset_compiled_count,
        }
        mining_api.chainer_service = None
        mining_api.chainer_dataset_path = None
        mining_api.chainer_dataset_mtime = None
        mining_api.chainer_dataset_facts = []
        mining_api.chainer_dataset_compile_errors = []
        mining_api.chainer_dataset_compiled_count = 0

    def tearDown(self) -> None:
        mining_api.chainer_service = self._saved_state["chainer_service"]
        mining_api.chainer_dataset_path = self._saved_state["chainer_dataset_path"]
        mining_api.chainer_dataset_mtime = self._saved_state["chainer_dataset_mtime"]
        mining_api.chainer_dataset_facts = self._saved_state["chainer_dataset_facts"]
        mining_api.chainer_dataset_compile_errors = self._saved_state["chainer_dataset_compile_errors"]
        mining_api.chainer_dataset_compiled_count = self._saved_state["chainer_dataset_compiled_count"]

    def test_load_dataset_facts_for_chainer_normalizes_lines(self) -> None:
        dataset_text = """
((topic A_1 "AI") (STV 1.0 1.0))
((tone A_1 "Analytical") (STV 0.9 0.8))
not a fact line
((topic A_1 "AI") (STV 1.0 1.0))
"""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(dataset_text)
            path = handle.name

        try:
            facts = mining_api.load_dataset_facts_for_chainer(path)
        finally:
            os.remove(path)

        self.assertEqual(
            facts,
            [
                '(: (fact:- (topic A_1 "AI")) (topic A_1 "AI") (STV 1.0 1.0))',
                '(: (fact:- (tone A_1 "Analytical")) (tone A_1 "Analytical") (STV 0.9 0.8))',
            ],
        )

    def test_reload_petta_dataset_if_ready_compiles_once_per_dataset_version(self) -> None:
        dataset_path = "/tmp/test-data.metta"
        facts = [
            '(: (fact:- (topic A_1 "AI")) (topic A_1 "AI") (STV 1.0 1.0))',
            '(: (fact:- (tone A_1 "Analytical")) (tone A_1 "Analytical") (STV 0.9 0.8))',
        ]
        service_a = FakeChainerService("kb_a")
        service_b = FakeChainerService("kb_b")

        with patch.object(mining_api, "dataset_file_path", return_value=dataset_path), \
             patch("experiments.mining_api.os.path.getmtime", side_effect=[123.0, 123.0, 123.0, 124.0, 124.0]), \
             patch.object(mining_api, "load_dataset_facts_for_chainer", return_value=facts), \
             patch.object(mining_api, "_create_chainer_service", side_effect=[service_a, service_b]):
            first = mining_api.reload_petta_dataset_if_ready(force=False)
            second = mining_api.reload_petta_dataset_if_ready(force=False)
            third = mining_api.reload_petta_dataset_if_ready(force=False)

        self.assertEqual(first["status"], "initialized")
        self.assertEqual(second["status"], "unchanged")
        self.assertEqual(third["status"], "reinitialized")
        self.assertEqual(first["compiled_fact_count"], 2)
        self.assertEqual(second["compiled_fact_count"], 2)
        self.assertEqual(third["compiled_fact_count"], 2)
        self.assertEqual(service_a.added_atoms, facts)
        self.assertEqual(service_b.added_atoms, facts)
        self.assertEqual(mining_api.chainer_dataset_facts, facts)


if __name__ == "__main__":
    unittest.main()
