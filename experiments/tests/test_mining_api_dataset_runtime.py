import os
import tempfile
import unittest
from unittest.mock import patch

try:
    from experiments import mining_api
    from experiments.api import runtime as petta_runtime
    MINING_API_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-specific optional deps
    mining_api = None
    petta_runtime = None
    MINING_API_IMPORT_ERROR = exc


class FakeChainerClient:
    def __init__(self) -> None:
        self.ensured_names: list[str] = []
        self.uploads: list[tuple[str, list[str]]] = []

    def ensure_knowledge_base(self, name: str) -> str:
        self.ensured_names.append(name)
        return "remote-kb"

    def add_statements(self, kb_id: str, sources: list[str]) -> dict:
        self.uploads.append((kb_id, list(sources)))
        return {"items": []}


@unittest.skipUnless(mining_api is not None, f"mining_api dependencies unavailable: {MINING_API_IMPORT_ERROR}")
class TestMiningApiDatasetRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_state = {
            "chainer_client": petta_runtime.chainer_client,
            "chainer_dataset_path": petta_runtime.chainer_dataset_path,
            "chainer_dataset_mtime": petta_runtime.chainer_dataset_mtime,
            "chainer_dataset_digest": petta_runtime.chainer_dataset_digest,
            "chainer_dataset_facts": list(petta_runtime.chainer_dataset_facts),
            "chainer_dataset_compiled_count": petta_runtime.chainer_dataset_compiled_count,
            "chainer_rule_atoms": list(petta_runtime.chainer_rule_atoms),
            "chainer_kb_id": petta_runtime.chainer_kb_id,
            "chainer_kb_signature": petta_runtime.chainer_kb_signature,
        }
        petta_runtime.chainer_client = FakeChainerClient()
        petta_runtime.chainer_dataset_path = None
        petta_runtime.chainer_dataset_mtime = None
        petta_runtime.chainer_dataset_digest = None
        petta_runtime.chainer_dataset_facts = []
        petta_runtime.chainer_dataset_compiled_count = 0
        petta_runtime.chainer_rule_atoms = []
        petta_runtime.chainer_kb_id = None
        petta_runtime.chainer_kb_signature = None

    def tearDown(self) -> None:
        for name, value in self._saved_state.items():
            setattr(petta_runtime, name, value)

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
                '(: fact_09adff7baf1d79b37b8dc3fa (topic A_1 "AI") (STV 1.0 1.0))',
                '(: fact_6443803b182111112f76b650 (tone A_1 "Analytical") (STV 0.9 0.8))',
            ],
        )

    def test_default_dataset_path_resolves_from_project_root(self) -> None:
        from experiments.api import config as api_config

        expected = os.path.join(
            api_config.PROJECT_ROOT,
            "experiments",
            "atomspace_visualizer",
            "public",
            "data.metta",
        )
        self.assertEqual(api_config.dataset_file_path(), os.path.abspath(expected))
        self.assertNotIn("/experiments/experiments/", api_config.dataset_file_path())

    def test_dataset_is_uploaded_only_when_chainer_is_requested(self) -> None:
        dataset_text = '((topic A_1 "AI") (STV 1.0 1.0))\n'
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(dataset_text)
            path = handle.name
        try:
            with patch.object(petta_runtime, "dataset_file_path", return_value=path):
                loaded = petta_runtime.reload_petta_dataset_if_ready(force=False)
                self.assertEqual(loaded["status"], "loaded")
                self.assertEqual(petta_runtime.chainer_client.uploads, [])

                client, kb_id = petta_runtime.ensure_remote_chainer()
        finally:
            os.remove(path)

        self.assertIs(client, petta_runtime.chainer_client)
        self.assertEqual(kb_id, "remote-kb")
        self.assertEqual(len(petta_runtime.chainer_client.ensured_names), 1)
        self.assertEqual(
            petta_runtime.chainer_client.uploads,
            [("remote-kb", petta_runtime.chainer_dataset_facts)],
        )

    def test_remote_chainer_accepts_query_scoped_facts(self) -> None:
        dataset_text = (
            '((topic A_1 "AI") (STV 1.0 1.0))\n'
            '((topic A_2 "Science") (STV 1.0 1.0))\n'
        )
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(dataset_text)
            path = handle.name
        try:
            with patch.object(petta_runtime, "dataset_file_path", return_value=path):
                petta_runtime.reload_petta_dataset_if_ready()
                selected = [petta_runtime.dataset_facts()[0]]
                petta_runtime.ensure_remote_chainer(selected)
        finally:
            os.remove(path)

        self.assertEqual(
            petta_runtime.chainer_client.uploads,
            [("remote-kb", selected)],
        )

    def test_different_fact_scopes_use_different_remote_kbs(self) -> None:
        dataset_text = (
            '((topic A_1 "AI") (STV 1.0 1.0))\n'
            '((topic A_2 "Science") (STV 1.0 1.0))\n'
        )
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(dataset_text)
            path = handle.name
        try:
            with patch.object(petta_runtime, "dataset_file_path", return_value=path):
                petta_runtime.reload_petta_dataset_if_ready()
                facts = petta_runtime.dataset_facts()
                petta_runtime.ensure_remote_chainer([facts[0]])
                petta_runtime.ensure_remote_chainer([facts[1]])
        finally:
            os.remove(path)

        self.assertEqual(len(petta_runtime.chainer_client.ensured_names), 2)
        self.assertNotEqual(
            petta_runtime.chainer_client.ensured_names[0],
            petta_runtime.chainer_client.ensured_names[1],
        )

    def test_rules_recorded_before_initial_query_are_persisted_and_uploaded(self) -> None:
        dataset_text = '((tone A_1 "Analytical") (STV 1.0 1.0))\n'
        rule = (
            '(: rule_1 (Implication (Premises (tone $x "Analytical")) '
            '(Conclusions (engagement $x "High"))) (STV 0.8 0.9))'
        )
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write(dataset_text)
            path = handle.name
        rules_path = petta_runtime.chainer_rules_path(path)
        try:
            with patch.object(petta_runtime, "dataset_file_path", return_value=path):
                # This is the ordering that previously lost rules: mining
                # returned before the parent API had loaded its dataset.
                petta_runtime.record_chainer_rules([rule])
                self.assertEqual(petta_runtime.ordered_chainer_rules(), [rule])
                self.assertTrue(os.path.isfile(rules_path))

                # Simulate a process restart, then load and synchronize again.
                petta_runtime.chainer_dataset_path = None
                petta_runtime.chainer_dataset_mtime = None
                petta_runtime.chainer_dataset_digest = None
                petta_runtime.chainer_dataset_facts = []
                petta_runtime.chainer_rule_atoms = []
                petta_runtime.chainer_kb_id = None
                petta_runtime.chainer_kb_signature = None

                petta_runtime.reload_petta_dataset_if_ready()
                petta_runtime.ensure_remote_chainer()
        finally:
            os.remove(path)
            if os.path.exists(rules_path):
                os.remove(rules_path)

        self.assertEqual(petta_runtime.ordered_chainer_rules(), [rule])
        uploaded_statements = petta_runtime.chainer_client.uploads[-1][1]
        self.assertIn(rule, uploaded_statements)
        self.assertEqual(len(uploaded_statements), 2)

    def test_persisted_rules_are_rejected_after_dataset_changes(self) -> None:
        rule = (
            '(: rule_1 (Implication (Premises (tone $x "Analytical")) '
            '(Conclusions (engagement $x "High"))) (STV 0.8 0.9))'
        )
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write('((tone A_1 "Analytical") (STV 1.0 1.0))\n')
            path = handle.name
        rules_path = petta_runtime.chainer_rules_path(path)
        try:
            with patch.object(petta_runtime, "dataset_file_path", return_value=path):
                petta_runtime.record_chainer_rules([rule])
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write('((tone A_2 "Neutral") (STV 1.0 1.0))\n')
                petta_runtime.reload_petta_dataset_if_ready(force=True)
                self.assertEqual(petta_runtime.ordered_chainer_rules(), [])
        finally:
            os.remove(path)
            if os.path.exists(rules_path):
                os.remove(rules_path)


if __name__ == "__main__":
    unittest.main()
