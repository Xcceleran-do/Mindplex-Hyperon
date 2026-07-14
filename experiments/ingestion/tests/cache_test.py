import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from experiments.ingestion.cache import run_ingestion_request


class TestIngestionCache(unittest.TestCase):
    def _run(self, callback, output_path, **overrides):
        options = {
            "username": "alice",
            "source_name": "mindplex",
            "limit": 50,
            "output_path": output_path,
            "source_config": None,
        }
        options.update(overrides)
        return run_ingestion_request(callback, **options)

    def test_disabled_ingestion_does_not_call_pipeline(self):
        callback = MagicMock()
        with patch.dict(os.environ, {"INGESTION_ENABLED": "false"}):
            result = self._run(callback, "/tmp/unused-data.metta")

        self.assertEqual(result["code"], "ingestion_disabled")
        callback.assert_not_called()

    def test_same_username_reuses_fresh_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "data.metta")

            def ingest(**_kwargs):
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write("(Fact article value)\n")
                return {"status": "success", "records": 50, "facts": 1}

            callback = MagicMock(side_effect=ingest)
            with patch.dict(
                os.environ,
                {"INGESTION_ENABLED": "true", "INGESTION_CACHE_TTL_DAYS": "3"},
            ):
                first = self._run(callback, output_path)
                second = self._run(callback, output_path)

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(callback.call_count, 1)

    def test_different_username_or_force_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "data.metta")

            def ingest(**_kwargs):
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write("(Fact article value)\n")
                return {"status": "success", "records": 50, "facts": 1}

            callback = MagicMock(side_effect=ingest)
            with patch.dict(os.environ, {"INGESTION_ENABLED": "true"}):
                self._run(callback, output_path)
                self._run(callback, output_path, username="bob")
                forced = self._run(callback, output_path, username="bob", force=True)

        self.assertFalse(forced["cached"])
        self.assertEqual(callback.call_count, 3)

    def test_larger_limit_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "data.metta")

            def ingest(**_kwargs):
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write("(Fact article value)\n")
                return {"status": "success", "records": 50, "facts": 1}

            callback = MagicMock(side_effect=ingest)
            with patch.dict(os.environ, {"INGESTION_ENABLED": "true"}):
                self._run(callback, output_path, limit=25)
                result = self._run(callback, output_path, limit=50)

        self.assertFalse(result["cached"])
        self.assertEqual(callback.call_count, 2)


if __name__ == "__main__":
    unittest.main()
