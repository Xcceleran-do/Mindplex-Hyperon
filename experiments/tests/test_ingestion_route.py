import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

from experiments.api.routes import register_core_routes


class TestIngestionRoute(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.run_ingestion = MagicMock()
        self.invalidate_dataset = MagicMock()
        register_core_routes(
            self.app,
            logger=MagicMock(),
            run_ingestion=self.run_ingestion,
            invalidate_chainer_dataset=self.invalidate_dataset,
            dataset_file_path=lambda: "/tmp/data.metta",
            get_chainer_service=lambda: MagicMock(health=lambda: {"status": "ok"}),
            default_conjunction_count=2,
            default_min_support=3,
            default_chain_depth=3,
            mining_jobs={},
            mining_job_type=MagicMock,
            run_mining_task=MagicMock(),
            simulate_engagement=MagicMock(return_value={}),
            run_chainer_query=MagicMock(return_value=[]),
            make_json_safe=lambda value: value,
        )
        self.client = self.app.test_client()

    def test_rejects_removed_local_ingestion_options(self):
        response = self.client.post(
            "/api/ingest",
            json={"username": "alice", "output_path": "/tmp/other.metta"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json["error"]["code"], "unsupported_ingestion_option")

    def test_rejects_non_mindplex_source_and_oversized_limit(self):
        response = self.client.post(
            "/api/ingest", json={"username": "alice", "source": "other"}
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/api/ingest", json={"username": "alice", "limit": 101}
        )
        self.assertEqual(response.status_code, 400)

    @patch("experiments.api.routes.run_ingestion_request")
    def test_maps_remote_failure_without_invalidating_dataset(self, run_request):
        run_request.return_value = {
            "status": "error",
            "code": "remote_ingestion_failed",
        }
        response = self.client.post("/api/ingest", json={"username": "alice"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json["error"]["code"], "metadata_extractor_unavailable"
        )
        self.invalidate_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
