import os
import tempfile
import unittest
from unittest.mock import patch

from experiments.ingestion.pipeline import run_ingestion
from experiments.ingestion.remote_client import (
    MetadataExtractorError,
    RemoteIngestionResult,
)


REMOTE_RESULT = RemoteIngestionResult(
    dataset_lines=[
        '((engagement mindplex_A_1 "Low") (STV 0.2 0.9))',
        '((audience-expertise mindplex_A_1 "Expert") (STV 0.8 0.9))',
    ],
    record_count=1,
    fact_count=2,
    plan_fingerprint="a" * 64,
    planner="gemini",
    model="gemini-test",
    properties=["engagement", "audience-expertise"],
    usage={"input_tokens": 10, "output_tokens": 4},
)


class TestIngestionPipeline(unittest.TestCase):
    @patch("experiments.ingestion.pipeline.MetadataExtractorClient")
    @patch("experiments.ingestion.pipeline.MindplexFetcher")
    def test_fetches_remotely_enriches_and_atomically_writes_dataset(
        self, fetcher_class, client_class
    ):
        fetcher_class.return_value.fetch_all.return_value = [
            {"id": "A_1", "content": "Article", "likes": 2, "comments": 1}
        ]
        client_class.from_env.return_value.ingest.return_value = REMOTE_RESULT
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "data.metta")
            result = run_ingestion(username="alice", output_path=output_path)
            with open(output_path, encoding="utf-8") as handle:
                content = handle.read()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["facts"], 2)
        self.assertEqual(result["plan_fingerprint"], "a" * 64)
        self.assertEqual(content, "\n".join(REMOTE_RESULT.dataset_lines) + "\n")
        fetcher_class.assert_called_once_with(username="alice")
        client_class.from_env.return_value.ingest.assert_called_once()

    @patch("experiments.ingestion.pipeline.MetadataExtractorClient")
    @patch("experiments.ingestion.pipeline.MindplexFetcher")
    def test_remote_failure_preserves_existing_dataset(
        self, fetcher_class, client_class
    ):
        fetcher_class.return_value.fetch_all.return_value = [{"id": "A_1"}]
        client_class.from_env.return_value.ingest.side_effect = MetadataExtractorError(
            "unavailable"
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "data.metta")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("existing\n")
            result = run_ingestion(username="alice", output_path=output_path)
            with open(output_path, encoding="utf-8") as handle:
                content = handle.read()

        self.assertEqual(result["code"], "remote_ingestion_failed")
        self.assertEqual(content, "existing\n")

    @patch("experiments.ingestion.pipeline.MindplexFetcher")
    def test_no_articles_does_not_call_remote_service(self, fetcher_class):
        fetcher_class.return_value.fetch_all.return_value = []
        result = run_ingestion(username="alice")
        self.assertEqual(result["code"], "no_articles")

    def test_non_mindplex_sources_are_rejected(self):
        result = run_ingestion(username="alice", source_name="other")
        self.assertEqual(result["code"], "unsupported_source")


if __name__ == "__main__":
    unittest.main()
