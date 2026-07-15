import unittest

from experiments.ingestion.remote_client import (
    MetadataExtractorClient,
    MetadataExtractorError,
)


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def plan_response():
    return {
        "plan": {
            "source_name": "mindplex",
            "entity_type": "item",
            "id_fields": ["id"],
            "text_fields": ["content"],
            "properties": [
                {"name": "engagement"},
                {"name": "audience-expertise"},
            ],
            "version": 1,
            "planner": "gemini",
            "fingerprint": "a" * 64,
        },
        "model": "gemini-test",
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }


def record_result(record_id="1"):
    return {
        "source_id": record_id,
        "errors": [],
        "facts": [
            {
                "atom": f'(engagement A_{record_id} "Low")',
                "property_name": "engagement",
                "strength": 0.2,
                "confidence": 0.9,
            },
            {
                "atom": f'(audience-expertise A_{record_id} "Expert")',
                "property_name": "audience-expertise",
                "strength": 0.8,
                "confidence": 0.85,
            },
        ],
    }


class TestMetadataExtractorClient(unittest.TestCase):
    def test_discovers_once_extracts_in_chunks_and_serializes_dataset(self):
        session = FakeSession(
            [
                FakeResponse(plan_response()),
                FakeResponse(
                    {
                        "records": [record_result("1")],
                        "usage": {"input_tokens": 10, "output_tokens": 4},
                    }
                ),
                FakeResponse(
                    {
                        "records": [record_result("2")],
                        "usage": {"input_tokens": 11, "output_tokens": 5},
                    }
                ),
            ]
        )
        client = MetadataExtractorClient(
            base_url="http://extractor:8080",
            api_key="mindplex:secret",
            chunk_size=1,
            session=session,
        )

        result = client.ingest(
            [{"id": "1", "content": "one"}, {"id": "2", "content": "two"}]
        )

        self.assertEqual(len(session.calls), 3)
        self.assertTrue(session.calls[0][0].endswith("/v1/plans/discover"))
        self.assertTrue(session.calls[1][0].endswith("/v1/extract"))
        self.assertEqual(
            session.calls[0][1]["headers"]["Authorization"],
            "Bearer mindplex:secret",
        )
        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.fact_count, 4)
        self.assertEqual(result.usage, {"input_tokens": 23, "output_tokens": 10})
        self.assertIn(
            '((engagement A_1 "Low") (STV 0.2 0.9))',
            result.dataset_lines,
        )

    def test_rejects_missing_required_fact(self):
        result = record_result()
        result["facts"] = result["facts"][:1]
        session = FakeSession(
            [FakeResponse(plan_response()), FakeResponse({"records": [result]})]
        )
        client = MetadataExtractorClient(
            base_url="http://extractor:8080",
            api_key="mindplex:secret",
            session=session,
        )
        with self.assertRaisesRegex(MetadataExtractorError, "required facts"):
            client.ingest([{"id": "1"}])

    def test_rejects_unsafe_atom_from_remote_service(self):
        result = record_result()
        result["facts"][0]["atom"] = '(eval (py-eval "danger"))'
        session = FakeSession(
            [FakeResponse(plan_response()), FakeResponse({"records": [result]})]
        )
        client = MetadataExtractorClient(
            base_url="http://extractor:8080",
            api_key="mindplex:secret",
            session=session,
        )
        with self.assertRaisesRegex(MetadataExtractorError, "unsafe fact atom"):
            client.ingest([{"id": "1"}])


if __name__ == "__main__":
    unittest.main()
