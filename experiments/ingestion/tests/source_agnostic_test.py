import unittest

from experiments.ingestion.converter import JsonToMetta
from experiments.ingestion.llm_client import LLMClient
from experiments.ingestion.orchestrator import IngestionOrchestrator
from experiments.ingestion.planner import ExtractionPlanner


class TestSourceAgnosticIngestion(unittest.TestCase):
    def test_heuristic_plan_and_agents_handle_arbitrary_json(self):
        records = [
            {
                "uuid": "book-1",
                "name": "A Practical Guide",
                "author": "Ada",
                "genre": "technical",
                "rating": 4.8,
                "views": 20,
                "likes": 5,
                "comments": 3,
                "min_to_read": "5 min read",
                "published_at": "2026-05-01",
                "description": "A hands-on guide for building robust services.",
            },
            {
                "uuid": "book-2",
                "name": "Quiet Systems",
                "author": "Ada",
                "genre": "technical",
                "rating": 3.6,
                "views": 60,
                "likes": 10,
                "comments": 5,
                "min_to_read": "12 min read",
                "published_at": "2025-01-01",
                "description": "A reflective essay about software operations.",
            },
        ]

        planner = ExtractionPlanner(llm_client=LLMClient(api_key=None))
        plan = planner.build_plan(records, source_name="books")

        self.assertEqual(plan.source_name, "books")
        self.assertIn("uuid", plan.id_fields)
        self.assertTrue(any(spec.name == "genre" for spec in plan.properties))
        self.assertTrue(any(spec.agent == "numeric_bucket" for spec in plan.properties))
        self.assertTrue(any(spec.name == "audience-expertise" for spec in plan.properties))
        self.assertTrue(any(spec.name == "engagement" and spec.agent == "calculated_metric" for spec in plan.properties))
        self.assertTrue(any(spec.name == "length" and spec.agent == "calculated_metric" for spec in plan.properties))
        self.assertTrue(any(spec.name == "reading-time" and spec.agent == "calculated_metric" for spec in plan.properties))
        self.assertFalse(any(spec.name in ("author", "title") for spec in plan.properties))

        orchestrator = IngestionOrchestrator(plan, corpus_records=records)
        enriched = [orchestrator.process(record) for record in records]

        self.assertIn("enriched_metadata", enriched[0])
        self.assertIn("genre", enriched[0]["enriched_metadata"])
        self.assertIn("rating_level", enriched[0]["enriched_metadata"])
        self.assertEqual(enriched[0]["enriched_metadata"]["length"]["value"], "Short")
        self.assertEqual(enriched[0]["enriched_metadata"]["reading_time"]["value"], "Medium")
        self.assertIn("engagement", enriched[0]["enriched_metadata"])
        self.assertEqual(enriched[0]["enriched_metadata"]["engagement"]["value"], "Low")
        self.assertEqual(enriched[1]["enriched_metadata"]["engagement"]["value"], "High")
        self.assertEqual(enriched[1]["enriched_metadata"]["engagement"]["stv"], (0.75, 0.9))

        metta = JsonToMetta(include_author_alias=False).convert(enriched)

        self.assertIn('(genre A_book-1 "technical")', metta)
        self.assertIn("(rating-level A_book-1", metta)
        self.assertIn('(engagement A_book-1 "Low")', metta)
        self.assertNotIn("(author ", metta)
        self.assertNotIn("(authored-by ", metta)


if __name__ == "__main__":
    unittest.main()
