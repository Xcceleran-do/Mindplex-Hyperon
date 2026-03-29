import json
import tempfile
import unittest
from pathlib import Path

from experiments.ingestion.pipeline import run_ingestion


class IngestionToolTest(unittest.TestCase):
    def test_json_ingestion_writes_metta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_json = tmp_path / "articles.json"
            output_metta = tmp_path / "out.metta"

            payload = [
                {
                    "id": "16624",
                    "title": "Community Reward Campaign Guidelines",
                    "author": "Hruy",
                    "category": "opinion",
                    "reading_time": 5,
                    "engagement": 0.1,
                    "content_type": "Tutorial",
                    "tone": "Instructional",
                },
                {
                    "id": "12110",
                    "title": "Beyond the Hype",
                    "author": "Hruy",
                    "category": "agi",
                    "reading_time": 10,
                    "engagement": 0.8,
                    "content_type": "Opinion",
                    "tone": "Formal",
                },
            ]
            input_json.write_text(json.dumps(payload), encoding="utf-8")

            result = run_ingestion(
                sources=[str(input_json)],
                output_path=str(output_metta),
                subject_prefix="A",
            )

            self.assertEqual(result["status"], "success")
            self.assertTrue(output_metta.exists())

            content = output_metta.read_text(encoding="utf-8")
            self.assertIn("((title A_16624", content)
            self.assertIn("(STV", content)
            self.assertIn("((reading-time A_16624", content)
            self.assertIn("((engagement A_16624", content)

    def test_text_ingestion_extracts_basic_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_text = tmp_path / "doc.txt"
            output_metta = tmp_path / "out_text.metta"

            input_text.write_text(
                "A practical guide to recommendation systems. "
                "It explains ranking and personalization clearly.",
                encoding="utf-8",
            )

            result = run_ingestion(
                sources=[str(input_text)],
                output_path=str(output_metta),
                subject_prefix="A",
            )

            self.assertEqual(result["status"], "success")
            content = output_metta.read_text(encoding="utf-8")
            self.assertIn("((content-type", content)
            self.assertIn("((word-count", content)

    def test_author_alias_and_explicit_model_stv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_json = tmp_path / "model_scored.json"
            output_metta = tmp_path / "out_model.metta"

            payload = [
                {
                    "id": "42",
                    "author": "Ada",
                    "tone": {
                        "value": "Formal",
                        "confidence": 0.93,
                        "strength": 0.88,
                    },
                }
            ]
            input_json.write_text(json.dumps(payload), encoding="utf-8")

            result = run_ingestion(
                sources=[str(input_json)],
                output_path=str(output_metta),
                subject_prefix="A",
            )

            self.assertEqual(result["status"], "success")
            content = output_metta.read_text(encoding="utf-8")
            self.assertIn("((author A_42 \"Ada\")", content)
            self.assertIn("((authored-by A_42 \"Ada\")", content)
            self.assertIn("((tone A_42 \"Formal\") (STV 0.93 0.88))", content)

    def test_result_contains_agent_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_json = tmp_path / "basic.json"
            output_metta = tmp_path / "basic_out.metta"
            input_json.write_text(json.dumps([{"id": "1", "author": "Nora"}]), encoding="utf-8")

            result = run_ingestion(
                sources=[str(input_json)],
                output_path=str(output_metta),
            )

            self.assertEqual(result["status"], "success")
            self.assertIn("agent_reports", result)
            self.assertGreater(len(result["agent_reports"]), 0)
            names = [report["name"] for report in result["agent_reports"]]
            self.assertIn("sentiment-analysis", names)
            self.assertIn("content-classification", names)
            self.assertIn("semantic-parser", names)
            self.assertIn("recommendation-signal", names)

    def test_recommendation_agents_emit_expected_triples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_text = tmp_path / "rich_doc.txt"
            output_metta = tmp_path / "rich_out.metta"
            input_text.write_text(
                "This practical guide explains how to improve recommendation quality "
                "with clear steps, benchmark metrics, and helpful analysis.",
                encoding="utf-8",
            )

            result = run_ingestion(
                sources=[str(input_text)],
                output_path=str(output_metta),
                subject_prefix="A",
            )

            self.assertEqual(result["status"], "success")
            content = output_metta.read_text(encoding="utf-8")
            self.assertIn("((audience-sentiment", content)
            self.assertIn("((content-class", content)
            self.assertIn("((semantic-keywords", content)
            self.assertIn("((recommendation-utility", content)

    def test_invalid_property_coverage_returns_error(self) -> None:
        result = run_ingestion(
            username="Hruy",
            min_property_coverage=1.5,
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("min_property_coverage", result["message"])


if __name__ == "__main__":
    unittest.main()
