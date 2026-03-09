import tempfile
import unittest
from pathlib import Path

from experiments.ingestion.adapters.bookcrossing_adapter import (
    build_bookcrossing_records,
    write_records_jsonl,
)
from experiments.ingestion.pipeline import run_ingestion


class BookcrossingAdapterTest(unittest.TestCase):
    def test_adapter_builds_records_and_ingests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (root / "BX-Books.csv").write_text(
                '"ISBN";"Book-Title";"Book-Author";"Year-Of-Publication";"Publisher"\n'
                '"0001";"AI for Humans";"Ada";"2003";"TechPress"\n'
                '"0002";"Data Stories";"Bob";"2018";"InsightHouse"\n',
                encoding="latin-1",
            )
            (root / "BX-Users.csv").write_text(
                '"User-ID";"Location";"Age"\n'
                '"10";"Addis";"27"\n'
                '"20";"Berlin";"43"\n',
                encoding="latin-1",
            )
            (root / "BX-Book-Ratings.csv").write_text(
                '"User-ID";"ISBN";"Book-Rating"\n'
                '"10";"0001";"8"\n'
                '"20";"0001";"6"\n'
                '"20";"0002";"0"\n',
                encoding="latin-1",
            )

            records = build_bookcrossing_records(str(root), limit_books=10)
            self.assertGreaterEqual(len(records), 2)
            self.assertIn("avg_rating", records[0])

            jsonl_path = write_records_jsonl(records, str(root / "prepared.jsonl"))
            result = run_ingestion(
                sources=[jsonl_path],
                output_path=str(root / "out.metta"),
            )

            self.assertEqual(result["status"], "success")
            content = (root / "out.metta").read_text(encoding="utf-8")
            self.assertIn("((content-type", content)
            self.assertIn("((author", content)


if __name__ == "__main__":
    unittest.main()
