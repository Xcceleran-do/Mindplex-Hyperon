import unittest

from experiments.ingestion.mind_adapter import (
    _engagement_bucket,
    _normalize_category,
    parse_impression_token,
)


class MindAdapterTest(unittest.TestCase):
    def test_parse_impression_token(self):
        self.assertEqual(parse_impression_token("N123-1"), ("N123", 1))
        self.assertEqual(parse_impression_token("N124-0"), ("N124", 0))
        self.assertEqual(parse_impression_token("N125"), ("N125", None))

    def test_engagement_bucket(self):
        self.assertEqual(_engagement_bucket(0.0), "Low")
        self.assertEqual(_engagement_bucket(0.07), "Medium")
        self.assertEqual(_engagement_bucket(0.25), "High")

    def test_normalize_category(self):
        self.assertEqual(_normalize_category("Sci Tech"), "sci-tech")
        self.assertEqual(_normalize_category("  U.S. News  "), "us-news")


if __name__ == "__main__":
    unittest.main()
