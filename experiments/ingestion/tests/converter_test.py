import unittest
from experiments.ingestion.converter import JsonToMetta
from experiments.ingestion.config import DETERMINISTIC_STV, UNKNOWN_STV


class TestJsonToMetta(unittest.TestCase):
    """Test suite for JsonToMetta converter"""

    def setUp(self):
        """Set up test fixtures"""
        self.converter = JsonToMetta()

    def test_convert_empty_list(self):
        """Test converting empty article list"""
        result = self.converter.convert([])
        self.assertEqual(result, "")

    def test_convert_single_article_minimal(self):
        """Test converting single article with minimal fields"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": "Test Article", "stv": DETERMINISTIC_STV},
                    "author": {"value": "John Doe", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Should contain both properties
        self.assertIn("(title A_1 \"Test Article\")", result)
        self.assertIn("(authored-by A_1 \"John Doe\")", result)
        self.assertIn("(STV 1.0 1.0)", result)

    def test_convert_article_with_all_properties(self):
        """Test converting article with all enriched properties"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "length": {"value": "Medium", "stv": (0.5, 0.9)},
                    "reading_time": {"value": "5 min", "stv": (0.6, 0.88)},
                    "tone": {"value": "Formal", "stv": (0.8, 0.92)},
                    "audience_expertise": {"value": "Intermediate", "stv": (0.7, 0.85)},
                    "content_type": {"value": "Tutorial", "stv": (0.75, 0.9)},
                    "date_period": {"value": "Recent", "stv": DETERMINISTIC_STV},
                    "primary_goal": {"value": "Inform", "stv": (0.85, 0.95)},
                    "audience_sentiment": {"value": "Positive", "stv": (0.8, 0.88)},
                    "popularity": {"value": "High", "stv": DETERMINISTIC_STV},
                    "engagement": {"value": "Medium", "stv": (0.5, 0.85)},
                    "author": {"value": "Jane Smith", "stv": DETERMINISTIC_STV},
                    "category": {"value": "tech", "stv": DETERMINISTIC_STV},
                    "title": {"value": "AI and Machine Learning", "stv": DETERMINISTIC_STV},
                    "topic": {"value": "AI", "stv": (0.9, 0.95)}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        lines = result.split("\n")
        
        # Should have 13 properties (one for each key in enriched_metadata)
        self.assertGreaterEqual(len(lines), 13)
        
        # Check specific properties are present
        self.assertTrue(any("length" in line for line in lines))
        self.assertTrue(any("tone" in line for line in lines))
        self.assertTrue(any("audience-expertise" in line for line in lines))
        self.assertTrue(any("content-type" in line for line in lines))
        self.assertTrue(any("primary-goal" in line for line in lines))

    def test_convert_sanitizes_special_characters(self):
        """Test that special characters in values are properly escaped"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": 'Article "with quotes"', "stv": DETERMINISTIC_STV},
                    "author": {"value": "Author\nwith\nnewlines", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Quotes should be escaped
        self.assertIn('\\"', result)
        # Newlines should be replaced with spaces
        self.assertIn("with newlines", result)

    def test_convert_multiple_articles(self):
        """Test converting multiple articles"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {"title": {"value": "Article 1", "stv": DETERMINISTIC_STV}}
            },
            {
                "id": 2,
                "enriched_metadata": {"title": {"value": "Article 2", "stv": DETERMINISTIC_STV}}
            },
            {
                "id": 3,
                "enriched_metadata": {"title": {"value": "Article 3", "stv": DETERMINISTIC_STV}}
            }
        ]
        
        result = self.converter.convert(articles)
        lines = result.split("\n")
        
        # Should have entries for all 3 articles
        self.assertIn("(title A_1 \"Article 1\")", result)
        self.assertIn("(title A_2 \"Article 2\")", result)
        self.assertIn("(title A_3 \"Article 3\")", result)

    def test_convert_skips_unknown_values(self):
        """Test that Unknown values are skipped"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": "Test", "stv": DETERMINISTIC_STV},
                    "topic": {"value": "Unknown", "stv": UNKNOWN_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Title should be present
        self.assertIn("(title A_1 \"Test\")", result)
        # Unknown topic should be skipped
        self.assertNotIn("Unknown", result)

    def test_convert_handles_missing_enriched_metadata(self):
        """Test handling articles without enriched_metadata"""
        articles = [
            {
                "id": 1
                # No enriched_metadata
            }
        ]
        
        # Should not crash
        result = self.converter.convert(articles)
        self.assertEqual(result, "")

    def test_convert_author_as_alias(self):
        """Test that author metadata creates both author and authored-by properties"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "author": {"value": "Bob", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Both properties should be present
        self.assertIn("(authored-by A_1 \"Bob\")", result)

    def test_convert_stv_values_preserved(self):
        """Test that STV values are correctly preserved in output"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "engagement": {"value": "High", "stv": (0.75, 0.88)},
                    "tone": {"value": "Casual", "stv": (0.6, 0.75)}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Check STV values are present with correct format
        self.assertIn("(STV 0.75 0.88)", result)
        self.assertIn("(STV 0.6 0.75)", result)

    def test_convert_numeric_id_converted_to_string(self):
        """Test that numeric article IDs are properly converted to A_<id> format"""
        articles = [
            {
                "id": 12345,
                "enriched_metadata": {
                    "title": {"value": "Test", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Should use A_12345 format
        self.assertIn("(title A_12345", result)

    def test_convert_old_format_fallback(self):
        """Test backward compatibility with old metadata format (without STV)"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "author": "Simple String Value"  # Old format without dict
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Old format should use default STV
        self.assertIn("(STV 0.5 0.5)", result)

    def test_convert_entities(self):
        """Test converting entities to has-topic triples"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "entities": [
                        {"value": "AI", "strength": 0.9, "confidence": 0.8},
                        {"value": "Hyperon", "strength": 0.95, "confidence": 0.9}
                    ]
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        self.assertIn("(has-topic A_1 \"AI\")", result)
        self.assertIn("(STV 0.9 0.8)", result)
        self.assertIn("(has-topic A_1 \"Hyperon\")", result)
        self.assertIn("(STV 0.95 0.9)", result)


class TestJsonToMettaEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.converter = JsonToMetta()

    def test_convert_handles_empty_string_values(self):
        """Test that empty string values are skipped"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": "", "stv": DETERMINISTIC_STV},
                    "author": {"value": "Jane", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Empty title should be skipped
        self.assertNotIn("(title", result)
        # Author should be present
        self.assertIn("(authored-by", result)

    def test_convert_unicode_characters(self):
        """Test handling of unicode characters"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": "Déjà vu: AI and 中文", "stv": DETERMINISTIC_STV},
                    "author": {"value": "François Müller", "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Should handle unicode without crashing
        self.assertIn("Déjà", result)
        self.assertIn("François", result)

    def test_convert_very_long_values(self):
        """Test handling of very long property values"""
        long_text = "A" * 1000
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "title": {"value": long_text, "stv": DETERMINISTIC_STV}
                }
            }
        ]
        
        result = self.converter.convert(articles)
        
        # Should handle long values
        self.assertIn(long_text, result)

    def test_convert_maintains_order(self):
        """Test that property processing order is consistent"""
        articles = [
            {
                "id": 1,
                "enriched_metadata": {
                    "length": {"value": "Medium", "stv": (0.5, 0.8)},
                    "reading_time": {"value": "5", "stv": (0.6, 0.75)},
                    "tone": {"value": "Formal", "stv": (0.7, 0.85)}
                }
            }
        ]
        
        result1 = self.converter.convert(articles)
        result2 = self.converter.convert(articles)
        
        # Results should be consistent
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
