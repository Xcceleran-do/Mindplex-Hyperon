import unittest
from unittest.mock import patch, MagicMock
import datetime
from experiments.ingestion.analyzer import ArticleAnalyzer
from experiments.ingestion.config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS,
    RETENTION_BUCKETS, DETERMINISTIC_STV, AI_FAILURE_STV, UNKNOWN_STV
)


class TestArticleAnalyzer(unittest.TestCase):
    """Test suite for ArticleAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ArticleAnalyzer(api_key="test-key")
        self.analyzer_no_key = ArticleAnalyzer(api_key=None)

    def test_parse_read_time_with_valid_string(self):
        """Test extracting minutes from read time string"""
        self.assertEqual(self.analyzer.parse_read_time("20 min read"), 20)
        self.assertEqual(self.analyzer.parse_read_time("5 min read."), 5)
        self.assertEqual(self.analyzer.parse_read_time("123 minutes"), 123)

    def test_parse_read_time_with_invalid_string(self):
        """Test parse_read_time handles invalid inputs"""
        self.assertEqual(self.analyzer.parse_read_time(""), 0)
        self.assertEqual(self.analyzer.parse_read_time(None), 0)
        self.assertEqual(self.analyzer.parse_read_time("no numbers here"), 0)

    def test_calculate_date_period_current(self):
        """Test date period calculation for recent dates"""
        now = datetime.datetime.now()
        recent_date = (now - datetime.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(self.analyzer.calculate_date_period(recent_date), "Current")

    def test_calculate_date_period_recent(self):
        """Test date period calculation for dates within 30 days"""
        now = datetime.datetime.now()
        date_15_days_ago = (now - datetime.timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(self.analyzer.calculate_date_period(date_15_days_ago), "Recent")

    def test_calculate_date_period_older(self):
        """Test date period calculation for dates within 90 days"""
        now = datetime.datetime.now()
        date_60_days_ago = (now - datetime.timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(self.analyzer.calculate_date_period(date_60_days_ago), "Older")

    def test_calculate_date_period_archived(self):
        """Test date period calculation for very old dates"""
        now = datetime.datetime.now()
        date_200_days_ago = (now - datetime.timedelta(days=200)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertEqual(self.analyzer.calculate_date_period(date_200_days_ago), "Archived")

    def test_calculate_date_period_invalid(self):
        """Test date period with invalid date format"""
        self.assertEqual(self.analyzer.calculate_date_period(None), "Unknown")
        self.assertEqual(self.analyzer.calculate_date_period("invalid-date"), "Unknown")

    def test_discretize_value_length_short(self):
        """Test length discretization for short articles"""
        label, stv = self.analyzer.discretize_value(300, LENGTH_BUCKETS)
        self.assertEqual(label, "Short")
        self.assertEqual(stv, (0.1, 0.3))

    def test_discretize_value_length_medium(self):
        """Test length discretization for medium articles"""
        label, stv = self.analyzer.discretize_value(1000, LENGTH_BUCKETS)
        self.assertEqual(label, "Medium")
        self.assertEqual(stv, (0.4, 0.7))

    def test_discretize_value_length_long(self):
        """Test length discretization for long articles"""
        label, stv = self.analyzer.discretize_value(2000, LENGTH_BUCKETS)
        self.assertEqual(label, "Long")
        self.assertEqual(stv, (0.8, 1.0))

    def test_discretize_value_reading_time(self):
        """Test reading time discretization"""
        label, stv = self.analyzer.discretize_value(3, READING_TIME_BUCKETS)
        self.assertEqual(label, "Short")

    def test_discretize_value_engagement(self):
        """Test engagement discretization"""
        label, stv = self.analyzer.discretize_value(75, ENGAGEMENT_BUCKETS)
        self.assertEqual(label, "High")

    def test_discretize_value_unknown(self):
        """Test discretization with None value"""
        label, stv = self.analyzer.discretize_value(None, LENGTH_BUCKETS)
        self.assertEqual(label, "Unknown")
        self.assertEqual(stv, UNKNOWN_STV)

    def test_calculate_proportional_stv_with_bounds(self):
        """Test proportional STV calculation with bounds"""
        stv = self.analyzer.calculate_proportional_stv(250, "Short", (0.1, 0.3), bucket_bounds=(0, 500))
        strength, confidence = stv
        # 250/500 = 0.5, so strength should be 0.1 + 0.5 * (0.3 - 0.1) = 0.1 + 0.1 = 0.2
        self.assertAlmostEqual(strength, 0.2, places=1)
        self.assertAlmostEqual(confidence, 0.9, places=1)

    def test_calculate_proportional_stv_without_bounds(self):
        """Test proportional STV calculation defaults to midpoint without bounds"""
        stv = self.analyzer.calculate_proportional_stv(100, "Medium", (0.4, 0.7))
        strength, confidence = stv
        # Default normalized = 0.5
        self.assertAlmostEqual(strength, 0.55, places=1)
        self.assertEqual(confidence, 0.9)

    def test_calculate_proportional_stv_clamping(self):
        """Test that proportional STV values are clamped to valid range"""
        # Test with value outside bounds
        stv = self.analyzer.calculate_proportional_stv(1000, "Long", (0.8, 1.0), bucket_bounds=(0, 500))
        strength, confidence = stv
        self.assertLessEqual(strength, 1.0)
        self.assertGreaterEqual(strength, 0.0)

    def test_call_asi_api_success(self):
        """Test successful API call to ASI"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"tone": "Formal"}'}}]
        }
        
        with patch('experiments.ingestion.analyzer.requests.post', return_value=mock_response):
            result = self.analyzer.call_asi_api([{"role": "user", "content": "test"}])
            self.assertEqual(result["choices"][0]["message"]["content"], '{"tone": "Formal"}')

    def test_call_asi_api_failure(self):
        """Test API call failure handling"""
        with patch('experiments.ingestion.analyzer.requests.post', side_effect=Exception("Connection error")):
            with self.assertRaises(Exception):
                self.analyzer.call_asi_api([{"role": "user", "content": "test"}])

    def test_enrich_with_ai_no_api_key(self):
        """Test AI enrichment when no API key is available"""
        article = {"id": 1, "post_title": "Test", "content": "test content"}
        result = self.analyzer_no_key.enrich_with_ai(article)
        
        # Should return Unknown values for all fields
        self.assertEqual(result["tone"]["value"], "Unknown")
        self.assertEqual(result["audience_expertise"]["value"], "Unknown")
        self.assertEqual(result["content_type"]["value"], "Unknown")
        self.assertEqual(result["primary_goal"]["value"], "Unknown")
        self.assertEqual(result["audience_sentiment"]["value"], "Unknown")

    @patch('experiments.ingestion.analyzer.ArticleAnalyzer.call_asi_api')
    def test_enrich_with_ai_success(self, mock_api):
        """Test successful AI enrichment with valid response"""
        mock_api.return_value = {
            "choices": [{
                "message": {
                    "content": '''{
                        "tone": {"value": "Formal", "strength": 0.9, "confidence": 0.95},
                        "audience_expertise": {"value": "Advanced", "strength": 0.85, "confidence": 0.9},
                        "content_type": {"value": "Tutorial", "strength": 0.8, "confidence": 0.88},
                        "primary_goal": {"value": "Inform", "strength": 0.92, "confidence": 0.93},
                        "audience_sentiment": {"value": "Positive", "strength": 0.75, "confidence": 0.82}
                    }'''
                }
            }]
        }
        
        article = {"id": 1, "post_title": "Test", "content": "test content"}
        result = self.analyzer.enrich_with_ai(article)
        
        self.assertEqual(result["tone"]["value"], "Formal")
        self.assertAlmostEqual(result["tone"]["strength"], 0.9, places=2)
        self.assertAlmostEqual(result["tone"]["confidence"], 0.95, places=2)

    @patch('experiments.ingestion.analyzer.ArticleAnalyzer.call_asi_api')
    def test_enrich_with_ai_invalid_response(self, mock_api):
        """Test AI enrichment with malformed response"""
        mock_api.return_value = {"choices": [{"message": {"content": "invalid json"}}]}
        
        article = {"id": 1, "post_title": "Test", "content": "test content"}
        result = self.analyzer.enrich_with_ai(article)
        
        # Should return Unknown values on parse error
        self.assertEqual(result["tone"]["value"], "Unknown")

    def test_process_basic_article(self):
        """Test processing a basic article"""
        article = {
            "id": 1,
            "post_title": "Test Article",
            "content": [{"type": "p", "content": "This is a test article with some content."}],
            "min_to_read": "5 min read",
            "published_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "views": 100,
            "likes": 50,
            "comments": 10,
            "author_username": "test_author",
            "categories": [{"slug": "tech"}]
        }
        
        with patch.object(self.analyzer, 'enrich_with_ai', return_value={
            "tone": {"value": "Formal", "strength": 0.8, "confidence": 0.9},
            "audience_expertise": {"value": "Intermediate", "strength": 0.7, "confidence": 0.8},
            "content_type": {"value": "Tutorial", "strength": 0.8, "confidence": 0.85},
            "primary_goal": {"value": "Inform", "strength": 0.9, "confidence": 0.95},
            "audience_sentiment": {"value": "Positive", "strength": 0.85, "confidence": 0.9}
        }):
            result = self.analyzer.process(article)
            
            self.assertIn("enriched_metadata", result)
            metadata = result["enriched_metadata"]
            
            self.assertIn("length", metadata)
            self.assertIn("reading_time", metadata)
            self.assertIn("author", metadata)
            self.assertEqual(metadata["author"]["value"], "test_author")
            self.assertEqual(metadata["author"]["stv"], DETERMINISTIC_STV)

    def test_process_article_with_rank_stats(self):
        """Test processing article with rank statistics"""
        article = {
            "id": 1,
            "post_title": "Test",
            "content": [{"type": "p", "content": "Test"}],
            "min_to_read": "3 min",
            "published_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "views": 100,
            "likes": 50,
            "comments": 10,
            "author_username": "author",
            "categories": []
        }
        
        rank_stats = {1: 5}  # Article ranked #5
        
        with patch.object(self.analyzer, 'enrich_with_ai', return_value={
            "tone": {"value": "Formal", "strength": 0.8, "confidence": 0.9},
            "audience_expertise": {"value": "Beginner", "strength": 0.7, "confidence": 0.8},
            "content_type": {"value": "News", "strength": 0.8, "confidence": 0.85},
            "primary_goal": {"value": "Inform", "strength": 0.9, "confidence": 0.95},
            "audience_sentiment": {"value": "Neutral", "strength": 0.85, "confidence": 0.9}
        }):
            result = self.analyzer.process(article, rank_stats=rank_stats)
            metadata = result["enriched_metadata"]
            
            self.assertEqual(metadata["popularity"]["value"], "Top_10")
            self.assertEqual(metadata["popularity"]["stv"], DETERMINISTIC_STV)


if __name__ == "__main__":
    unittest.main()
