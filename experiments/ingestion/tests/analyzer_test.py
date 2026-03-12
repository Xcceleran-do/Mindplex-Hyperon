import unittest
from unittest.mock import patch, MagicMock
import datetime
import re
from experiments.ingestion.analyzer import (
    ArticleAnalyzer, SemanticAgent, EntityLinkingAgent, 
    ClassificationAgent, SentimentAgent, OpenIEAgent, BaseProcessor
)
from experiments.ingestion.config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS,
    DETERMINISTIC_STV, UNKNOWN_STV
)

class TestProcessors(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-key"
        self.article = {"id": 1, "post_title": "Test", "content": [{"type": "p", "content": "test content"}]}

    def test_base_processor_snippet(self):
        proc = BaseProcessor(self.api_key)
        article = {"content": [{"type": "p", "content": "Paragraph 1"}, {"type": "p", "content": "Paragraph 2"}]}
        snippet = proc.get_text_snippet(article)
        self.assertEqual(snippet, "Paragraph 1 Paragraph 2")

    @patch('experiments.ingestion.analyzer.requests.post')
    def test_semantic_agent_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '''{
                        "tone": {"value": "Formal", "strength": 0.9, "confidence": 0.9},
                        "audience_expertise": {"value": "Advanced", "strength": 0.8, "confidence": 0.8},
                        "content_type": {"value": "Tutorial", "strength": 0.7, "confidence": 0.7},
                        "primary_goal": {"value": "Inform", "strength": 0.6, "confidence": 0.6},
                        "audience_sentiment": {"value": "Positive", "strength": 0.5, "confidence": 0.5}
                    }'''
                }
            }]
        }
        mock_post.return_value = mock_response
        
        proc = SemanticAgent(self.api_key)
        result = proc.process(self.article)
        self.assertEqual(result["tone"]["value"], "Formal")
        self.assertEqual(result["tone"]["strength"], 0.9)

    @patch('experiments.ingestion.analyzer.requests.post')
    def test_entity_linking_agent_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"entities": [{"value": "AI", "type": "tech", "strength": 0.9, "confidence": 0.9}]}'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        proc = EntityLinkingAgent(self.api_key)
        result = proc.process(self.article)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["value"], "AI")

    @patch('experiments.ingestion.analyzer.requests.post')
    def test_classification_agent_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"domain": "News", "format": "Article", "confidence": 0.9}'}}]
        }
        mock_post.return_value = mock_response
        proc = ClassificationAgent(self.api_key)
        result = proc.process(self.article)
        self.assertEqual(result["domain"], "News")

class TestArticleAnalyzer(unittest.TestCase):
    """Test suite for ArticleAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ArticleAnalyzer(api_key="test-key")

    def test_discretize_value(self):
        label, stv = self.analyzer.discretize_value(300, LENGTH_BUCKETS)
        self.assertEqual(label, "Short")
        self.assertEqual(stv, (0.1, 0.3))

    def test_calculate_proportional_stv(self):
        stv = self.analyzer.calculate_proportional_stv(250, "Short", (0.1, 0.3), bucket_bounds=(0, 500))
        self.assertAlmostEqual(stv[0], 0.2, places=1)

    def test_process_article(self):
        article = {
            "id": 1,
            "post_title": "Test Article",
            "content": "This is content",
            "min_to_read": "5 min",
            "published_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "views": 100, "likes": 50, "comments": 10,
            "author_username": "user",
            "categories": [{"slug": "tech"}]
        }
        
        with patch.object(self.analyzer.classifier, 'process', return_value={"domain": "News", "confidence": 0.9}):
            with patch.object(self.analyzer.semantic, 'process', return_value={
                "tone": {"value": "Formal", "strength": 0.8, "confidence": 0.9},
                "audience_expertise": {"value": "Intermediate", "strength": 0.7, "confidence": 0.8},
                "content_type": {"value": "Tutorial", "strength": 0.8, "confidence": 0.85},
                "primary_goal": {"value": "Inform", "strength": 0.9, "confidence": 0.95},
                "audience_sentiment": {"value": "Positive", "strength": 0.85, "confidence": 0.9}
            }):
                with patch.object(self.analyzer.sentiment, 'process', return_value={"sentiment": "Positive", "strength": 0.8, "confidence": 0.9}):
                    with patch.object(self.analyzer.entities, 'process', return_value={"entities": [{"value": "Metta"}]}):
                        with patch.object(self.analyzer.openie, 'process', return_value={"triples": []}):
                            result = self.analyzer.process(article)
                            self.assertIn("enriched_metadata", result)
                            self.assertEqual(result["enriched_metadata"]["tone"]["value"], "Formal")
                            self.assertEqual(result["enriched_metadata"]["domain"]["value"], "News")
                            self.assertEqual(len(result["enriched_metadata"]["entities"]), 1)

if __name__ == "__main__":
    unittest.main()
