import unittest
from unittest.mock import patch, MagicMock
import datetime
import re
from experiments.ingestion.analyzer import (
    DocumentAnalyzer, DynamicMetadataAgent, EntityLinkingAgent, 
    ClassificationAgent, SentimentAgent, OpenIEAgent, BaseProcessor
)
from experiments.ingestion.config import (
    LENGTH_BUCKETS, READING_TIME_BUCKETS, ENGAGEMENT_BUCKETS,
    DETERMINISTIC_STV, UNKNOWN_STV
)

class TestProcessors(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-key"
        self.document = {"id": 1, "post_title": "Test", "content": [{"type": "p", "content": "test content"}]}

    def test_base_processor_snippet(self):
        proc = BaseProcessor(self.api_key)
        document = {"content": [{"type": "p", "content": "Paragraph 1"}, {"type": "p", "content": "Paragraph 2"}]}
        snippet = proc.get_text_snippet(document)
        self.assertEqual(snippet, "Paragraph 1 Paragraph 2")

    @patch('experiments.ingestion.analyzer.requests.post')
    def test_dynamic_metadata_agent_success(self, mock_post):
        mock_schema_response = MagicMock()
        mock_schema_response.json.return_value = {
            "choices": [{"message": {"content": '{"metadata_fields": ["tone", "audience_expertise"]}'}}]
        }
        mock_extract_response = MagicMock()
        mock_extract_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '''{
                        "tone": {"value": "Formal", "strength": 0.9, "confidence": 0.9},
                        "audience_expertise": {"value": "Advanced", "strength": 0.8, "confidence": 0.8}
                    }'''
                }
            }]
        }
        mock_post.side_effect = [mock_schema_response, mock_extract_response]
        
        proc = DynamicMetadataAgent(self.api_key)
        result = proc.process(self.document)
        self.assertEqual(result["tone"]["value"], "Formal")
        self.assertEqual(result["tone"]["strength"], 0.9)

    @patch('experiments.ingestion.analyzer.requests.post')
    def test_entity_linking_agent_success(self, mock_post):
        mock_type_response = MagicMock()
        mock_type_response.json.return_value = {
            "choices": [{"message": {"content": '{"entity_types": ["Technology"]}'}}]
        }
        mock_extract_response = MagicMock()
        mock_extract_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"entities": [{"value": "AI", "type": "Technology", "strength": 0.9, "confidence": 0.9}]}'
                }
            }]
        }
        mock_post.side_effect = [mock_type_response, mock_extract_response]
        
        proc = EntityLinkingAgent(self.api_key)
        result = proc.process(self.document)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["value"], "AI")

    def test_sentiment_agent_success(self):
        proc = SentimentAgent(self.api_key)
        proc.sentiment_pipe = MagicMock()
        proc.sentiment_pipe.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
        
        result = proc.process(self.document)
        self.assertEqual(result["sentiment"], "Positive")
        self.assertEqual(result["strength"], 0.95)

    def test_classification_agent_success(self):
        proc = ClassificationAgent(self.api_key)
        proc.classifier = MagicMock()
        # Side effect to return different results for domain and format calls
        proc.classifier.side_effect = [
            {'labels': ['News'], 'scores': [0.9]},
            {'labels': ['Article'], 'scores': [0.8]}
        ]
        
        result = proc.process(self.document)
        self.assertEqual(result["domain"], "News")

    def test_openie_agent_success(self):
        proc = OpenIEAgent(self.api_key)
        mock_nlp = MagicMock()
        mock_token = MagicMock()
        mock_token.dep_ = "subj"
        mock_token.text = "AI"
        mock_token.pos_ = "NOUN"
        
        mock_verb = MagicMock()
        mock_verb.dep_ = "ROOT"
        mock_verb.text = "is"
        mock_verb.pos_ = "VERB"
        
        mock_obj = MagicMock()
        mock_obj.dep_ = "obj"
        mock_obj.text = "awesome"
        mock_obj.pos_ = "ADJ"
        
        mock_sent = [mock_token, mock_verb, mock_obj]
        
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc
        proc.nlp = mock_nlp
        
        result = proc.process(self.document)
        self.assertEqual(len(result["triples"]), 1)
        self.assertEqual(result["triples"][0]["subject"], "AI")
        self.assertEqual(result["triples"][0]["predicate"], "is")

class TestDocumentAnalyzer(unittest.TestCase):
    """Test suite for DocumentAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = DocumentAnalyzer(api_key="test-key")

    def test_discretize_value(self):
        label, stv = self.analyzer.discretize_value(300, LENGTH_BUCKETS)
        self.assertEqual(label, "Short")
        self.assertEqual(stv, (0.1, 0.3))

    def test_calculate_proportional_stv(self):
        stv = self.analyzer.calculate_proportional_stv(250, "Short", (0.1, 0.3), bucket_bounds=(0, 500))
        self.assertAlmostEqual(stv[0], 0.2, places=1)

    def test_process_document(self):
        document = {
            "id": 1,
            "post_title": "Test Document",
            "content": "This is content",
            "min_to_read": "5 min",
            "published_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "views": 100, "likes": 50, "comments": 10,
            "author_username": "user",
            "categories": [{"slug": "tech"}]
        }
        
        with patch.object(self.analyzer.classifier, 'process', return_value={"domain": "News", "confidence": 0.9}):
            with patch.object(self.analyzer.dynamic_metadata, 'process', return_value={
                "tone": {"value": "Formal", "strength": 0.8, "confidence": 0.9},
                "audience_expertise": {"value": "Intermediate", "strength": 0.7, "confidence": 0.8}
            }):
                with patch.object(self.analyzer.sentiment, 'process', return_value={"sentiment": "Positive", "strength": 0.8, "confidence": 0.9}):
                    with patch.object(self.analyzer.entities, 'process', return_value={"entities": [{"value": "Metta"}]}):
                        with patch.object(self.analyzer.openie, 'process', return_value={"triples": []}):
                            
                            # Additional patch to prevent __init__ of DocumentAnalyzer from downloading models
                            # actually it's already instantiated in setUp. Let's make sure setUp is mocked or it will download.
                            result = self.analyzer.process(document)
                            self.assertIn("enriched_metadata", result)
                            self.assertEqual(result["enriched_metadata"]["tone"]["value"], "Formal")
                            self.assertEqual(result["enriched_metadata"]["domain"]["value"], "News")
                            self.assertEqual(len(result["enriched_metadata"]["entities"]), 1)

if __name__ == "__main__":
    unittest.main()
