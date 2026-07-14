import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
from experiments.ingestion.pipeline import run_ingestion


def enriched_with_engagement(article, **_kwargs):
    return {
        **article,
        "enriched_metadata": {
            "engagement": {"value": "Low", "stv": (0.2, 0.9)}
        },
    }


class TestIngestionPipeline(unittest.TestCase):
    """Test suite for ingestion pipeline orchestration"""

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_run_ingestion_success(self, mock_makedirs, mock_exists, mock_file, mock_load_env, 
                                   mock_converter_class, mock_analyzer_class, mock_fetcher_class):
        """Test successful ingestion pipeline execution"""
        # Setup mocks
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": 1,
                "post_title": "Article 1",
                "views": 100,
                "content": [{"type": "p", "content": "test"}],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author1",
                "categories": [{"slug": "tech"}]
            },
            {
                "id": 2,
                "post_title": "Article 2",
                "views": 50,
                "content": [{"type": "p", "content": "test"}],
                "min_to_read": "3 min",
                "published_timestamp": "2024-01-02 10:00:00",
                "likes": 5,
                "comments": 2,
                "author_username": "author2",
                "categories": [{"slug": "tech"}]
            }
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_converter = MagicMock()
        mock_converter.convert.return_value = "(test metta output)"
        mock_converter_class.return_value = mock_converter
        
        mock_exists.return_value = True
        
        # Run pipeline
        result = run_ingestion(username="test_user")
        
        # Verify pipeline executed
        self.assertEqual(result["status"], "success")
        self.assertIn("2", result["message"])
        
        # Verify fetcher was created and used
        mock_fetcher_class.assert_called_once_with(username="test_user")
        mock_fetcher.fetch_all.assert_called_once()
        
        # Verify analyzer was created
        mock_analyzer_class.assert_called_once()
        
        # Verify converter was created and used
        mock_converter_class.assert_called_once()
        mock_converter.convert.assert_called_once()
        
        # Verify file was written
        mock_file.assert_called()

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    def test_run_ingestion_no_articles_found(self, mock_load_env, mock_fetcher_class):
        """Test pipeline handling when no articles are found"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = []
        mock_fetcher_class.return_value = mock_fetcher
        
        result = run_ingestion(username="test_user")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("No articles found", result["message"])

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    def test_run_ingestion_uses_env_username(self, mock_load_env, mock_fetcher_class):
        """Test that pipeline uses environment username if not provided"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = []
        mock_fetcher_class.return_value = mock_fetcher
        
        with patch.dict(os.environ, {"MINDPLEX_USERNAME": "env_user"}):
            run_ingestion()
            
        # Should use environment username
        mock_fetcher_class.assert_called_once_with(username="env_user")

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    def test_run_ingestion_default_username(self, mock_load_env, mock_fetcher_class):
        """Test that pipeline uses default username if not in environment"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = []
        mock_fetcher_class.return_value = mock_fetcher
        
        with patch.dict(os.environ, {}, clear=True):
            run_ingestion()
            
        # Should use default username
        from experiments.ingestion.config import DEFAULT_USERNAME
        mock_fetcher_class.assert_called_once_with(username=DEFAULT_USERNAME)

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    def test_run_ingestion_validates_article_views(self, mock_load_env, mock_analyzer_class, mock_fetcher_class):
        """Test that pipeline validates and casts view counts"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": 1,
                "post_title": "Test",
                "views": "not_a_number",  # Invalid
                "content": [],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author",
                "categories": []
            }
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        with patch('experiments.ingestion.pipeline.JsonToMetta'):
            with patch('builtins.open', new_callable=mock_open):
                with patch('os.makedirs'):
                    result = run_ingestion()
        
        # Pipeline should handle invalid view counts
        self.assertEqual(result["status"], "success")
        
        # Check that views were validated
        processed_article = mock_analyzer.process.call_args[0][0]
        self.assertEqual(processed_article["views"], 0)  # Should default to 0

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_run_ingestion_rank_stats_passed_to_analyzer(self, mock_makedirs, mock_file, 
                                                         mock_load_env, mock_converter_class,
                                                         mock_analyzer_class, mock_fetcher_class):
        """Test that ranking stats are correctly passed to analyzer"""
        mock_articles = [
            {
                "id": 1,
                "post_title": "Article 1",
                "views": 100,
                "content": [{"type": "p", "content": "test"}],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author1",
                "categories": [{"slug": "tech"}]
            },
            {
                "id": 2,
                "post_title": "Article 2",
                "views": 50,
                "content": [{"type": "p", "content": "test"}],
                "min_to_read": "3 min",
                "published_timestamp": "2024-01-02 10:00:00",
                "likes": 5,
                "comments": 2,
                "author_username": "author2",
                "categories": [{"slug": "tech"}]
            }
        ]
        
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = mock_articles
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_converter = MagicMock()
        mock_converter.convert.return_value = "(test)"
        mock_converter_class.return_value = mock_converter
        
        run_ingestion()
        
        # Verify rank_stats parameter was passed
        # Article 1 has higher views (100) so rank 1, Article 2 has rank 2
        calls = mock_analyzer.process.call_args_list
        self.assertEqual(len(calls), 2)
        
        # First article should have rank 1
        self.assertEqual(calls[0][1]["rank_stats"][1], 1)
        # Second article should have rank 2
        self.assertEqual(calls[1][1]["rank_stats"][2], 2)

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    def test_run_ingestion_creates_output_directory(self, mock_load_env, mock_fetcher_class):
        """Test that pipeline creates output directory if it doesn't exist"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": 1,
                "post_title": "Test",
                "views": 100,
                "content": [],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author",
                "categories": []
            }
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        with patch('experiments.ingestion.pipeline.ArticleAnalyzer') as mock_analyzer_class:
            mock_analyzer_class.return_value.process.side_effect = enriched_with_engagement
            with patch('experiments.ingestion.pipeline.JsonToMetta'):
                with patch('builtins.open', new_callable=mock_open):
                    with patch('os.makedirs') as mock_makedirs:
                        run_ingestion()
                        
                        # Verify makedirs was called for the output directory
                        mock_makedirs.assert_called()

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_run_ingestion_limit_respected(self, mock_makedirs, mock_file, mock_load_env,
                                           mock_converter_class, mock_analyzer_class, mock_fetcher_class):
        """Test that pipeline respects the fetch limit"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": i,
                "post_title": f"Article {i}",
                "views": 100 - i,
                "content": [{"type": "p", "content": "test"}],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author",
                "categories": []
            }
            for i in range(1, 51)
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_converter = MagicMock()
        mock_converter.convert.return_value = "(test)"
        mock_converter_class.return_value = mock_converter
        
        run_ingestion()
        
        # Pipeline should fetch up to limit
        mock_fetcher.fetch_all.assert_called_once_with(limit=50)

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_run_ingestion_uses_asi_api_key(self, mock_makedirs, mock_file, mock_load_env,
                                            mock_converter_class, mock_analyzer_class, mock_fetcher_class):
        """Test that pipeline passes ASI API key to analyzer"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": 1,
                "post_title": "Test",
                "views": 100,
                "content": [],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author",
                "categories": []
            }
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        with patch.dict(os.environ, {"ASI_API_KEY": "test-api-key"}):
            run_ingestion()
        
        # Verify analyzer was created with API key
        mock_analyzer_class.assert_called_once_with(api_key="test-api-key")

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_run_ingestion_creates_metta_file(self, mock_makedirs, mock_file, mock_load_env,
                                              mock_converter_class, mock_analyzer_class, mock_fetcher_class):
        """Test that pipeline writes MeTTa output to file"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "id": 1,
                "post_title": "Test",
                "views": 100,
                "content": [],
                "min_to_read": "5 min",
                "published_timestamp": "2024-01-01 10:00:00",
                "likes": 10,
                "comments": 5,
                "author_username": "author",
                "categories": []
            }
        ]
        mock_fetcher_class.return_value = mock_fetcher
        
        mock_analyzer = MagicMock()
        mock_analyzer.process.side_effect = enriched_with_engagement
        mock_analyzer_class.return_value = mock_analyzer
        
        mock_converter = MagicMock()
        metta_output = "(test metta facts)"
        mock_converter.convert.return_value = metta_output
        mock_converter_class.return_value = mock_converter
        
        mock_file_handle = MagicMock()
        
        with patch('builtins.open', mock_open()) as mock_open_call:
            with patch('os.makedirs'):
                run_ingestion()
        
        # Verify file was opened for writing
        mock_open_call.assert_called()

    @patch('experiments.ingestion.pipeline.MindplexFetcher')
    @patch('experiments.ingestion.pipeline.ArticleAnalyzer')
    @patch('experiments.ingestion.pipeline.JsonToMetta')
    @patch('experiments.ingestion.pipeline.load_dotenv')
    @patch('builtins.open', new_callable=mock_open)
    def test_missing_engagement_preserves_existing_dataset(
        self,
        mock_file,
        mock_load_env,
        mock_converter_class,
        mock_analyzer_class,
        mock_fetcher_class,
    ):
        mock_fetcher_class.return_value.fetch_all.return_value = [
            {"id": 1, "post_title": "No target", "views": 1}
        ]
        mock_analyzer_class.return_value.process.return_value = {
            "id": 1,
            "enriched_metadata": {},
        }

        result = run_ingestion(username="test_user")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["code"], "engagement_required")
        self.assertEqual(result["missing_engagement_records"], 1)
        mock_converter_class.return_value.convert.assert_not_called()
        mock_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
