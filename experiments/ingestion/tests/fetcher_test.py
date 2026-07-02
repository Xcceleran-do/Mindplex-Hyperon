import unittest
from unittest.mock import patch, MagicMock
import os
from experiments.ingestion.fetcher import MindplexFetcher
from experiments.ingestion.config import DEFAULT_USERNAME


class TestMindplexFetcher(unittest.TestCase):
    """Test suite for MindplexFetcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = MindplexFetcher(username="test_user")

    def test_init_with_token(self):
        """Test fetcher initialization with API token"""
        with patch.dict(os.environ, {"MINDPLEX_API_TOKEN": "test-token-123"}):
            fetcher = MindplexFetcher(username="test_user")
            self.assertEqual(fetcher.token, "test-token-123")
            self.assertEqual(fetcher.username, "test_user")
            self.assertIn("Authorization", fetcher.headers)
            self.assertEqual(fetcher.headers["Authorization"], "Bearer test-token-123")

    def test_init_without_token(self):
        """Test fetcher initialization without API token"""
        with patch.dict(os.environ, {}, clear=True):
            fetcher = MindplexFetcher(username="test_user")
            self.assertIsNone(fetcher.token)
            self.assertNotIn("Authorization", fetcher.headers)

    def test_init_default_username(self):
        """Test fetcher initialization with default username"""
        fetcher = MindplexFetcher()
        self.assertEqual(fetcher.username, DEFAULT_USERNAME)

    def test_fetcher_has_default_headers(self):
        """Test that fetcher includes default headers"""
        self.assertIn("User-Agent", self.fetcher.headers)
        self.assertIn("Accept", self.fetcher.headers)
        self.assertEqual(self.fetcher.headers["User-Agent"], "MindplexMiner/1.0")
        self.assertEqual(self.fetcher.headers["Accept"], "application/json")

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_success(self, mock_get):
        """Test successful page fetch"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "published_posts": [
                {"id": 1, "post_title": "Article 1"},
                {"id": 2, "post_title": "Article 2"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_page(page=1)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["published_posts"]), 2)
        mock_get.assert_called_once()

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_with_correct_url(self, mock_get):
        """Test that fetch_page constructs the correct URL"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"published_posts": []}
        mock_get.return_value = mock_response
        
        self.fetcher.fetch_page(page=2)
        
        # Verify URL was constructed correctly
        called_url = mock_get.call_args[0][0]
        called_params = mock_get.call_args[1]["params"]
        self.assertIn("test_user", called_url)
        self.assertIn("/v1/users/test_user/posts", called_url)
        self.assertEqual(called_params["page"], 2)
        self.assertEqual(called_params["limit"], 50)
        self.assertEqual(called_params["include"], "author,stats")

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_passes_headers(self, mock_get):
        """Test that fetch_page includes headers in request"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"published_posts": []}
        mock_get.return_value = mock_response
        
        self.fetcher.fetch_page(page=1)
        
        # Verify headers were passed
        self.assertEqual(mock_get.call_args[1]["headers"], self.fetcher.headers)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_handles_http_error(self, mock_get):
        """Test handling of HTTP errors"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
        mock_get.return_value = mock_response
        
        with self.assertRaises(Exception):
            self.fetcher.fetch_page(page=1)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_handles_invalid_json(self, mock_get):
        """Test handling of invalid JSON response"""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError):
            self.fetcher.fetch_page(page=1)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_single_page(self, mock_get):
        """Test fetching articles across multiple pages (single page case)"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "published_posts": [
                {"id": 1, "post_title": "Article 1"},
                {"id": 2, "post_title": "Article 2"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_all(limit=10)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 2)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_multiple_pages(self, mock_get):
        """Test fetching articles across multiple pages"""
        page_1_response = MagicMock()
        page_1_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(1, 21)
            ]
        }
        
        page_2_response = MagicMock()
        page_2_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(21, 41)
            ]
        }
        
        page_3_response = MagicMock()
        page_3_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(41, 46)
            ]
        }
        
        mock_get.side_effect = [page_1_response, page_2_response, page_3_response]
        
        result = self.fetcher.fetch_all(limit=100)
        
        # Should have articles from all pages but not exceed limit
        self.assertEqual(len(result), 45)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_respects_limit(self, mock_get):
        """Test that fetch_all respects the limit parameter"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(1, 51)
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_all(limit=25)
        
        # Should not exceed limit
        self.assertEqual(len(result), 25)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_stops_on_empty_batch(self, mock_get):
        """Test that fetch_all stops when receiving empty batch"""
        page_1_response = MagicMock()
        page_1_response.json.return_value = {
            "published_posts": [
                {"id": 1, "post_title": "Article 1"}
            ]
        }
        
        page_2_response = MagicMock()
        page_2_response.json.return_value = {
            "published_posts": []
        }
        
        mock_get.side_effect = [page_1_response, page_2_response]
        
        result = self.fetcher.fetch_all(limit=100)
        
        # Should stop after empty batch at page 2 (not call page 3)
        self.assertEqual(len(result), 1)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_stops_on_small_batch(self, mock_get):
        """Test that fetch_all stops when receiving small final batch"""
        page_1_response = MagicMock()
        page_1_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(1, 21)
            ]
        }
        
        page_2_response = MagicMock()
        page_2_response.json.return_value = {
            "published_posts": [
                {"id": i, "post_title": f"Article {i}"} for i in range(21, 25)
            ]
        }
        
        mock_get.side_effect = [page_1_response, page_2_response]
        
        result = self.fetcher.fetch_all(limit=100)
        
        # Should stop on small batch (< 10)
        self.assertEqual(len(result), 24)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_handles_api_failure(self, mock_get):
        """Test fetch_all handles API failures gracefully"""
        page_1_response = MagicMock()
        page_1_response.json.return_value = {
            "published_posts": [
                {"id": 1, "post_title": "Article 1"}
            ]
        }
        
        mock_get.side_effect = [page_1_response, Exception("API Error")]
        
        result = self.fetcher.fetch_all(limit=100)
        
        # Should return what it could fetch
        self.assertEqual(len(result), 1)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_accepts_paginated_data_key(self, mock_get):
        """Test fetch_all handles the current Mindplex paginated response shape"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": 1, "title": "Article 1", "viewCount": 7, "stats": {"likeCount": 2, "commentCount": 3}}
            ],
            "total": 1,
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_all(limit=10)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["post_title"], "Article 1")
        self.assertEqual(result[0]["views"], 7)
        self.assertEqual(result[0]["likes"], 2)
        self.assertEqual(result[0]["comments"], 3)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_handles_missing_records_key(self, mock_get):
        """Test fetch_all handles response without a recognized records key"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "payload": []
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_all(limit=10)
        
        # Should gracefully handle missing key
        self.assertEqual(len(result), 0)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_all_with_zero_limit(self, mock_get):
        """Test fetch_all with limit of 0"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "published_posts": [
                {"id": 1, "post_title": "Article 1"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = self.fetcher.fetch_all(limit=0)
        
        # Should return empty since limit=0
        self.assertEqual(len(result), 0)

    def test_fetcher_with_custom_username(self):
        """Test fetcher with custom username"""
        custom_username = "custom_user_123"
        fetcher = MindplexFetcher(username=custom_username)
        
        self.assertEqual(fetcher.username, custom_username)

    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_uses_correct_page_number(self, mock_get):
        """Test that correct page numbers are used in pagination"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"published_posts": []}
        mock_get.return_value = mock_response
        
        # Fetch pages 1, 2, and 3
        self.fetcher.fetch_page(page=1)
        self.fetcher.fetch_page(page=2)
        self.fetcher.fetch_page(page=3)
        
        # Verify correct URLs were called
        pages = [call[1]["params"]["page"] for call in mock_get.call_args_list]
        self.assertEqual(pages, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
