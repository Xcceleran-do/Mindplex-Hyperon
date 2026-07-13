import unittest
from unittest.mock import patch, MagicMock
import os
import requests
import json
import tempfile
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

    def test_init_accepts_service_account_credentials(self):
        """Test dedicated service-account env names for backend sessions."""
        with patch.dict(
            os.environ,
            {
                "MINDPLEX_SERVICE_EMAIL": "service@example.com",
                "MINDPLEX_SERVICE_PASSWORD": "service-secret",
            },
            clear=True,
        ):
            fetcher = MindplexFetcher(username="test_user")

        self.assertEqual(fetcher.login_email, "service@example.com")
        self.assertEqual(fetcher.login_password, "service-secret")
        self.assertTrue(fetcher.auth_status()["service_login_configured"])

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

    def test_init_prefers_cached_tokens_over_env_tokens(self):
        """Test that rotated cached tokens survive stale .env values."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            json.dump({"access_token": "cached-access", "refresh_token": "cached-refresh"}, handle)
            cache_path = handle.name

        try:
            with patch.dict(
                os.environ,
                {
                    "MINDPLEX_API_TOKEN": "env-access",
                    "MINDPLEX_API_REFRESH_TOKEN": "env-refresh",
                    "MINDPLEX_TOKEN_CACHE_PATH": cache_path,
                },
                clear=True,
            ):
                fetcher = MindplexFetcher(username="test_user")
        finally:
            os.remove(cache_path)

        self.assertEqual(fetcher.token, "cached-access")
        self.assertEqual(fetcher.refresh_token, "cached-refresh")
        self.assertEqual(fetcher.headers["Authorization"], "Bearer cached-access")

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

    @patch.object(MindplexFetcher, '_refresh_url', return_value='http://auth.local/refresh')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_refreshes_access_token_after_401(self, mock_get, mock_post, _mock_refresh_url):
        """Test that a 401 page request refreshes the token and retries once."""
        expired_response = MagicMock()
        expired_response.status_code = 401

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"published_posts": [{"id": 1, "post_title": "Article 1"}]}

        refresh_response = MagicMock()
        refresh_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
        }

        mock_get.side_effect = [expired_response, success_response]
        mock_post.return_value = refresh_response

        with patch.dict(
            os.environ,
            {
                "MINDPLEX_API_TOKEN": "expired-access-token",
                "MINDPLEX_API_REFRESH_TOKEN": "persistent-refresh-token",
                "MINDPLEX_SERVICE_EMAIL": "",
                "MINDPLEX_SERVICE_PASSWORD": "",
                "MINDPLEX_TOKEN_CACHE_PATH": "",
            },
        ):
            fetcher = MindplexFetcher(username="test_user")
            result = fetcher.fetch_page(page=1)

        self.assertEqual(result["published_posts"][0]["id"], 1)
        self.assertEqual(fetcher.token, "new-access-token")
        self.assertEqual(fetcher.refresh_token, "new-refresh-token")
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_post.call_args[0][0], "http://auth.local/refresh")
        self.assertEqual(mock_post.call_args[1]["json"], {"refreshToken": "persistent-refresh-token"})
        self.assertEqual(mock_get.call_args_list[1][1]["headers"]["Authorization"], "Bearer new-access-token")

    @patch.object(MindplexFetcher, '_refresh_url', return_value='http://auth.local/refresh')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_persists_rotated_tokens_after_refresh(self, mock_get, mock_post, _mock_refresh_url):
        """Test that rotated refresh tokens are written to the configured cache."""
        expired_response = MagicMock()
        expired_response.status_code = 401

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"published_posts": [{"id": 1, "post_title": "Article 1"}]}

        refresh_response = MagicMock()
        refresh_response.json.return_value = {
            "data": {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
            },
        }

        mock_get.side_effect = [expired_response, success_response]
        mock_post.return_value = refresh_response

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "tokens.json")
            with patch.dict(
                os.environ,
                {
                    "MINDPLEX_API_TOKEN": "expired-access-token",
                    "MINDPLEX_API_REFRESH_TOKEN": "persistent-refresh-token",
                    "MINDPLEX_TOKEN_CACHE_PATH": cache_path,
                },
                clear=True,
            ):
                fetcher = MindplexFetcher(username="test_user")
                result = fetcher.fetch_page(page=1)

            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)

        self.assertEqual(result["published_posts"][0]["id"], 1)
        self.assertEqual(cached["access_token"], "new-access-token")
        self.assertEqual(cached["refresh_token"], "new-refresh-token")

    @patch.object(MindplexFetcher, '_refresh_url', return_value='http://auth.local/refresh')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_returns_none_when_refresh_fails(self, mock_get, mock_post, _mock_refresh_url):
        """Test that a failed refresh keeps fetch_page graceful for ingestion."""
        expired_response = MagicMock()
        expired_response.status_code = 401
        expired_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")

        refresh_response = MagicMock()
        refresh_response.raise_for_status.side_effect = requests.exceptions.HTTPError("refresh failed")

        mock_get.return_value = expired_response
        mock_post.return_value = refresh_response

        with patch.dict(
            os.environ,
            {
                "MINDPLEX_API_TOKEN": "expired-access-token",
                "MINDPLEX_API_REFRESH_TOKEN": "persistent-refresh-token",
            },
        ):
            fetcher = MindplexFetcher(username="test_user")
            result = fetcher.fetch_page(page=1)

        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_post.call_count, 1)

    @patch.object(MindplexFetcher, '_login_url', return_value='http://auth.local/login')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_logs_in_after_401_when_refresh_token_missing(self, mock_get, mock_post, _mock_login_url):
        """Test that credentials can mint tokens when no refresh token is configured."""
        expired_response = MagicMock()
        expired_response.status_code = 401

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"published_posts": [{"id": 1, "post_title": "Article 1"}]}

        login_response = MagicMock()
        login_response.json.return_value = {
            "data": {
                "access_token": "login-access-token",
                "refresh_token": "login-refresh-token",
            },
        }

        mock_get.side_effect = [expired_response, success_response]
        mock_post.return_value = login_response

        with patch.dict(
            os.environ,
            {
                "MINDPLEX_API_TOKEN": "expired-access-token",
                "MINDPLEX_SERVICE_EMAIL": "user@example.com",
                "MINDPLEX_SERVICE_PASSWORD": "secret-password",
            },
            clear=True,
        ):
            fetcher = MindplexFetcher(username="test_user")
            result = fetcher.fetch_page(page=1)

        self.assertEqual(result["published_posts"][0]["id"], 1)
        self.assertEqual(fetcher.token, "login-access-token")
        self.assertEqual(fetcher.refresh_token, "login-refresh-token")
        self.assertEqual(mock_post.call_args[0][0], "http://auth.local/login")
        self.assertEqual(
            mock_post.call_args[1]["json"],
            {"email": "user@example.com", "password": "secret-password"},
        )
        self.assertEqual(mock_get.call_args_list[1][1]["headers"]["Authorization"], "Bearer login-access-token")

    @patch.object(MindplexFetcher, '_login_url', return_value='http://auth.local/login')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_logs_in_before_first_request_when_token_missing(self, mock_get, mock_post, _mock_login_url):
        """Test that service credentials mint a token before the first API request."""
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"published_posts": [{"id": 1, "post_title": "Article 1"}]}

        login_response = MagicMock()
        login_response.json.return_value = {
            "accessToken": "login-access-token",
            "refreshToken": "login-refresh-token",
        }

        mock_get.return_value = success_response
        mock_post.return_value = login_response

        with patch.dict(
            os.environ,
            {
                "MINDPLEX_SERVICE_EMAIL": "service@example.com",
                "MINDPLEX_SERVICE_PASSWORD": "service-secret",
            },
            clear=True,
        ):
            fetcher = MindplexFetcher(username="test_user")
            result = fetcher.fetch_page(page=1)

        self.assertEqual(result["published_posts"][0]["id"], 1)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_post.call_args[0][0], "http://auth.local/login")
        self.assertEqual(
            mock_post.call_args[1]["json"],
            {"email": "service@example.com", "password": "service-secret"},
        )
        self.assertEqual(mock_get.call_args[1]["headers"]["Authorization"], "Bearer login-access-token")

    @patch.object(MindplexFetcher, '_refresh_url', return_value='http://auth.local/refresh')
    @patch.object(MindplexFetcher, '_login_url', return_value='http://auth.local/login')
    @patch('experiments.ingestion.fetcher.requests.post')
    @patch('experiments.ingestion.fetcher.requests.get')
    def test_fetch_page_logs_in_after_refresh_forbidden(self, mock_get, mock_post, _mock_login_url, _mock_refresh_url):
        """Test that credentials recover from a revoked/forbidden refresh token."""
        expired_response = MagicMock()
        expired_response.status_code = 401

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"published_posts": [{"id": 1, "post_title": "Article 1"}]}

        forbidden_refresh = MagicMock()
        forbidden_refresh.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")

        login_response = MagicMock()
        login_response.json.return_value = {
            "data": {
                "access_token": "login-access-token",
                "refresh_token": "login-refresh-token",
            },
        }

        mock_get.side_effect = [expired_response, success_response]
        mock_post.side_effect = [forbidden_refresh, login_response]

        with patch.dict(
            os.environ,
            {
                "MINDPLEX_API_TOKEN": "expired-access-token",
                "MINDPLEX_API_REFRESH_TOKEN": "revoked-refresh-token",
                "MINDPLEX_SERVICE_EMAIL": "user@example.com",
                "MINDPLEX_SERVICE_PASSWORD": "secret-password",
            },
            clear=True,
        ):
            fetcher = MindplexFetcher(username="test_user")
            result = fetcher.fetch_page(page=1)

        self.assertEqual(result["published_posts"][0]["id"], 1)
        self.assertEqual([call[0][0] for call in mock_post.call_args_list], ["http://auth.local/refresh", "http://auth.local/login"])
        self.assertEqual(fetcher.token, "login-access-token")
        self.assertEqual(fetcher.refresh_token, "login-refresh-token")

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
