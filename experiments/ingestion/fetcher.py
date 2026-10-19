# experiments/ingestion/fetcher.py
import requests
import os
import json
from .config import (
    MINDPLEX_API_DOMAIN as DEFAULT_MINDPLEX_API_DOMAIN,
    MINDPLEX_AUTH_LOGIN_ENDPOINT,
    MINDPLEX_TOKEN_REFRESH_ENDPOINT,
    USER_ARTICLES_ENDPOINT_TEMPLATE,
    DEFAULT_HEADERS,
    DEFAULT_USERNAME,
)

DEFAULT_TOKEN_CACHE_PATH = ""

class MindplexFetcher:
    def __init__(self, username=DEFAULT_USERNAME):
        self.token = os.getenv("MINDPLEX_API_TOKEN")
        self.refresh_token = os.getenv("MINDPLEX_API_REFRESH_TOKEN")
        self.login_email = os.getenv("MINDPLEX_SERVICE_EMAIL")
        self.login_password = os.getenv("MINDPLEX_SERVICE_PASSWORD")
        self.token_cache_path = os.getenv("MINDPLEX_TOKEN_CACHE_PATH", DEFAULT_TOKEN_CACHE_PATH)
        self.username = username
        self.headers = DEFAULT_HEADERS.copy()
        self._load_cached_tokens()
        self._set_access_token(self.token)

    def _load_cached_tokens(self):
        if not self.token_cache_path or not os.path.exists(self.token_cache_path):
            return

        try:
            with open(self.token_cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not read Mindplex token cache: {exc}")
            return

        cached_access = data.get("access_token")
        cached_refresh = data.get("refresh_token")
        if cached_access:
            self.token = cached_access
        if cached_refresh:
            self.refresh_token = cached_refresh

    def _save_cached_tokens(self):
        if not self.token_cache_path:
            return

        payload = {}
        if self.token:
            payload["access_token"] = self.token
        if self.refresh_token:
            payload["refresh_token"] = self.refresh_token
        if not payload:
            return

        try:
            os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
            tmp_path = f"{self.token_cache_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_path, self.token_cache_path)
            try:
                os.chmod(self.token_cache_path, 0o600)
            except OSError:
                pass
        except OSError as exc:
            print(f"Could not write Mindplex token cache: {exc}")

    def _set_access_token(self, token):
        self.token = token
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        else:
            self.headers.pop("Authorization", None)

    def _api_domain(self):
        return os.getenv("MINDPLEX_API_DOMAIN", DEFAULT_MINDPLEX_API_DOMAIN)

    def _refresh_url(self):
        return f"{self._api_domain().rstrip('/')}/{MINDPLEX_TOKEN_REFRESH_ENDPOINT.lstrip('/')}"

    def _login_url(self):
        return f"{self._api_domain().rstrip('/')}/{MINDPLEX_AUTH_LOGIN_ENDPOINT.lstrip('/')}"

    def _apply_token_response(self, data):
        token_data = data.get("data") if isinstance(data.get("data"), dict) else data
        access_token = (
            token_data.get("access_token")
            or token_data.get("accessToken")
            or token_data.get("token")
            or token_data.get("access")
        )
        refresh_token = token_data.get("refresh_token") or token_data.get("refreshToken")

        if not access_token:
            print("Mindplex auth response did not include an access token.")
            return False

        self._set_access_token(access_token)
        os.environ["MINDPLEX_API_TOKEN"] = access_token

        if refresh_token:
            self.refresh_token = refresh_token
            os.environ["MINDPLEX_API_REFRESH_TOKEN"] = refresh_token

        self._save_cached_tokens()
        return True

    def _post_auth(self, url, payload):
        try:
            response = requests.post(
                url,
                headers=DEFAULT_HEADERS.copy(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            print(f"Error requesting Mindplex auth token: {exc}")
            return None

    def _refresh_access_token(self):
        if not self.refresh_token:
            return False

        data = self._post_auth(self._refresh_url(), {"refreshToken": self.refresh_token})
        if not data:
            return False

        return self._apply_token_response(data)

    def _login_access_token(self):
        if not self.login_email or not self.login_password:
            return False

        data = self._post_auth(
            self._login_url(),
            {
                "email": self.login_email,
                "password": self.login_password,
            },
        )
        if not data:
            return False

        return self._apply_token_response(data)

    def _refresh_or_login_access_token(self):
        return self._refresh_access_token() or self._login_access_token()

    def has_auth_material(self):
        return bool(self.token or self.refresh_token or (self.login_email and self.login_password))

    def ensure_authenticated(self):
        if self.token:
            return True
        if not self.has_auth_material():
            return False
        return self._refresh_or_login_access_token()

    def auth_status(self):
        return {
            "access_token_loaded": bool(self.token),
            "refresh_token_loaded": bool(self.refresh_token),
            "service_login_configured": bool(self.login_email and self.login_password),
            "token_cache_enabled": bool(self.token_cache_path),
            "token_cache_path": self.token_cache_path or None,
            "api_base_url": self._api_domain(),
            "login_url": self._login_url(),
            "refresh_url": self._refresh_url(),
        }

    def _get(self, url, params):
        self.ensure_authenticated()
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        if response.status_code == 401 and self._refresh_or_login_access_token():
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        return response

    def fetch_page(self, page=1, page_size=50):
        """Fetches a single page of articles for the user."""
        path = USER_ARTICLES_ENDPOINT_TEMPLATE.format(username=self.username, page=page)
        url = f"{self._api_domain().rstrip('/')}{path}"
        params = {
            "page": page,
            "limit": max(1, min(int(page_size or 50), 100)),
            "include": "author,stats",
        }
        
        try:
            print(f"Fetching {url} (page={page}, limit={params['limit']})...")
            response = self._get(url, params)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            return None

    def fetch_all(self, limit=100):
        """Fetches articles up to a limit."""
        articles = []
        page = 1
        page_size = max(1, min(int(limit or 1), 20))
        
        while len(articles) < limit:
            data = self.fetch_page(page, page_size=page_size)
            if not data:
                break
            
            batch = extract_mindplex_records(data)
            
            if not batch:
                print("No more posts found.")
                break
                
            articles.extend(normalize_mindplex_record(record, self.username) for record in batch)
            
            total = data.get("total") if isinstance(data, dict) else None
            if isinstance(total, int) and len(articles) >= total:
                break
            if len(batch) < page_size:
                break
                
            page += 1
            
        return articles[:limit]


def extract_mindplex_records(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("published_posts", "data", "items", "results", "records"):
        records = data.get(key)
        if isinstance(records, list):
            return records
    return []


def normalize_mindplex_record(record, fallback_username=None):
    if not isinstance(record, dict):
        return record

    normalized = dict(record)
    stats = normalized.get("stats") if isinstance(normalized.get("stats"), dict) else {}
    author = normalized.get("author") if isinstance(normalized.get("author"), dict) else {}

    normalized.setdefault("post_title", normalized.get("title", "Untitled"))
    normalized.setdefault("views", normalized.get("viewCount", 0))
    normalized.setdefault("likes", stats.get("likeCount", 0))
    normalized.setdefault("comments", stats.get("commentCount", 0))
    normalized.setdefault(
        "shares", normalized.get("shareCount", stats.get("shareCount", 0))
    )
    normalized.setdefault("reactions", stats.get("reactionCount", 0))
    normalized.setdefault("min_to_read", normalized.get("estimatedReadingMinutes", 0))
    normalized.setdefault("published_timestamp", format_mindplex_timestamp(normalized.get("publishedAt")))
    normalized.setdefault("author_username", author.get("username") or fallback_username or "unknown")
    # Send one canonical copy of each source field to the remote planner. Keeping
    # both API aliases and normalized counters would double-count engagement.
    normalized.pop("stats", None)
    for alias in (
        "viewCount",
        "shareCount",
        "estimatedReadingMinutes",
        "publishedAt",
    ):
        normalized.pop(alias, None)
    return normalized


def format_mindplex_timestamp(value):
    if not value:
        return value
    text = str(value).replace("T", " ").replace("Z", "")
    return text[:19]
