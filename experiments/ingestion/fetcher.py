# experiments/ingestion/fetcher.py
import requests
import os
from .config import MINDPLEX_API_DOMAIN, USER_ARTICLES_ENDPOINT_TEMPLATE, DEFAULT_HEADERS, DEFAULT_USERNAME

class MindplexFetcher:
    def __init__(self, username=DEFAULT_USERNAME):
        self.token = os.getenv("MINDPLEX_API_TOKEN")
        self.username = username
        self.headers = DEFAULT_HEADERS.copy()
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def fetch_page(self, page=1, page_size=50):
        """Fetches a single page of articles for the user."""
        path = USER_ARTICLES_ENDPOINT_TEMPLATE.format(username=self.username, page=page)
        url = f"{MINDPLEX_API_DOMAIN.rstrip('/')}{path}"
        params = {
            "page": page,
            "limit": max(1, min(int(page_size or 50), 100)),
            "include": "author,stats",
        }
        
        try:
            print(f"Fetching {url}...")
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
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
    normalized.setdefault("min_to_read", normalized.get("estimatedReadingMinutes", 0))
    normalized.setdefault("published_timestamp", format_mindplex_timestamp(normalized.get("publishedAt")))
    normalized.setdefault("author_username", author.get("username") or fallback_username or "unknown")
    return normalized


def format_mindplex_timestamp(value):
    if not value:
        return value
    text = str(value).replace("T", " ").replace("Z", "")
    return text[:19]


class JsonApiFetcher:
    """Generic JSON API fetcher for non-Mindplex ingestion sources.

    Configure with a full ``url`` and optional dotted ``records_path`` pointing
    to the list inside the JSON response. This keeps source access separate
    from extraction planning.
    """

    def __init__(self, url, records_path=None, headers=None):
        self.url = url
        self.records_path = records_path
        self.headers = headers or DEFAULT_HEADERS.copy()

    def fetch_all(self, limit=100):
        response = requests.get(self.url, headers=self.headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        records = read_raw_path(payload, self.records_path) if self.records_path else payload
        if isinstance(records, dict):
            for key in ("items", "results", "data", "records"):
                if isinstance(records.get(key), list):
                    records = records[key]
                    break
        if not isinstance(records, list):
            raise ValueError("Configured JSON API source did not return a record list.")
        return records[:limit]


def build_fetcher(source_name="mindplex", username=None, source_config=None):
    source_config = source_config or {}
    if source_name == "mindplex":
        return MindplexFetcher(username=username or DEFAULT_USERNAME)

    allow_request_url = os.getenv("INGESTION_ALLOW_REQUEST_URLS", "false").lower() == "true"
    request_url = source_config.get("url")
    if request_url and not allow_request_url:
        raise ValueError(
            "Request-provided ingestion URLs are disabled. "
            "Set INGESTION_ALLOW_REQUEST_URLS=true for development, "
            "or configure INGESTION_SOURCE_URL on the server."
        )

    url = request_url or os.getenv("INGESTION_SOURCE_URL")
    if not url:
        raise ValueError("INGESTION_SOURCE_URL is required for non-Mindplex ingestion sources.")
    records_path = source_config.get("records_path") or os.getenv("INGESTION_RECORDS_PATH")
    return JsonApiFetcher(url=url, records_path=records_path)


def read_raw_path(data, path):
    current = data
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current
