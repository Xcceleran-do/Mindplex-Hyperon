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

    def fetch_page(self, page=1):
        """Fetches a single page of articles for the user."""
        path = USER_ARTICLES_ENDPOINT_TEMPLATE.format(username=self.username, page=page)
        url = f"{MINDPLEX_API_DOMAIN}{path}"
        
        try:
            print(f"Fetching {url}...")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            return None

    def fetch_all(self, limit=100):
        """Fetches articles up to a limit."""
        articles = []
        page = 1
        
        while len(articles) < limit:
            data = self.fetch_page(page)
            if not data:
                break
            
            # The API returns { "published_posts": [...] }
            batch = data.get('published_posts', [])
            
            if not batch:
                print("No more posts found.")
                break
                
            articles.extend(batch)
            
            # Check if we've reached the end (if batch size is small, likely last page)
            # Or we can check 'count' in response if available/reliable
            if len(batch) < 10: # Assuming default page size is around 10-20
                break
                
            page += 1
            
        return articles[:limit]


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
