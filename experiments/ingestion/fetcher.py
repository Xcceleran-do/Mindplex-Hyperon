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
        """Fetches a single page of documents for the user."""
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
        """Fetches documents up to a limit."""
        documents = []
        page = 1
        
        while len(documents) < limit:
            data = self.fetch_page(page)
            if not data:
                break
            
            # The API returns { "published_posts": [...] }
            batch = data.get('published_posts', [])
            
            if not batch:
                print("No more posts found.")
                break
                
            documents.extend(batch)
            
            # Check if we've reached the end (if batch size is small, likely last page)
            # Or we can check 'count' in response if available/reliable
            if len(batch) < 10: # Assuming default page size is around 10-20
                break
                
            page += 1
            
        return documents[:limit]

class FileFetcher:
    """Fetcher for local files (PDF, CSV, JSON)"""
    def __init__(self, directory):
        self.directory = directory

    def fetch_all(self, limit=50):
        import os
        supported_exts = ('.pdf', '.csv', '.json', '.txt')
        files = []
        if not os.path.exists(self.directory):
            print(f"Directory not found: {self.directory}")
            return []

        # If directory is actually a file
        if os.path.isfile(self.directory):
            files = [self.directory]
        else:
            for f in os.listdir(self.directory):
                if f.lower().endswith(supported_exts):
                    files.append(os.path.join(self.directory, f))

        documents = []
        for i, file_path in enumerate(files[:limit]):
            documents.append({
                "id": f"file_{i}",
                "post_title": os.path.basename(file_path),
                "file_path": file_path,
                "author_username": "local_system"
            })
        return documents
