from ..fetcher import MindplexFetcher

class JsonApiConnector:

    def __init__(self, config):
        from ..config import DEFAULT_USERNAME
        self.username = config.get("username", DEFAULT_USERNAME)

    def fetch(self):
        fetcher = MindplexFetcher(self.username)
        articles = fetcher.fetch_all(limit=100)
        return articles