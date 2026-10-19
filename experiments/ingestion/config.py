import os


MINDPLEX_API_DOMAIN = os.getenv("MINDPLEX_API_DOMAIN", "https://api-dev.mindplex.ai")
MINDPLEX_TOKEN_REFRESH_ENDPOINT = "/v1/auth/refresh"
MINDPLEX_AUTH_LOGIN_ENDPOINT = "/v1/auth/login"
USER_ARTICLES_ENDPOINT_TEMPLATE = "/v1/users/{username}/posts"
DEFAULT_USERNAME = "mod_alex"
DEFAULT_HEADERS = {
    "User-Agent": "MindplexMiner/1.0",
    "Accept": "application/json",
}
