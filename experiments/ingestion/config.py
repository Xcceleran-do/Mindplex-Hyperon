# experiments/ingestion/config.py

# API Configuration
MINDPLEX_API_DOMAIN = "https://staging.mindplex.ai/wp-json"
# Endpoint pattern: /mp_gl/v1/posts/social/{username}/{page}
USER_ARTICLES_ENDPOINT_TEMPLATE = "/mp_gl/v1/posts/publisher/{username}/{page}"

DEFAULT_USERNAME = "Ben_G"

DEFAULT_HEADERS = {
    "User-Agent": "MindplexMiner/1.0",
    "Accept": "application/json"
}

# Discretization Buckets
LENGTH_BUCKETS = {
    "Short": lambda x: x < 500,
    "Medium": lambda x: 500 <= x <= 1500,
    "Long": lambda x: x > 1500
}

READING_TIME_BUCKETS = {
    "Very_Short": lambda x: x < 2,
    "Short": lambda x: 2 <= x < 5,
    "Medium": lambda x: 5 <= x <= 10,
    "Long": lambda x: x > 10
}

ENGAGEMENT_BUCKETS = {
    # Ratio of (Likes + Claps) / Views
    # Note: If views are 0, this metric might be misleading or undefined.
    "Very_High": lambda x: x > 500,
    "High": lambda x: 50 <= x <= 100,
    "Medium": lambda x: 30 <= x < 50,
    "Low": lambda x: x < 30
}

# AI Prompts
ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following article content and metadata to provide categorical classifications.
Return ONLY a JSON object with the following keys: "tone", "audience_expertise", "content_type", "primary_goal", "audience_sentiment".

Article Title: {title}
Content Snippet: {content_snippet}

Classifications to use:
- Tone: Formal, Casual, Instructional
- Audience Expertise: Beginner, Intermediate, Advanced, Expert
- Content Type: Tutorial, Opinion, Review, Interview
- Primary Goal: Inform, Persuade, Entertain
- Audience Sentiment (predicted based on content tone): Positive, Neutral, Negative, Mixed
"""
