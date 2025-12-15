# experiments/ingestion/config.py

# API Configuration
MINDPLEX_API_DOMAIN = "https://console.mindplex.ai/wp-json"
# Endpoint pattern: /mp_gl/v1/posts/social/{username}/{page}
USER_ARTICLES_ENDPOINT_TEMPLATE = "/mp_gl/v1/posts/publisher/{username}/{page}"

DEFAULT_USERNAME = "ben_g"

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
    "Very_High": lambda x: x > 0.35,
    "High": lambda x: 0.20 <= x <= 0.35,
    "Medium": lambda x: 0.10 <= x < 0.20,
    "Low": lambda x: x < 0.10
}

RETENTION_BUCKETS = {
    # Completion rate (if available)
    "High_Completion": lambda x: x > 0.80,
    "Moderate_Completion": lambda x: 0.50 <= x <= 0.80,
    "Low_Completion": lambda x: x < 0.50
}

# AI Prompts
ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following article content and metadata to provide categorical classifications.
Return ONLY a JSON object with the following keys: "tone", "complexity", "content_type", "primary_goal", "audience_sentiment".

Article Title: {title}
Content Snippet: {content_snippet}

Classifications to use:
- Tone: Formal, Casual, Instructional, Satirical, Reflective, Motivational
- Complexity: Beginner, Intermediate, Advanced, Expert
- Content Type: Listicle, Tutorial, Opinion, Case Study, News, Review, Interview
- Primary Goal: Inform, Persuade, Entertain, Generate_Leads, Sell_Product
- Audience Sentiment (predicted based on content tone): Positive, Neutral, Negative, Mixed
"""
