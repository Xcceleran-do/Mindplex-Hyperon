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

# Discretization Buckets with Proportional STV Ranges
LENGTH_BUCKETS = {
    "Short": (lambda x: x < 500, (0.1, 0.3)),
    "Medium": (lambda x: 500 <= x <= 1500, (0.4, 0.7)),
    "Long": (lambda x: x > 1500, (0.8, 1.0))
}

READING_TIME_BUCKETS = {
    "Very_Short": (lambda x: x < 2, (0.1, 0.2)),
    "Short": (lambda x: 2 <= x < 5, (0.3, 0.5)),
    "Medium": (lambda x: 5 <= x <= 10, (0.6, 0.8)),
    "Long": (lambda x: x > 10, (0.9, 1.0))
}

ENGAGEMENT_BUCKETS = {
    # Fixed order: Low -> Medium -> High -> Very_High
    "Low": (lambda x: x < 30, (0.1, 0.3)),
    "Medium": (lambda x: 30 <= x < 50, (0.4, 0.6)),
    "High": (lambda x: 50 <= x <= 100, (0.7, 0.8)),
    "Very_High": (lambda x: x > 100, (0.9, 1.0))
}

RETENTION_BUCKETS = {
    "Low_Completion": (lambda x: x < 0.50, (0.1, 0.3)),
    "Moderate_Completion": (lambda x: 0.50 <= x <= 0.80, (0.4, 0.7)),
    "High_Completion": (lambda x: x > 0.80, (0.8, 1.0))
}


# STV Strength Maps for Different Property Types
DETERMINISTIC_STV = (1.0, 1.0)  # For API-sourced data: author, category, title
AI_FAILURE_STV = (0.5, 0.5)      # For failed AI analysis (fallback)
UNKNOWN_STV = (0.5, 0.5)         # For unknown/missing values

# AI Prompts
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following article content and metadata to provide categorical classifications with STV values.

Return ONLY a JSON object with the following structure:
{{
    "tone": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "audience_expertise": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "content_type": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "primary_goal": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "audience_sentiment": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}}
}}

Article Title: {title}
Content Snippet: {content_snippet}

Classification Guidelines:
- Tone: Formal, Casual, Instructional
  * Formal: Academic, professional language, structured arguments
  * Casual: Conversational, personal anecdotes, relaxed style
  * Instructional: Step-by-step, educational, directive language

- Audience Expertise: Beginner, Intermediate, Advanced, Expert
  * Beginner: Simple concepts, basic terminology, explanatory
  * Intermediate: Some background assumed, moderate complexity
  * Advanced: Technical depth, specialized knowledge assumed
  * Expert: Cutting-edge, highly specialized, research-level

- Content Type: Tutorial, Opinion, Review, Interview
  * Tutorial: Teaching how to do something, practical guidance
  * Opinion: Personal viewpoint, persuasive argument, editorial
  * Review: Critical assessment, comparison, evaluation
  * Interview: Q&A format, dialogue, personal stories

- Primary Goal: Inform, Persuade, Entertain
  * Inform: Educational, factual, knowledge-sharing
  * Persuade: Convincing, argumentative, call-to-action
  * Entertain: Engaging, storytelling, humorous

- Audience Sentiment: Positive, Neutral, Negative, Mixed
  * Positive: Optimistic, encouraging, favorable outlook
  * Neutral: Objective, balanced, factual reporting
  * Negative: Critical, warning, concerning outlook
  * Mixed: Balanced pros/cons, nuanced perspective

STV Assignment Guidelines:
- strength (s): The probability or degree of truth in your classification [0.0-1.0]
- confidence (c): Your certainty in that strength measurement [0.0-1.0]
- Be honest about your certainty level - if you're guessing, use lower confidence
- If you're very confident based on clear evidence, use higher confidence
- Consider content clarity, explicit indicators, and your expertise in the domain


Consider the content's language, structure, purpose, and target audience when assigning both classifications and STV values.
"""
