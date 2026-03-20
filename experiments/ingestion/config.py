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
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following document content and metadata to provide categorical classifications with STV values.

Return ONLY a JSON object with the following structure:
{{
    "tone": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "audience_expertise": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "content_type": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "primary_goal": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
    "audience_sentiment": {{"value": "classification", "strength": 0.0-1.0, "confidence": 0.0-1.0}}
}}

Document Content Snippet: {content_snippet}

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

Consider the content's language, structure, purpose, and target audience when assigning both classifications and STV values.
"""

METADATA_SCHEMA_PROMPT_TEMPLATE = """You are a dynamic schema generator. Analyze the following document and determine what structured metadata fields would be valuable to extract. Do NOT extract the values yet, just identify the metadata keys/fields that are applicable.

Return ONLY a JSON object containing a list of strings:
{{
    "metadata_fields": ["Field1", "Field2", "Field3"]
}}

Document Content: {content_snippet}
"""

METADATA_EXTRACTION_PROMPT_TEMPLATE = """Extract the following metadata fields from the document.
Fields to extract: {fields_list}

Return ONLY a JSON object where each key is a requested field, and the value is the extracted information and STV. If a field cannot be found, omit it or set value to null.
{{
    "Field1": {{"value": "ExtractedValue", "strength": 0.0-1.0, "confidence": 0.0-1.0}}
}}

Document Content: {content_snippet}
"""

ENTITY_TYPE_PROMPT_TEMPLATE = """Identify the high-level categories/types of entities present in this document (e.g., Person, Programming Language, Company, Concept, Location).

Return ONLY a JSON object containing a list of strings:
{{
    "entity_types": ["Type1", "Type2"]
}}

Document Content: {content_snippet}
"""

ENTITY_EXTRACTION_PROMPT_TEMPLATE = """Extract the most relevant entities belonging to the following types from the document.
Entity Types to extract: {entity_types}

Return ONLY a JSON object with the following structure:
{{
    "entities": [
        {{"value": "entity_name", "type": "matched_type", "strength": 0.0-1.0, "confidence": 0.0-1.0}},
        ...
    ]
}}

Document Content: {content_snippet}
"""

CLASSIFICATION_PROMPT_TEMPLATE = """Based on the document content, classify its primary domain and format. Be specific but concise.

Return ONLY a JSON object:
{{
    "domain": "DomainName",
    "format": "FormatName",
    "confidence": 0.0-1.0
}}

Document Content: {content_snippet}
"""

SENTIMENT_PROMPT_TEMPLATE = """Analyze the sentiment of the following document.
Categorize as: Positive, Negative, Neutral, or Mixed.

Return ONLY a JSON object:
{{
    "sentiment": "Category",
    "strength": 0.0-1.0,
    "confidence": 0.0-1.0
}}

Document: {text}
"""

OPEN_IE_PROMPT_TEMPLATE = """Extract structured (Subject, Predicate, Object) triples from the following text.
Focus on the most important factual statements.

Return ONLY a JSON object:
{{
    "triples": [
        {{"subject": "S", "predicate": "P", "object": "O", "confidence": 0.0-1.0}},
        ...
    ]
}}

Text: {text}
"""
