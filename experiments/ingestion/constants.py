"""Constants used by the ingestion multi-agent pipeline."""

SUPPORTED_FILE_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".txt",
    ".md",
    ".log",
    ".html",
    ".htm",
    ".metta",
}

IGNORED_PROPERTY_SUFFIXES = {
    "_confidence",
    "-confidence",
    "_strength",
    "-strength",
}

GENERIC_BUCKET_LABELS = ["Very_Low", "Low", "Medium", "High", "Very_High"]

POSITIVE_TERMS = {
    "good",
    "great",
    "excellent",
    "positive",
    "benefit",
    "improve",
    "success",
    "win",
    "clear",
    "practical",
    "helpful",
}

NEGATIVE_TERMS = {
    "bad",
    "poor",
    "negative",
    "risk",
    "fail",
    "failure",
    "problem",
    "concern",
    "difficult",
    "hard",
    "loss",
}

INSTRUCTIONAL_HINTS = {"guide", "how", "tutorial", "step", "learn", "explain", "instruction"}
ANALYTICAL_HINTS = {"analysis", "benchmark", "evaluation", "compare", "study", "report"}
OPINION_HINTS = {"opinion", "think", "believe", "view", "perspective", "editorial"}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "be",
    "with",
    "as",
    "by",
    "this",
    "that",
    "it",
    "from",
    "at",
    "your",
    "you",
}
