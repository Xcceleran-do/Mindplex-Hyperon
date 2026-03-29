"""Analysis-oriented ingestion agents."""

from __future__ import annotations

from ..models import IngestionConfig, IngestionState
from ..tool_router import ToolRouter
from .base import Agent


class SentimentAnalysisAgent(Agent):
    name = "sentiment-analysis"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        for record in state.records:
            text = str(record.get("content", ""))
            title = str(record.get("title", ""))
            if not (text or title):
                continue
            sentiment = tools.sentiment_analysis(f"{title} {text}")
            record["audience_sentiment"] = sentiment


class ContentClassificationAgent(Agent):
    name = "content-classification"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        for record in state.records:
            title = str(record.get("title", ""))
            text = str(record.get("content", ""))
            if not (title or text):
                continue
            classification = tools.classify_content(title, text)
            record["content_class"] = classification


class SemanticParsingAgent(Agent):
    name = "semantic-parser"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        for record in state.records:
            title = str(record.get("title", ""))
            text = str(record.get("content", ""))
            parsed = tools.semantic_parse(title, text)
            keywords = parsed.get("keywords", [])
            if keywords:
                record["semantic_keywords"] = {
                    "value": ", ".join(keywords),
                    "confidence": parsed.get("confidence", 0.75),
                    "strength": parsed.get("strength", 0.8),
                }


class RecommendationSignalAgent(Agent):
    name = "recommendation-signal"

    def run(self, state: IngestionState, config: IngestionConfig, tools: ToolRouter) -> None:
        for record in state.records:
            signals = tools.derive_recommendation_signals(record)
            for key, value in signals.items():
                record[key] = value
