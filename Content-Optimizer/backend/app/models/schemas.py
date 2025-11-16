"""Pydantic schemas (Step 9).

Defines structured request/response models used across the API layers.

Design notes:
 - Keep recommendation-related models minimal but allow future extension (e.g. add more explanation fields).
 - Provide explicit health/admin schemas for better OpenAPI documentation.
 - Content models separate internal feature vectors (not exposed) from public metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Generic / utility models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(description="Overall API status: 'ok' or 'error'.")
    neo4j_version: Optional[str] = Field(default=None, description="Neo4j server version if available.")
    content_nodes: Optional[int] = Field(default=None, description="Count of Content nodes.")
    has_embeddings: Optional[bool] = Field(default=None, description="Whether any Content node has graph embeddings.")


class Pagination(BaseModel):
    offset: int = Field(0, ge=0, description="Offset for pagination.")
    limit: int = Field(10, gt=0, le=100, description="Maximum number of items to return.")


# ---------------------------------------------------------------------------
# Content / domain
# ---------------------------------------------------------------------------
class ContentMeta(BaseModel):
    contentId: str
    title: Optional[str] = None


class ScoreExplanation(BaseModel):
    score: float = Field(..., description="Raw model score for the item.")
    model: Optional[str] = Field(None, description="Model version identifier.")
    # room for future fields (e.g., feature contributions)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    creatorId: Optional[str] = Field('', description="Creator ID to filter by (optional).")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optional constraints (future use).")
    topK: int = Field(10, gt=0, le=50, description="Number of recommendations requested.")


class RecommendationItem(BaseModel):
    contentId: str
    title: Optional[str] = None
    score: float
    explanation: ScoreExplanation


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    modelVersion: Optional[str] = Field(None, description="Model version used for scoring.")


# ---------------------------------------------------------------------------
# Admin / retrain
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    limit: Optional[int] = Field(None, description="Optional limit of content items to use during retraining.")


class RetrainResponse(BaseModel):
    status: str = Field(..., description="Status message of retrain trigger.")
    modelVersion: Optional[str] = Field(None, description="New model version if training completed synchronously.")
    error: Optional[str] = Field(None, description="Error detail if retrain failed.")


class ProjectGraphResponse(BaseModel):
    status: str
    graphName: Optional[str] = None
    nodeCount: Optional[int] = None
    relationshipCount: Optional[int] = None


__all__ = [
    'HealthResponse',
    'Pagination',
    'ContentMeta',
    'ScoreExplanation',
    'RecommendationRequest',
    'RecommendationItem',
    'RecommendationResponse',
    'RetrainRequest',
    'RetrainResponse',
    'ProjectGraphResponse',
]
