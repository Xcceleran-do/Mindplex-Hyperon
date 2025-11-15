from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict

router = APIRouter()

class RecommendationRequest(BaseModel):
    creatorId: str
    constraints: Optional[Dict] = None
    topK: Optional[int] = 10

class RecommendationItem(BaseModel):
    contentId: str
    title: str
    score: float
    explanation: Dict

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    modelVersion: Optional[str] = None

@router.post('/recommendations', response_model=RecommendationResponse)
async def recommend(req: RecommendationRequest):
    # Placeholder: return empty list
    return RecommendationResponse(recommendations=[], modelVersion='v0')
