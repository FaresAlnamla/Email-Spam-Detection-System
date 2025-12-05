# api/schemas.py
from __future__ import annotations

"""
Pydantic schemas for the Spam Detector API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictIn(BaseModel):
    """Request body for /predict: a single message to classify."""
    text: str = Field(..., description="A single message to classify")


class PredictOut(BaseModel):
    """Response body for /predict."""
    text: str
    pred: str
    proba_spam: Optional[float] = None
    request_id: Optional[str] = None


class BatchPredictIn(BaseModel):
    """Request body for /batch: a list of messages."""
    texts: List[str] = Field(..., description="List of messages to classify")


class BatchPredictItem(BaseModel):
    """Single item in the /batch response."""
    text: str
    pred: str
    proba_spam: Optional[float] = None


class BatchPredictOut(BaseModel):
    """Response body for /batch."""
    size: int
    items: List[BatchPredictItem]
    request_id: Optional[str] = None
