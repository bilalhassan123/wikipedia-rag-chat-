"""Pydantic request and response models.

This module is excluded from coverage (DESIGN.md §12) — it contains data
classes only, no behaviour.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IndexArticleRequest(BaseModel):
    url: str = Field(..., description="A Wikipedia article URL")


class IndexArticleResponse(BaseModel):
    title: str
    summary: str
    chunk_count: int


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatHistoryItem] = Field(default_factory=list, max_length=20)


class ChatEvidence(BaseModel):
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    evidence: list[ChatEvidence]
