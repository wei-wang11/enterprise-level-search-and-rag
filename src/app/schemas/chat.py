from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    usecase: str = "default"
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    doc_id: str
    title: str
    snippet: str
    page: Optional[int] = None
    url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class RetrievalHitResponse(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float
    page: Optional[int] = None
    url: Optional[str] = None


class RetrieveResponse(BaseModel):
    retrieval_query: str
    hits: list[RetrievalHitResponse] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    usecase: str = "default"
    retrieval_query: Optional[str] = None
    hits: list[RetrievalHitResponse] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    usecase: str = "default"
    retrieval_query: Optional[str] = None
    hits: list[RetrievalHitResponse] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    verdict: str
    reasoning: str
