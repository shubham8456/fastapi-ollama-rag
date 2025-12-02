"""Pydantic models for request/response validation."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question word count."""
        words = v.strip().split()
        if len(words) > 200:
            raise ValueError(f"Question exceeds maximum word count of 200 (got {len(words)})")
        return v.strip()


class SourceDocument(BaseModel):
    """Source document metadata."""

    doc_id: str
    score: float
    snippet: str
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str
    sources: List[SourceDocument]
    retrieval_model: str
    embedding_model: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    version: str
    embedding_model: str
    retrieval_model: str
    ollama_available: bool


class IngestRequest(BaseModel):
    """Request model for document ingestion (future use)."""

    document_name: str
    content: str


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    status: str
    chunks_created: int
    message: str
