"""API route handlers."""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceDocument,
)
from app.core.config import settings
from app.api.dependencies import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    """Serve main UI."""
    html_file = Path("app/templates/index.html")
    return FileResponse(html_file)


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_service=Depends(get_rag_service)):
    """Health check endpoint."""
    ollama_ok = await rag_service.ollama_service.health_check()
    
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        embedding_model=settings.embedding_model_name,
        retrieval_model=settings.retrieval_model_name,
        ollama_available=ollama_ok,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_service=Depends(get_rag_service),
):
    """Process RAG query."""
    try:
        answer, sources = await rag_service.query(
            question=request.question,
            top_k=request.top_k,
        )
        
        source_docs = [
            SourceDocument(
                doc_id=src.id,
                score=src.score,
                snippet=src.text[:200] + "..." if len(src.text) > 200 else src.text,
                metadata=src.metadata,
            )
            for src in sources
        ]
        
        return QueryResponse(
            answer=answer,
            sources=source_docs,
            retrieval_model=settings.retrieval_model_name,
            embedding_model=settings.embedding_model_name,
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
