"""Dependency injection for FastAPI."""
from functools import lru_cache
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService
from app.services.ollama_service import OllamaService
from app.storage.vector_store import create_vector_store
from app.core.config import settings


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        lazy_load=settings.lazy_load_models,
        cache_timeout=settings.model_cache_timeout,
    )


@lru_cache()
def get_ollama_service() -> OllamaService:
    """Get or create Ollama service singleton."""
    return OllamaService(
        base_url=settings.ollama_base_url,
        model_name=settings.retrieval_model_name,
    )


@lru_cache()
def get_vector_store():
    """Get or create vector store singleton."""
    embedding_service = get_embedding_service()
    dim = 384  # Arctic-xs dimension
    
    return create_vector_store(
        store_type=settings.vector_store_type,
        index_path=settings.faiss_index_path,
        dimension=dim,
    )


def get_rag_service() -> RAGService:
    """Get RAG service with all dependencies."""
    return RAGService(
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        ollama_service=get_ollama_service(),
    )
