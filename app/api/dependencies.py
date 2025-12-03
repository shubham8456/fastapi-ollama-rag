"""Dependency injection for FastAPI."""
from functools import lru_cache
from app.services.rag_service import RAGService
from app.storage.vector_store import create_vector_store, VectorStoreInterface
from app.core.config import settings
from pathlib import Path


@lru_cache()
def get_embedding_service():
    """Get or create embedding service singleton."""
    from app.services.embedding_service import EmbeddingService
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        lazy_load=settings.lazy_load_models,
        cache_timeout=settings.model_cache_timeout,
    )


@lru_cache()
def get_ollama_service():
    """Get or create Ollama service singleton."""
    from app.services.ollama_service import OllamaService
    return OllamaService(
        base_url=settings.ollama_base_url,
        model_name=settings.retrieval_model_name,
    )


@lru_cache()
def get_vector_store() -> VectorStoreInterface:
    """Get or create vector store singleton."""
    dim = getattr(settings, 'qdrant_dimension', 384)
    
    if settings.vector_store_type.lower() == 'faiss':
        index_path = Path(settings.faiss_index_path)
        if not index_path.is_absolute():
            index_path = Path("/app") / index_path
        return create_vector_store(
            store_type='faiss',
            index_path=str(index_path),
            dimension=dim,
        )
    elif settings.vector_store_type.lower() == 'qdrant':
        return create_vector_store(
            store_type='qdrant',
            qdrant_url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
            dimension=dim,
        )
    else:
        raise ValueError(f"Unsupported vector store: {settings.vector_store_type}")


def get_rag_service() -> RAGService:
    """Get RAG service with all dependencies."""
    return RAGService(
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        ollama_service=get_ollama_service(),
    )
