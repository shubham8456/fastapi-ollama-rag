"""Vector store abstraction and implementations."""

from app.storage.interface import VectorStoreInterface
from app.storage.faiss_store import FaissVectorStore
from app.storage.qdrant_store import QdrantVectorStore


# Factory function for easy switching
def create_vector_store(store_type: str, **kwargs) -> VectorStoreInterface:
    """Factory to create vector store instances.

    Args:
        store_type: Type of vector store ('faiss', 'qdrant', etc.)
        **kwargs: Additional arguments for the vector store

    Returns:
        VectorStoreInterface implementation
    """
    if store_type.lower() == "faiss":
        return FaissVectorStore(**kwargs)
    elif store_type.lower() == "qdrant":
        return QdrantVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
