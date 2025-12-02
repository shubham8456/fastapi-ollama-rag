"""Vector store abstraction and implementations."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Search result from vector store."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage and retrieval."""

    @abstractmethod
    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to the vector store.

        Args:
            ids: List of unique identifiers
            embeddings: Numpy array of shape (n, dim)
            texts: List of text chunks
            metadatas: Optional list of metadata dicts
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dim,)
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of vectors in store."""
        pass


class FaissVectorStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""

    def __init__(self, index_path: str, dimension: int = 384):
        """Initialize FAISS vector store.

        Args:
            index_path: Path to save/load FAISS index
            dimension: Embedding dimension (default 384 for Arctic-xs)
        """
        import faiss
        import pickle
        from pathlib import Path

        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # Try to load existing index
        if self.index_path.exists():
            self.load()

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to FAISS index."""
        import faiss

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in ids])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search FAISS index for similar vectors."""
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.count()))

        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.ids):  # Valid index
                results.append(
                    SearchResult(
                        id=self.ids[idx],
                        score=float(1.0 / (1.0 + dist)),  # Convert distance to similarity score
                        metadata=self.metadatas[idx],
                        text=self.texts[idx],
                    )
                )

        return results

    def persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        import faiss
        import pickle

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path) + ".index")

        # Save metadata
        metadata = {
            "ids": self.ids,
            "texts": self.texts,
            "metadatas": self.metadatas,
            "dimension": self.dimension,
        }
        with open(str(self.index_path) + ".meta", "wb") as f:
            pickle.dump(metadata, f)

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss
        import pickle

        index_file = str(self.index_path) + ".index"
        meta_file = str(self.index_path) + ".meta"

        if Path(index_file).exists() and Path(meta_file).exists():
            # Load FAISS index
            self.index = faiss.read_index(index_file)

            # Load metadata
            with open(meta_file, "rb") as f:
                metadata = pickle.load(f)

            self.ids = metadata["ids"]
            self.texts = metadata["texts"]
            self.metadatas = metadata["metadatas"]
            self.dimension = metadata["dimension"]

    def count(self) -> int:
        """Return number of vectors in index."""
        return self.index.ntotal


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
    # elif store_type.lower() == "qdrant":
    #     return QdrantVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
