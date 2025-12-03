"""Vector Store interface."""

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
