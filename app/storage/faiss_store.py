"""Faiss Store implementations."""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from app.storage.interface import VectorStoreInterface, SearchResult

logger = logging.getLogger(__name__)


class FaissVectorStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""

    def __init__(self, index_path: str, dimension: int = 384):
        """Initialize FAISS vector store.

        Args:
            index_path: Path to save/load FAISS index
            dimension: Embedding dimension (default 384 for Arctic-xs)
        """
        import faiss
        from pathlib import Path

        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # Check for existing index files
        index_file = Path(str(self.index_path) + ".index")
        meta_file = Path(str(self.index_path) + ".meta")

        logger.info("Initializing FAISS vector store")
        logger.info(f"Looking for index at: {index_file}")
        logger.info(f"Index file exists: {index_file.exists()}")
        logger.info(f"Meta file exists: {meta_file.exists()}")

        # Try to load existing index
        if index_file.exists() and meta_file.exists():
            logger.info("Loading existing FAISS index...")
            self.load()
            logger.info(f"Successfully loaded index with {self.count()} vectors")
        else:
            logger.warning("No existing index found, starting with empty index")

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to FAISS index."""

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

        if self.count() == 0:
            raise ValueError("Vector store is empty. Please run build_index.py first.")

        # Search
        k = min(top_k, self.count())
        if k <= 0:  # Additional safety check
            return []

        distances, indices = self.index.search(query_embedding, k)

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

        logger.info(f"Persisted {self.count()} vectors to {self.index_path}")

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss
        import pickle
        from pathlib import Path

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

            logger.info(f"Loaded {len(self.ids)} vectors from disk")
        else:
            logger.warning(f"Index files not found at {self.index_path}")

    def count(self) -> int:
        """Return number of vectors in index."""
        return self.index.ntotal
