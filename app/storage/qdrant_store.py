"""Qdrant Store implementations."""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from app.storage.interface import VectorStoreInterface, SearchResult

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreInterface):
    """Qdrant-based vector store implementation."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "rag_documents",
        dimension: int = 384,
    ):
        """Initialize a Qdrant-based vector store.
    
        This sets up a Qdrant client, ensures that the target collection exists
        (creating it if necessary), and configures the collection to store dense
        vectors of the given dimension using cosine similarity.
    
        Args:
            qdrant_url: Base URL of the Qdrant instance (e.g. "http://qdrant:6333").
            collection_name: Name of Qdrant collection to store vectors and payloads.
            dimension: Embedding dimension (default 384 for Arctic-xs).
        """
        from qdrant_client import QdrantClient
        
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.dimension = dimension

        # Initialize Qdrant client
        self.client = QdrantClient(url=qdrant_url)

        # Create collection if it doesn't exist
        logger.info(f"Initializing Qdrant client at {qdrant_url}")
        self._create_collection_if_not_exists()

        logger.info(f"Qdrant collection '{collection_name}' ready (dimension: {dimension})")

    def _create_collection_if_not_exists(self) -> None:
        """Create collection if it doesn't exist."""
        from qdrant_client.http.models import Distance, VectorParams

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")

                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
                logger.info("Qdrant collection created successfully")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to Qdrant."""
        from qdrant_client.http.models import PointStruct
        import uuid

        # Convert string IDs to UUIDs (Qdrant requirement)
        uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)) for id_ in ids]

        # Prepare points
        points = []
        for i, (orig_id, uuid_id, embedding, text) in enumerate(zip(ids, uuid_ids, embeddings, texts)):
            point = PointStruct(
                id=uuid_id,
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    "original_id": orig_id,
                    "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
                },
            )
            points.append(point)

        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(f"Added {len(points)} embeddings to Qdrant")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search Qdrant for similar vectors."""
        if self.count() == 0:
            raise ValueError("Vector store is empty. Please run build_index.py first.")

        from qdrant_client import models

        # Search Qdrant
        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),  # Vector query
            limit=top_k,
            search_params=models.SearchParams(),
        )

        # Convert to SearchResult format
        results = []
        for hit in query_result.points:
            hit_id = hit.id if isinstance(hit.id, str) else str(hit.id)
            results.append(
                SearchResult(
                    id=hit.payload.get("original_id", hit_id),
                    score=float(hit.score),
                    metadata=hit.payload.get("metadata", {}),
                    text=hit.payload.get("text", ""),
                )
            )

        logger.info(f"Qdrant search returned {len(results)} results")
        return results

    def persist(self) -> None:
        """Qdrant persists automatically, but log for consistency."""
        logger.info(f"Qdrant collection '{self.collection_name}' persisted (automatic)")

    def load(self) -> None:
        """Qdrant loads automatically on connection."""
        logger.info(f"Qdrant collection '{self.collection_name}' loaded (automatic)")

    def count(self) -> int:
        """Return number of vectors in Qdrant collection."""
        count_response = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )
        return count_response.count
