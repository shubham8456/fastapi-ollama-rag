"""Embedding model service with lazy loading support."""
import logging
from typing import List, Optional
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Transformers."""

    def __init__(
        self,
        model_name: str,
        lazy_load: bool = True,
        cache_timeout: int = 300,
    ):
        """Initialize embedding service.

        Args:
            model_name: Hugging Face model name
            lazy_load: Whether to load model on-demand
            cache_timeout: Seconds to keep model in memory after last use
        """
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.cache_timeout = cache_timeout

        self.model = None
        self.tokenizer = None
        self.last_used: Optional[datetime] = None

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load embedding model and tokenizer into memory."""
        if self.model is not None:
            logger.info("Embedding model already loaded")
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        from transformers import AutoTokenizer, AutoModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Move to CPU (for Raspberry Pi compatibility)
        self.device = "cpu"
        self.model.to(self.device)

        logger.info("Embedding model loaded successfully")

    def _unload_model(self) -> None:
        """Unload model from memory."""
        if self.model is None:
            return

        logger.info("Unloading embedding model to free memory")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        # Force garbage collection
        import gc
        gc.collect()

    def _should_unload(self) -> bool:
        """Check if model should be unloaded based on cache timeout."""
        if not self.lazy_load or self.last_used is None:
            return False

        elapsed = (datetime.now() - self.last_used).total_seconds()
        return elapsed > self.cache_timeout

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        import torch

        # Load model if needed
        if self.model is None:
            self._load_model()

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Update last used timestamp
        self.last_used = datetime.now()

        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()

        return embeddings_np

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: Query text

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embeddings = self.embed([query])
        return embeddings[0]

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension size
        """
        # Load model temporarily if needed
        if self.model is None:
            self._load_model()
            sample_emb = self.embed(["test"])
            dim = sample_emb.shape[1]
            if self.lazy_load:
                self._unload_model()
            return dim

        # Get from loaded model
        return self.model.config.hidden_size

    def cleanup(self) -> None:
        """Manual cleanup to unload model."""
        self._unload_model()
