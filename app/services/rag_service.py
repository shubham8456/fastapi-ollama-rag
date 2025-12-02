"""RAG service orchestrating embeddings, retrieval, and generation."""
import logging
from typing import List
from app.storage.vector_store import VectorStoreInterface, SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG pipeline service."""
    
    def __init__(
        self,
        vector_store: VectorStoreInterface,
        embedding_service: EmbeddingService,
        ollama_service: OllamaService,
    ):
        """Initialize RAG service.
        
        Args:
            vector_store: Vector storage implementation
            embedding_service: Embedding generation service
            ollama_service: Ollama LLM service
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.ollama_service = ollama_service
    
    def _build_prompt(self, question: str, context_chunks: List[SearchResult]) -> str:
        """Build prompt for LLM with context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"[Source {i+1}]: {chunk.text}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context below. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context_text}

User Question: {question}

Answer:"""
        return prompt
    
    async def query(self, question: str, top_k: int = 5) -> tuple[str, List[SearchResult]]:
        """Execute RAG query pipeline.
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # 1. Embed question
        query_embedding = self.embedding_service.embed_query(question)
        
        # 2. Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info(f"Retrieved {len(results)} context chunks")
        
        # 3. Build prompt
        prompt = self._build_prompt(question, results)
        
        # 4. Generate answer
        answer = await self.ollama_service.generate(prompt)
        logger.info("Answer generated successfully")
        
        return answer, results
