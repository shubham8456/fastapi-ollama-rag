"""Ollama LLM service for text generation."""
import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with Ollama API."""
    
    def __init__(self, base_url: str, model_name: str):
        """Initialize Ollama service.
        
        Args:
            base_url: Ollama server base URL
            model_name: Model name to use for generation
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(self, prompt: str) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama is accessible.
        
        Returns:
            True if Ollama is available
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
