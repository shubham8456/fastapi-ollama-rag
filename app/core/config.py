"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "RAG FastAPI POC"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Models
    embedding_model_name: str = "Snowflake/snowflake-arctic-embed-xs"
    retrieval_model_name: str = "gemma2:2b"
    ollama_base_url: str = "http://ollama:11434"

    # Vector Store
    vector_store_type: str = "faiss"
    faiss_index_path: str = "data/embeddings/faiss_index"
    top_k_results: int = 5

    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_query_words: int = 200

    # Runtime
    lazy_load_models: bool = True
    model_cache_timeout: int = 300


settings = Settings()
