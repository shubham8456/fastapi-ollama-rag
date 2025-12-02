# FastAPI-Ollama RAG Implementation

A Retrieval-Augmented Generation (RAG) system built with FastAPI, running entirely on local open-source models.

## System Architecture

- **Backend**: FastAPI (Python 3.14)
- **Embedding Model**: Snowflake Arctic Embed XS (local via Transformers)
- **Retrieval Model**: Gemma 2 2B (local via Ollama)
- **Vector Store**: FAISS (modular, easily swappable with Qdrant)
- **Package Manager**: UV
- **Deployment**: Docker + Docker Compose

## Features

- ✅ Fully local, no API keys required
- ✅ Lazy model loading for memory efficiency
- ✅ Modular vector store interface (FAISS/Qdrant)
- ✅ Interactive web UI
- ✅ Optimized for deployment on machines with limited resources

## System Requirements

### Minimum
- RAM: 8GB (4GB may work with aggressive lazy loading)
- CPU: 4 cores
- Storage: 10GB free

## Installation

### 1. Clone Repository
```bash
git clone git@github.com:shubham8456/fastapi-ollama-rag.git
cd fastapi-ollama-rag
```

### 2. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Environment File
```bash
cp .env.example .env
```

### 4. Add Sample Documents
Place PDF files in `data/documents/` directory.

### 5. Build Index (First Time)
```bash
uv run python -m scripts.build_index
```

### 6. Start with Docker Compose
```bash
docker compose up --build
```

### 7. Pull Gemma Model in Ollama
```bash
docker exec -it rag-ollama ollama pull gemma2:2b
```

## API Endpoints

### GET `/`
Serves the web UI.

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "FastAPI-Ollama RAG Implementation",
  "version": "0.1.0",
  "embedding_model": "Snowflake/snowflake-arctic-embed-xs",
  "retrieval_model": "gemma2:2b",
  "ollama_available": true
}
```

### POST `/query`
Execute RAG query.

**Request:**
```json
{
  "question": "What is RAG?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "RAG stands for...",
  "sources": [
    {
      "doc_id": "chunk_0",
      "score": 0.95,
      "snippet": "Context snippet...",
      "metadata": {
        "source": "doc.pdf"
      }
    }
  ],
  "retrieval_model": "gemma2:2b",
  "embedding_model": "Snowflake/snowflake-arctic-embed-xs"
}
```

## Usage

1. Open browser: `http://localhost:8000`
2. Enter your question (max 200 words)
3. Click Submit or press Ctrl+Enter
4. View answer with source citations

## Development

### Local Development (without Docker)
```bash
uv sync
uv run uvicorn app.main:app --reload
```

### Code Formatting
```bash
uv run black app/
uv run ruff check app/
```

## Switching to Qdrant

To switch from FAISS to Qdrant:

1. Update `.env`:
```yaml
VECTOR_STORE_TYPE=qdrant
```

2. Implement `QdrantVectorStore` in `app/storage/vector_store.py`

3. Add qdrant-client to dependencies

4. Update docker-compose.yml to include Qdrant service

## Performance Notes

- First query after idle: 30-60s (model loading)
- Subsequent queries: 5-15s
- Embedding 50-page PDF: 3-10 minutes on Raspberry-Pi

## Troubleshooting

**Ollama not responding:**
```bash
docker logs rag-ollama
docker exec -it rag-ollama ollama list
```

**Out of memory:**
- Reduce `top_k` in queries
- Enable `LAZY_LOAD_MODELS=true`
- Use smaller chunk sizes

**Slow responses:**
- Consider precomputing embeddings
- Use smaller models

## License

MIT

## Author
[Shubham Rawat](https://www.github.com/shubham8456/)

Built with FastAPI and ollama for RAG implementation
