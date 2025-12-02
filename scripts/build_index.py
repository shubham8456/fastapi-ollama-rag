"""Script to build FAISS index from documents."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.document_service import DocumentService
from app.storage.vector_store import create_vector_store


def main():
    """Build index from documents in data/documents/."""
    print("Initializing services...")
    
    # Initialize services
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model_name,
        lazy_load=False,
    )
    
    doc_service = DocumentService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    vector_store = create_vector_store(
        store_type=settings.vector_store_type,
        index_path=settings.faiss_index_path,
        dimension=384,
    )
    
    # Process documents
    docs_path = Path("data/documents")
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/documents/")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        chunks = doc_service.process_document(pdf_file)
        all_chunks.extend(chunks)
    
    print(f"Total chunks: {len(all_chunks)}")
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_service.embed(texts)
    
    # Add to vector store
    print("Adding to vector store...")
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    vector_store.add_embeddings(
        ids=ids,
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas,
    )
    
    # Persist
    print("Persisting index...")
    vector_store.persist()
    
    print("✓ Index built successfully!")
    print(f"Total vectors: {vector_store.count()}")


if __name__ == "__main__":
    main()
