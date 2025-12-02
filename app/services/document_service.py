"""Document processing and chunking service."""
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for loading and processing documents."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document service.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: Path) -> str:
        """Load text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": metadata or {},
                })
                chunk_id += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a document file into chunks.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Processing document: {file_path.name}")
        
        if file_path.suffix.lower() == ".pdf":
            text = self.load_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")
        
        metadata = {
            "source": file_path.name,
            "file_type": file_path.suffix,
        }
        
        return self.chunk_text(text, metadata)
