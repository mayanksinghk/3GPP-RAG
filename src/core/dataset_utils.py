import abc
import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Standardized data structure for all ingested chunks."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class BaseChunkingStrategy(abc.ABC):
    """
    Abstract interface for document ingestion and chunking.
    Enforces a standard contract so different parsing algorithms (e.g., Semantic, 
    Recursive Character, AST-based) can be swapped seamlessly in the pipeline.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abc.abstractmethod
    def chunk_document(self, text: str, source_metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits a full document's text into processed logical chunks.
        
        Args:
            text: The raw Markdown/HTML string of the document.
            source_metadata: Dictionary containing document-level context 
                             (e.g., spec_number, release, series).
                             
        Returns:
            List[Chunk]: A list of Chunk objects ready for vector embedding.
        """
        pass
