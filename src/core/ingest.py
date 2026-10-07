import os
import json
import re
import uuid
import logging
import abc
from typing import List, Dict, Any
from dataclasses import dataclass

from src.core.dataset_utils import BaseChunkingStrategy, Chunk
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# ==========================================
# 1. Concrete Strategy Implementation
# ==========================================

class ThreeGPPMarkdownChunker(BaseChunkingStrategy):
    """
    Concrete implementation tailored for 3GPP specifications.
    Splits text along 3GPP hierarchical clause boundaries (e.g., '5.3.1').
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
        self.clause_pattern = re.compile(r"^(#+)\s+(\d+(?:\.\d+)*)\s+(.*)", re.MULTILINE)

    def chunk_document(self, text: str, source_metadata: Dict[str, Any]) -> List[Chunk]:
        chunks = []
        matches = list(self.clause_pattern.finditer(text))
        
        sections = []
        if not matches:
            sections.append({"number": "0", "title": "Document", "content": text})
        else:
            for i, match in enumerate(matches):
                start_idx = match.end()
                end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
                sections.append({
                    "number": match.group(2),
                    "title": match.group(3).strip(),
                    "content": text[start_idx:end_idx].strip()
                })

        for section in sections:
            if not section["content"]:
                continue
                
            chunk_metadata = source_metadata.copy()
            chunk_metadata.update({
                "clause_number": section["number"],
                "clause_title": section["title"],
                "breadcrumb": f"[§{section['number']} {section['title']}]"
            })
            
            # Sub-chunking if the section exceeds size limits
            if len(section["content"]) > self.chunk_size:
                start = 0
                while start < len(section["content"]):
                    end = start + self.chunk_size
                    sub_text = section["content"][start:end]
                    
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=f"{chunk_metadata['breadcrumb']}\n\n{sub_text.strip()}",
                        metadata=chunk_metadata.copy()
                    ))
                    start += (self.chunk_size - self.chunk_overlap)
            else:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=f"{chunk_metadata['breadcrumb']}\n\n{section['content']}",
                    metadata=chunk_metadata.copy()
                ))
                
        return chunks

# ==========================================
# 2. Main Ingestion Pipeline
# ==========================================

# ==========================================
# 2. Main Ingestion Pipeline
# ==========================================

class DocumentIngestor:
    """
    Coordinates the document ingestion lifecycle.
    Bypasses LLM generation and relies purely on structural metadata extracted by the chunking strategy.
    """
    def __init__(self, chunking_strategy: BaseChunkingStrategy):
        self.chunker = chunking_strategy
        # LLM initialization removed to maximize processing speed

    def process_document(self, text: str, source_metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Executes the full pipeline: splits the text using the injected strategy 
        and relies on the structural metadata already injected by the chunker.
        """
        logger.info(f"Chunking document with {self.chunker.__class__.__name__}...")
        
        # 1. Delegate text splitting to the injected strategy
        # The ThreeGPPMarkdownChunker already extracts structural metadata
        chunks = self.chunker.chunk_document(text, source_metadata)
        
        logger.info(f"Successfully processed {len(chunks)} chunks via deterministic regex.")
        
        # 2. Return chunks immediately without LLM bottleneck
        return chunks

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import json
    from dataclasses import asdict

    logging.basicConfig(level=logging.INFO)
    
    strategy = ThreeGPPMarkdownChunker(chunk_size=1000)
    ingestor = DocumentIngestor(chunking_strategy=strategy)
    
    sample_text = "## 5.3.1 Connection Setup\nThe UE shall transmit the RRCSetupRequest message..."
    base_meta = {"spec_number": "38331", "release": "Rel-18", "doc_type": "TS"}
    
    # Process the document
    final_chunks = ingestor.process_document(sample_text, base_meta)
    
    # Write the results to a JSONL file
    output_file = "sample_chunks.jsonl"
    with open(output_file, "w") as f:
        for chunk in final_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
            
    print(f"Saved {len(final_chunks)} chunks to {output_file}")