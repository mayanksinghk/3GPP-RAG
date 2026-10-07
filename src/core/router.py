import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ContextRouter:
    """
    Intercepts retrieved chunks, dynamically routes them based on metadata 
    (e.g., source code vs. standards), and applies specialized ranking or formatting.
    """
    def __init__(self):
        # In the future, you can add cross-encoder models here for advanced re-ranking
        pass

    def route_and_format(self, retrieved_chunks: List[Dict]) -> str:
        """
        Separates heterogeneous chunks by their doc_type and formats them 
        into a unified, structured prompt context.
        """
        if not retrieved_chunks:
            return ""

        # Route chunks into specific buckets based on their metadata
        standards_chunks = []
        codebase_chunks = []
        
        for chunk in retrieved_chunks:
            # Assuming 'doc_type' is TS/TR for 3GPP, and 'CPP_CODE' for your future codebase
            doc_type = chunk.get("doc_type", "TS")
            
            if doc_type in ["TS", "TR"]:
                standards_chunks.append(chunk)
            elif doc_type == "CPP_CODE":
                codebase_chunks.append(chunk)
            else:
                # Fallback for generic documents
                standards_chunks.append(chunk)

        # Build the final context string dynamically
        formatted_context = ""
        
        if standards_chunks:
            formatted_context += "### 3GPP TECHNICAL SPECIFICATIONS ###\n"
            for c in standards_chunks:
                formatted_context += f"- Source: §{c.get('section_path')} (TS{c.get('spec_number')} {c.get('release', '')})\n"
                formatted_context += f"  Content: {c.get('chunk_text')}\n\n"
                
        if codebase_chunks:
            formatted_context += "### C++ CODEBASE ###\n"
            for c in codebase_chunks:
                # Code requires different context formatting (e.g., file paths, AST nodes)
                formatted_context += f"- File: {c.get('source_s3_key')} | Node: {c.get('section_path')}\n"
                formatted_context += f"  ```cpp\n{c.get('chunk_text')}\n  ```\n\n"

        return formatted_context.strip()