## Major Components of a RAG System
1. Ingestion & Parsing Pipeline
	- Document Parsers: Extracts text, tables, and structures from raw formats like PDFs, DOCX, or HTML (e.g., PyMuPDF, Unstructured).
	- Chunking Strategy: Breaks long text into smaller, manageable, overlapping segments (tokens or sentences) for precise matching.
2. Indexing & Storage Layer
	- Embedding Model: Converts text chunks into dense numeric vectors capturing semantic meaning (e.g., OpenAI text-embedding-3).
	- Vector Database / Search Index: Stores vectors and metadata for fast similarity matching (e.g., Pinecone, ChromaDB, Milvus, Elasticsearch).
3. Retrieval & Processing Layer
    - Retriever: Scans the database using semantic (vector) or hybrid (keyword + vector) search to pull the top-K matching context chunks.
    - Reranker: Filters and re-orders retrieved chunks based on deeper relevance scoring to minimize noise sent to the model.
4. Generation Layer
    - Prompt & Context Builder: Combines the user query with the filtered retrieved chunks into an optimized prompt template.
    - Large Language Model (LLM): Synthesizes the final grounded answer utilizing the provided context (e.g., GPT-4o, Claude 3.5, or local open-weights models via Ollama).
    - Citation Layer: Appends source documents and references back to the UI for verification

## Frontend
Best Dedicated Document Chat RAG: AnythingLLM



## References
- https://github.com/Indianinnovation/3gpp-rag-pipeline/tree/main
- https://github.com/huangl22/Chat3GPP
- https://www.ibm.com/think/topics/ridge-regression#44232980