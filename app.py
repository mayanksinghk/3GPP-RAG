import os
import sys
import json
import time
import uuid
import argparse
import logging
import glob
import re
from typing import List, Optional
from dataclasses import asdict
from tqdm import tqdm

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure these imports match your actual file structure
from src.core.indexing import QdrantManager, LocalOllamaEmbedder, ChunkValidator, VectorIndexer
from src.core.ingest import DocumentIngestor, ThreeGPPMarkdownChunker
from src.core.generation import RAGQueryEngine
from src.core.evaluate import RAGEvaluator, DEFAULT_GOLDEN_QA

import traceback

try:
    from src.create_dataset.ThreeGPPDownloader import ThreeGPPCorpusBuilder
except Exception as e:
    print(f"\n[CRITICAL IMPORT ERROR] {e}")
    traceback.print_exc()
    ThreeGPPCorpusBuilder = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-app")

# ==========================================
# FastAPI App & Schema for AnythingLLM
# ==========================================
api_app = FastAPI(title="3GPP & Codebase RAG API for AnythingLLM")
query_engine: Optional[RAGQueryEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "3gpp-rag-engine"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local-rag"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


@api_app.get("/v1/models", response_model=ModelListResponse)
def list_models():
    """Enables AnythingLLM to discover this engine as an available model."""
    return ModelListResponse(
        data=[
            ModelCard(id="3gpp-rag-engine"),
            ModelCard(id="qwen2.5-coder:7b")
        ]
    )


@api_app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Intercepts AnythingLLM prompts and runs the full RAG pipeline.
    """
    global query_engine
    if query_engine is None:
        query_engine = RAGQueryEngine()

    # Extract the latest user query from the conversation history
    user_messages = [m.content for m in req.messages if m.role.lower() == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided.")
    
    current_query = user_messages[-1]
    logger.info(f"Received query from AnythingLLM: {current_query}")

    # Execute CRAG Pipeline
    result = query_engine.ask(current_query)
    answer_text = result.get("answer", "")
    citations = result.get("citations", [])
    confidence = result.get("confidence", 0.0)

    # Format citations as an appendix for the AnythingLLM chat view
    if citations:
        answer_text += "\n\n---\n**Key References:**\n"
        for i, c in enumerate(citations, 1):
            answer_text += f"- [{i}] TS {c.get('spec', '?')} §{c.get('section', '?')} (Score: {c.get('score', 0):.2f})\n"
        answer_text += f"\n*Retrieval Confidence: {confidence:.0%}*"

    # Construct standard OpenAI response payload
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "3gpp-rag-engine",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(current_query.split()),
            "completion_tokens": len(answer_text.split()),
            "total_tokens": len(current_query.split()) + len(answer_text.split())
        }
    }


# ==========================================
# Core Pipeline Actions
# ==========================================
def action_download_corpus(release: str, series_list: List[str], all_series: bool, output_dir: str):
    """Downloads and converts 3GPP specifications into Markdown using ThreeGPPCorpusBuilder."""
    if ThreeGPPCorpusBuilder is None:
        logger.error("ThreeGPPCorpusBuilder could not be imported. Please verify its location in your project.")
        sys.exit(1)

    logger.info(f"Initializing ThreeGPPCorpusBuilder for Release: {release}")
    
    # Generate the full list of 3GPP series if the --all-series flag is passed
    ALL_3GPP_SERIES = [f"{i:02d}" for i in range(39)] + ["50", "51", "52", "53", "54", "55"]
    target_series = ALL_3GPP_SERIES if all_series else series_list
    
    logger.info(f"Target series to process: {', '.join(target_series)}")
    
    # 1. Initialize with all three required positional arguments
    builder = ThreeGPPCorpusBuilder(release=release, series_list=target_series, output_dir=output_dir)
    
    # 2. Execute the asynchronous download pipeline
    import asyncio
    asyncio.run(builder.download())
    
    # 3. Execute the synchronous Markdown conversion pipeline
    builder.process()

    logger.info(f"✓ Document corpus successfully downloaded and converted to {output_dir}")

def action_init_db():
    """Drops existing tables and initializes fresh pgvector tables & indexes."""
    logger.info("Initializing Qdrant schema from scratch...")
    db = QdrantManager()
    db.setup_schema()
    logger.info("✓ Database schema, vector extension, and HNSW/GIN indexes created successfully.")


def action_ingest_and_index(input_file: str, spec_number: str = "38331", release: str = "Rel-18"):
    """Ingests a markdown/text document, extracts LLM metadata, and indexes into pgvector."""
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    logger.info(f"Chunking and enriching file '{input_file}' (Spec: {spec_number})...")
    chunker = ThreeGPPMarkdownChunker(chunk_size=1000)
    ingestor = DocumentIngestor(chunking_strategy=chunker)
    
    base_metadata = {
        "spec_number": spec_number,
        "release": release,
        "doc_type": "TS",
        "source_s3_key": input_file
    }
    chunks = ingestor.process_document(content, base_metadata)
    chunk_dicts = [asdict(c) for c in chunks]

    # Save intermediate JSONL
    out_jsonl = f"{os.path.splitext(input_file)[0]}_chunks.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in chunk_dicts:
            f.write(json.dumps(c) + "\n")
    logger.info(f"Saved {len(chunk_dicts)} chunks to {out_jsonl}")

    # Index into Qdrant
    db = QdrantManager()
    embedder = LocalOllamaEmbedder()
    validator = ChunkValidator()
    indexer = VectorIndexer(db=db, embedder=embedder, validator=validator)
    
    indexed, failed = indexer.index_chunks(chunk_dicts)
    logger.info(f"✓ Completed indexing: {indexed} succeeded, {failed} failed.")


def action_evaluate(golden_file: Optional[str] = None):
    """Runs the LLM-as-a-judge evaluation harness against golden QA pairs."""
    logger.info("Starting pipeline evaluation...")
    engine = RAGQueryEngine()
    evaluator = RAGEvaluator(engine)

    qa_set = DEFAULT_GOLDEN_QA
    if golden_file and os.path.exists(golden_file):
        with open(golden_file, "r", encoding="utf-8") as f:
            qa_set = json.load(f)

    report = evaluator.evaluate_batch(qa_set)
    agg = report["aggregate"]

    print(f"\n{'='*55}")
    print("           RAG EVALUATION METRICS REPORT")
    print(f"{'='*55}")
    print(f" Faithfulness (No Hallucination) : {agg['faithfulness']:.2%}")
    print(f" Answer Relevancy                : {agg['relevancy']:.2%}")
    print(f" Answer Completeness             : {agg['completeness']:.2%}")
    print(f" Avg Response Latency            : {agg['avg_latency']:.2f}s")
    print(f"{'='*55}\n")

    saved_path = evaluator.save_report(report)
    logger.info(f"Full evaluation report saved to: {saved_path}")


# ==========================================
# CLI Dispatcher
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Master Controller for 3GPP RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: download (Build Corpus)
    download_parser = subparsers.add_parser("download", help="Download and convert 3GPP specs to Markdown")
    download_parser.add_argument("--release", type=str, default="Rel-18", help="Target 3GPP release (e.g., Rel-18)")
    download_parser.add_argument("--series", type=str, nargs='+', default=["38"], help="List of series to download (e.g., 24 33 38)")
    download_parser.add_argument("--all-series", action="store_true", help="Download all available series for the release")
    download_parser.add_argument("--out-dir", type=str, default="./output_markdowns", help="Directory to save the markdown files")

    # Command: init-db
    subparsers.add_parser("init-db", help="Create/reset Qdrant schema and pgvector indexes")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a raw specification document or directory")
    ingest_parser.add_argument("--file", type=str, default=None, help="Path to a single markdown file")
    ingest_parser.add_argument("--dir", type=str, default=None, help="Path to a directory containing multiple markdown files")
    ingest_parser.add_argument("--spec", type=str, default="unknown", help="3GPP specification number (used as fallback for dir)")
    ingest_parser.add_argument("--release", type=str, default="Rel-18", help="Release version (e.g. Rel-18)")
    ingest_parser.add_argument("--reset-db", action="store_true", help="Delete and recreate the Qdrant database schema before ingesting")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run LLM-as-a-judge benchmark")
    eval_parser.add_argument("--golden", type=str, default=None, help="Path to custom golden Q&A JSON file")

    # Command: serve (default for AnythingLLM)
    serve_parser = subparsers.add_parser("serve", help="Start the OpenAI-compatible API server for AnythingLLM")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")

    args = parser.parse_args()

    if args.command == "download":
        action_download_corpus(args.release, args.series, args.all_series, args.out_dir)

    elif args.command == "init-db":
        action_init_db()
    
    elif args.command == "ingest":
        # Check for the reset flag and initialize the database if present
        if getattr(args, "reset_db", False):
            logger.info("Reset flag detected. Wiping database and recreating schema...")
            action_init_db()

        if getattr(args, "dir", None):
            # Batch process all .md files in the provided directory
            md_files = glob.glob(os.path.join(args.dir, "**/*.md"), recursive=True)
            if not md_files:
                logger.error(f"No .md files found in directory: {args.dir}")
                sys.exit(1)
                
            logger.info(f"Found {len(md_files)} markdown files. Starting batch ingestion...")

            for file_path in tqdm(md_files, desc="Overall Corpus Progress", unit="file"):
                filename = os.path.basename(file_path)
                spec_guess = args.spec
                
                # Attempt to extract spec number like '38331' or '38.331' from the filename
                match = re.search(r'(\d{2}\.?\d{3})', filename)
                if match:
                    spec_guess = match.group(1).replace(".", "")
                    
                action_ingest_and_index(file_path, spec_guess, args.release)
                
        elif getattr(args, "file", None):
            # Process a single file
            action_ingest_and_index(args.file, args.spec, args.release)
        else:
            logger.error("You must provide either --file or --dir for ingestion.")
            sys.exit(1)

    elif args.command == "evaluate":
        action_evaluate(args.golden)
        
    elif args.command == "serve" or args.command is None:
        port = getattr(args, "port", 8000)
        host = getattr(args, "host", "127.0.0.1")
        logger.info(f"Starting AnythingLLM-compatible API server on http://{host}:{port}")
        uvicorn.run(api_app, host=host, port=port)


if __name__ == "__main__":
    main()