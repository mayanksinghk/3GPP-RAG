import os
import json
import abc
import logging
import uuid
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

from qdrant_client import QdrantClient, models
from openai import OpenAI

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

# ==========================================
# 1. Component Interfaces
# ==========================================

class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

class ChunkValidator:
    """Handles metadata completeness validation."""
    def __init__(self, log_path: str = "rejected_chunks.jsonl"):
        self.log_path = log_path

    def is_valid(self, chunk: Dict) -> bool:
        meta = chunk.get("metadata", {})
        section_path = (meta.get("breadcrumb") or meta.get("section_path") or "").strip()
        
        if not section_path or section_path.lower() in ("auto-ingested", "unknown", "none"):
            return False
        if section_path.isdigit():
            return False
        return True

    def log_rejection(self, chunk: Dict, reason: str):
        meta = chunk.get("metadata", {})
        entry = {
            "chunk_id": chunk.get("chunk_id", "?"),
            "spec_number": meta.get("spec_number", "?"),
            "reason": reason,
            "text_preview": chunk.get("text", "")[:100]
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ==========================================
# 2. Concrete Implementations
# ==========================================

class LocalOllamaEmbedder(BaseEmbedder):
    """Generates embeddings using a local Ollama instance via OpenAI client."""
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1"),
            api_key="sk-dummy-key"
        )
        self.model_id = os.getenv("EMBED_MODEL_ID", "nomic-embed-text")
        self.dim = int(os.getenv("EMBED_DIM", 768))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [t.strip()[:8000] if t and t.strip() else "empty" for t in texts]
        
        try:
            response = self.client.embeddings.create(
                input=clean_texts,
                model=self.model_id
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [[0.0] * self.dim for _ in texts]


class QdrantManager:
    """Handles database connections, schema creation, and vector search."""
    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION", "3gpp_chunks")
        self.dim = int(os.getenv("EMBED_DIM", 768))
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    def setup_schema(self):
        logger.info(f"Recreating Qdrant collection '{self.collection_name}'...")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dim, 
                distance=models.Distance.COSINE,
            )
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="spec_number",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def hybrid_search(self, query: str, query_emb: List[float], top_k: int = 10, spec_filter: str = None) -> List[Dict]:
            query_filter = None
            
            if spec_filter:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="spec_number",
                            match=models.MatchValue(value=spec_filter)
                        )
                    ]
                )
                
            # 1. Use query_points instead of search
            # 2. Use query= instead of query_vector=
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_emb,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True
            )
            
            results = []
            # 3. Iterate over response.points instead of the raw response
            for scored_point in response.points:
                payload = scored_point.payload or {}
                results.append({
                    "chunk_id": payload.get("chunk_id", str(scored_point.id)),
                    "spec_number": payload.get("spec_number", ""),
                    "release": payload.get("release", ""),
                    "section_path": payload.get("section_path", ""),
                    "doc_type": payload.get("doc_type", "TS"),
                    "chunk_text": payload.get("chunk_text", ""),
                    "score": round(scored_point.score, 4)
                })
            return results

# ==========================================
# 3. Main Indexing Orchestrator
# ==========================================

class VectorIndexer:
    def __init__(self, db: QdrantManager, embedder: BaseEmbedder, validator: ChunkValidator, batch_size: int = 200):
        self.db = db
        self.embedder = embedder
        self.validator = validator
        self.batch_size = batch_size

    def index_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        valid_chunks = []
        for c in chunks:
            if self.validator.is_valid(c):
                valid_chunks.append(c)
            else:
                self.validator.log_rejection(c, "invalid_section_path")

        if len(valid_chunks) != len(chunks):
            logger.warning(f"Rejected {len(chunks) - len(valid_chunks)} chunks for invalid metadata.")
            
        if not valid_chunks:
            return 0, 0

        # Pre-slice chunks into batches
        batches = [valid_chunks[i:i + self.batch_size] for i in range(0, len(valid_chunks), self.batch_size)]
        
        def process_batch(batch):
            texts = [c.get("text", "") for c in batch]
            embeddings = self.embedder.embed_texts(texts)
            
            points = []
            for chunk, emb in zip(batch, embeddings):
                meta = chunk.get("metadata", {})
                chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                
                payload = {
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id", "unknown"),
                    "spec_series": meta.get("spec_series", ""),
                    "spec_number": meta.get("spec_number", ""),
                    "release": meta.get("release", ""),
                    "section_path": meta.get("breadcrumb", meta.get("section_path", "")),
                    "doc_type": meta.get("doc_type", "TS"),
                    "chunk_text": chunk.get("text", ""),
                    "token_count": meta.get("token_count", 0),
                    "source_s3_key": meta.get("source_s3_key", "")
                }
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=payload
                    )
                )

            self.db.client.upsert(
                collection_name=self.db.collection_name,
                points=points
            )
            return len(points)

        total_ok = 0
        total_fail = 0
        
        # Multithreaded execution with tqdm progress bar
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_batch, b): b for b in batches}
            
            for future in tqdm(as_completed(futures), total=len(batches), desc="Indexing Batches", unit="batch"):
                try:
                    total_ok += future.result()
                except Exception as e:
                    logger.error(f"Batch insertion failed: {e}")
                    total_fail += len(futures[future])

        return total_ok, total_fail


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--local-file", type=str, help="Path to a local JSONL file to index")
    args = parser.parse_args()

    # Initialize dependencies
    db = QdrantManager()
    embedder = LocalOllamaEmbedder()
    validator = ChunkValidator()
    indexer = VectorIndexer(db, embedder, validator)

    if args.reindex:
        db.setup_schema()

    if args.local_file and os.path.exists(args.local_file):
        with open(args.local_file, 'r') as f:
            chunks = [json.loads(line) for line in f if line.strip()]
            
        logger.info(f"Indexing {len(chunks)} chunks from {args.local_file}...")
        ok, fail = indexer.index_chunks(chunks)
        logger.info(f"Done. Indexed: {ok}, Failed: {fail}")
        
        # Run Sanity Search
        logger.info("Running Sanity Search...")
        query = "RRC state machine idle connected"
        q_emb = embedder.embed_texts([query])[0]
        results = db.hybrid_search(query, q_emb, top_k=2)
        for r in results:
            print(f"[{r['score']:.3f}] {r['section_path']} | {r['chunk_text'][:80]}...")