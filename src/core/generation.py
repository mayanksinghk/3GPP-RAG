import os
import json
import re
import logging
import time
from typing import TypedDict, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Import your existing components from indexing (Updated to Qdrant)
from src.core.indexing import QdrantManager, LocalOllamaEmbedder


logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

# ==========================================
# Telecom Synonym Expansion & Blacklists
# ==========================================
TELECOM_SYNONYMS = {
    "clear code": ["Cause IE", "CauseRadioNetwork", "cause code", "release cause", "RLF-Cause"],
    "drop cause": ["CauseRadioNetwork", "radioConnectionWithUELost", "release cause"],
    "failure code": ["Cause IE", "CauseProtocol", "CauseMisc"],
    "error code": ["Cause IE", "CauseProtocol"],
    "release reason": ["ReleaseCause", "release cause", "RRC release"],
    "disconnect reason": ["Cause IE", "CauseRadioNetwork"],
    "reject cause": ["5GMM cause", "CauseNas"],
    "handover failure": ["handoverFailure", "CauseRadioNetwork"],
}

SPEC_ALIASES = {
    "ngap": "38.413", "f1ap": "38.473", "e1ap": "38.463", "xnap": "38.423",
    "rrc": "38331", "nas": "24501", "5gmm": "24501", "5gsm": "24501",
    "gtp": "29.281", "gtpu": "29.281",
}

CLAUSE_BLACKLIST = {
    "cause_code_query": [
        "9.3.1.111", "4.5.6", "5.6.1.4.1", "9.11.3.39", "5.3.10.5", "G.1.1.3.1"
    ]
}

SECTION_BLACKLIST_PATTERNS = ["change history", "abbreviations", "annex a", "annex b", "0 introduction", "payload container"]

CAUSE_CODE_SPECS = ["24501", "38331", "38473", "38463", "38423", "38.413"]

# ==========================================
# State Definition
# ==========================================
class RAGState(TypedDict):
    user_query: str
    spec_filter: Optional[str]
    release_filter: Optional[str]
    sub_queries: List[str]
    extracted_spec: Optional[str]
    extracted_rel: Optional[str]
    retrieved_chunks: List[dict]
    retrieval_hops: int
    eval_result: str
    refined_query: str
    answer: str
    citations: List[dict]
    confidence: float
    route: str
    hop_count: int


# ==========================================
# Context Router (Domain Routing & Formatting)
# ==========================================
class ContextRouter:
    """Routes and formats retrieved chunks based on their domain origin."""
    
    def route_and_format(self, retrieved_chunks: List[Dict], release_history: str = "") -> str:
        if not retrieved_chunks:
            return ""

        standards_chunks, codebase_chunks = [], []
        
        for chunk in retrieved_chunks:
            doc_type = chunk.get("doc_type", "TS")
            if doc_type in ["TS", "TR"]:
                standards_chunks.append(chunk)
            elif doc_type == "CPP_CODE":
                codebase_chunks.append(chunk)
            else:
                standards_chunks.append(chunk)

        formatted_context = ""
        
        if standards_chunks:
            formatted_context += "### 3GPP TECHNICAL SPECIFICATIONS ###\n"
            for i, c in enumerate(standards_chunks, 1):
                spec = c.get('spec_number', '?')
                section = c.get('section_path', '?')
                release = c.get('release', '?')
                formatted_context += f"[Source {i}: TS {spec} §{section} | {release}]\n{c.get('chunk_text', '')}\n\n"
                
        if codebase_chunks:
            formatted_context += "### C++ CODEBASE ###\n"
            for i, c in enumerate(codebase_chunks, 1):
                formatted_context += f"[Source {i}: File {c.get('source_s3_key')} | Node: {c.get('section_path')}]\n"
                formatted_context += f"```cpp\n{c.get('chunk_text', '')}\n```\n\n"

        if release_history:
            formatted_context += f"\n\n---\n\n[Source: CR Change History Database]\n{release_history}"

        return formatted_context.strip()


# ==========================================
# Advanced RAG Engine
# ==========================================
class RAGQueryEngine:
    def __init__(self):
        self.db = QdrantManager()
        self.embedder = LocalOllamaEmbedder()
        self.router = ContextRouter()
        
        # 1. Local Client: Reranking, Evaluation & Chunk Organization (Ollama)
        self.flash_client = OpenAI(
            base_url=os.getenv("FLASH_LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("FLASH_LLM_API_KEY", "sk-dummy-key")
        )
        self.flash_model = os.getenv("LLM_FLASH_MODEL_ID", "qwen2.5-coder:1.5b")
        
        # 2. Cloud Client: Final Synthesis & Answer Generation (Groq)
        self.generator_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("LLM_API_KEY")
        )
        self.generator_model = os.getenv("LLM_MODEL_ID", "llama-3.3-70b-versatile")
        
        self._retrieval_executor = ThreadPoolExecutor(max_workers=8)
        
        # Compile LangGraph
        self.graph = self._build_graph()

    # --- Node: Planner ---
    def _planner_node(self, state: RAGState) -> dict:
        logger.info("  [planner] Decomposing query into sub-queries (Flash Model)...")
        q = state["user_query"].lower()
        
        # Synonym Expansion
        expansions, detected_specs = [], []
        for term, synonyms in TELECOM_SYNONYMS.items():
            if term in q: expansions.extend(synonyms)
        for alias, spec_num in SPEC_ALIASES.items():
            if alias in q: detected_specs.append(spec_num)
            
        expanded_query = f"{state['user_query']}\n\n[Search also for: {', '.join(set(expansions))}]" if expansions else state['user_query']

        system_prompt = """You are a 3GPP specification query planner. Extract spec_number, release_version, and decompose the query into 2-5 focused sub-queries.
Return ONLY valid JSON (no markdown fences):
{"spec": "38413"|null, "release": "Rel-18"|null, "sub_queries": ["query1", "query2"], "search_terms": ["Term1", "Term2"]}"""

        try:
            resp = self.flash_client.chat.completions.create(
                model=self.flash_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": expanded_query}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.choices[0].message.content.strip())
            plan = json.loads(raw)
        except Exception as e:
            logger.warning(f"Planner failed: {e}")
            plan = {}

        sub_queries = plan.get("sub_queries") or [state["user_query"]]
        search_terms = plan.get("search_terms") or []
        if search_terms:
            sub_queries.append(" ".join(search_terms[:8]))

        spec_from_plan = plan.get("spec") or state.get("spec_filter")
        if not spec_from_plan and detected_specs:
            spec_from_plan = detected_specs[0]

        return {
            "extracted_spec": spec_from_plan,
            "extracted_rel": plan.get("release") or state.get("release_filter"),
            "sub_queries": sub_queries
        }

    # --- Node: Retriever ---
    def _retriever_node(self, state: RAGState) -> dict:
        logger.info("  [retriever] Multi-query hybrid search + RRF (parallel)...")
        user_spec = state.get("extracted_spec") or state.get("spec_filter")
        release_filter = state.get("extracted_rel") or state.get("release_filter")
        
        search_tasks = [(q, user_spec, release_filter, 8) for q in state["sub_queries"]]
        
        cause_keywords = ["cause", "clear code", "release cause", "reject", "failure cause"]
        is_cause_query = any(kw in state["user_query"].lower() for kw in cause_keywords)
        
        if not user_spec and is_cause_query:
            targeted_queries = ["NGAP cause RadioNetwork Transport", "5GMM cause value rejection", "RRC establishment cause"]
            for spec in CAUSE_CODE_SPECS:
                for tq in targeted_queries:
                    search_tasks.append((tq, spec, None, 3))

        # Parallel Execution
        def _search(args):
            q, spec, release, top_k = args
            q_emb = self.embedder.embed_texts([q])[0]
            return self.db.hybrid_search(q, q_emb, top_k=top_k, spec_filter=spec)

        t0 = time.time()
        futures = [self._retrieval_executor.submit(_search, task) for task in search_tasks]
        result_sets = [f.result() for f in futures if f.result()]
        logger.info(f"    [parallel] {len(search_tasks)} searches completed in {time.time()-t0:.1f}s")

        # Apply Blacklist
        if is_cause_query:
            bl = CLAUSE_BLACKLIST.get("cause_code_query", [])
            result_sets = [
                [c for c in rs if not any(b in (c.get("section_path") or "") for b in bl)] 
                for rs in result_sets
            ]

        # Reciprocal Rank Fusion & Normalization
        rrf_scores, chunk_map = {}, {}
        for rs in result_sets:
            for rank, chunk in enumerate(rs):
                cid = chunk["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
                chunk_map[cid] = chunk

        fused = []
        if rrf_scores:
            max_s, min_s = max(rrf_scores.values()), min(rrf_scores.values())
            spread = max_s - min_s or 1.0
            
            seen_clauses = set()
            for cid in sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True):
                c = chunk_map[cid].copy()
                clause = (c.get("section_path") or "unknown").strip()
                if clause not in seen_clauses: # Deduplicate by clause
                    c["score"] = (rrf_scores[cid] - min_s) / spread
                    fused.append(c)
                    seen_clauses.add(clause)

        total = state.get("retrieved_chunks", []) + fused
        return {"retrieved_chunks": total, "retrieval_hops": state.get("retrieval_hops", 0) + 1}

    # --- Node: IsREL Filter ---
    def _isrel_node(self, state: RAGState) -> dict:
        MIN_KEEP, MAX_KEEP = 15, 25
        chunks = state.get("retrieved_chunks", [])
        if len(chunks) <= MIN_KEEP: return {"retrieved_chunks": chunks}

        relevant, ambiguous = [], []
        query = state["user_query"].lower()

        for c in chunks:
            sec = (c.get("section_path") or "").lower()
            if "2 references" in sec or "1 scope" in sec: continue
            if "5gs mobile identity" in sec and "cause" in query: continue
            
            score = c.get("score", 0)
            if score >= 0.5 or any(kw in sec for kw in ["cause", "failure", "reject", "release", "9.3.1"]):
                relevant.append(c)
            elif score >= 0.3:
                relevant.append(c)
            else:
                ambiguous.append(c)

        if len(relevant) < MIN_KEEP and ambiguous:
            relevant.extend(sorted(ambiguous, key=lambda x: x.get("score", 0), reverse=True)[:MIN_KEEP - len(relevant)])
        
        relevant = sorted(relevant, key=lambda x: x.get("score", 0), reverse=True)[:MAX_KEEP]
        return {"retrieved_chunks": relevant}

    # --- Node: Reranker ---
    def _reranker_node(self, state: RAGState) -> dict:
        chunks = state.get("retrieved_chunks", [])
        for c in chunks:
            sec = (c.get("section_path") or "").lower()
            if "references" in sec or "introduction" in sec: c["score"] *= 0.6
            elif "cause" in sec or "failure" in sec: c["score"] *= 1.2
            elif "[table]" in sec: c["score"] *= 1.15
            
        chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        selected, spec_counts = [], {}
        apply_diversity = len(set(c.get("spec_number", "") for c in chunks[:20])) > 2
        
        for c in chunks:
            spec = c.get("spec_number", "unknown")
            if apply_diversity and spec_counts.get(spec, 0) >= 6: continue
            selected.append(c)
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
            if len(selected) >= 20: break
            
        return {"retrieved_chunks": selected}

    # --- Node: Evaluator (CRAG) ---
    def _evaluator_node(self, state: RAGState) -> dict:
        chunks = state.get("retrieved_chunks", [])
        valid_scores = [c.get("score", 0) for c in chunks]
        
        if valid_scores and max(valid_scores) >= 0.45:
            return {"eval_result": "correct", "refined_query": state["user_query"]}
        if not chunks:
            return {"eval_result": "incorrect", "refined_query": state["user_query"]}

        summary = "\n".join([f"[{c.get('section_path')}] {c['chunk_text'][:200]}" for c in chunks[:5]])
        system_prompt = """Evaluate retrieval quality (correct/ambiguous/incorrect) based on if the chunks answer the user query.
Return ONLY JSON: {"evaluation": "correct", "refined_query": null, "reason": "..."}"""

        try:
            resp = self.flash_client.chat.completions.create(
                model=self.flash_model,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": f"Query: {state['user_query']}\n\nChunks:\n{summary}"}
                ],
                response_format={"type": "json_object"}
            )
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.choices[0].message.content.strip())
            result = json.loads(raw)
            return {"eval_result": result.get("evaluation", "correct"), "refined_query": result.get("refined_query") or state["user_query"]}
        except Exception:
            return {"eval_result": "correct", "refined_query": state["user_query"]}

    # --- Node: Refiner ---
    def _refiner_node(self, state: RAGState) -> dict:
        return {"sub_queries": [state.get("refined_query", state["user_query"]), state["user_query"]], "retrieved_chunks": []}

    # --- Node: Generator ---
    def _generator_node(self, state: RAGState) -> dict:
        logger.info("  [generator] Producing expert-level answer (Heavy Model on Groq)...")
        relevant_chunks = [c for c in state["retrieved_chunks"] if c.get("score", 0) >= 0.20]

        if not relevant_chunks:
            return {"answer": "## ⚠️ Document Not Indexed\nInformation not found.", "citations": [], "confidence": 0.0, "route": "done"}

        # SQL history query removed: Qdrant does not natively support complex relational joins on non-vector schemas.
        # Ensure metadata contains history if required, or query a parallel relational DB.
        history = ""

        # Route and Format context via ContextRouter
        context = self.router.route_and_format(relevant_chunks, history)

        prompt = f"""You are a principal 3GPP standards architect. First, write a detailed text answer based strictly on the provided context.
Use ## headers, tables, and exact spec citations [Source N]. 
Strictly at the very end of your response, append a JSON block: {{"confidence": 0.0-1.0}}

CONTEXT:
{context}

QUESTION: {state["user_query"]}"""

        resp = self.generator_client.chat.completions.create(
            model=self.generator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        answer_raw = resp.choices[0].message.content

        confidence = 0.5
        json_match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', answer_raw)
        if json_match:
            try:
                meta = json.loads(json_match.group())
                confidence = float(meta.get("confidence", 0.5))
                answer_raw = answer_raw[:json_match.start()].strip()
            except Exception: pass

        citations = [{"spec": c.get("spec_number"), "section": c.get("section_path"), "score": round(c.get("score", 0), 3)} for c in relevant_chunks[:8]]
        
        return {"answer": answer_raw, "citations": citations, "confidence": confidence, "route": "done"}

    # --- Graph Construction ---
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("isrel", self._isrel_node)
        graph.add_node("reranker", self._reranker_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("refiner", self._refiner_node)
        graph.add_node("generator", self._generator_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "isrel")
        graph.add_edge("isrel", "reranker")
        graph.add_edge("reranker", "evaluator")

        def crag_router(state: RAGState):
            if state.get("eval_result") == "correct": return "generate"
            if state.get("eval_result") == "ambiguous" and state.get("retrieval_hops", 0) < 2: return "refine"
            return "generate"

        graph.add_conditional_edges("evaluator", crag_router, {"generate": "generator", "refine": "refiner"})
        graph.add_edge("refiner", "retriever")
        
        def final_router(state: RAGState): return END
        graph.add_conditional_edges("generator", final_router, {END: END})

        return graph.compile()

    def ask(self, query: str, spec: str = None, release: str = None) -> dict:
        initial_state = {
            "user_query": query, "spec_filter": spec, "release_filter": release,
            "sub_queries": [], "retrieved_chunks": [], "retrieval_hops": 0
        }
        return self.graph.invoke(initial_state)

# ==========================================
# CLI Execution
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    engine = RAGQueryEngine()
    
    print("\n--- 3GPP Local RAG Engine ---")
    query = "What are the RRC states in 5G NR and the transitions between them?"
    result = engine.ask(query)
    
    print("\n[ANSWER]")
    print(result["answer"])
    print(f"\n[CONFIDENCE]: {result['confidence']:.0%} | [HOPS]: {result.get('retrieval_hops')}")
    print("[CITATIONS]")
    for i, c in enumerate(result["citations"], 1):
        print(f"  [{i}] TS {c['spec']} §{c['section']} (Score: {c['score']})")
