import os
import json
import re
import argparse
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Import the RAG engine you built in query.py
from src.core.generation import RAGQueryEngine

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

# ==========================================
# Default Golden Q&A Dataset
# ==========================================
DEFAULT_GOLDEN_QA = [
    {
        "question": "What are the three RRC states in 5G NR?",
        "ground_truth": "5G NR defines three RRC states: RRC_IDLE, RRC_INACTIVE, and RRC_CONNECTED. RRC_INACTIVE is a new state introduced in NR that was not present in LTE."
    },
    {
        "question": "What protocol does the UE use to request a paging response in RRC_IDLE?",
        "ground_truth": "In RRC_IDLE, the UE monitors the paging channel and responds via a random access procedure on PRACH to initiate RRC connection establishment."
    },
    {
        "question": "What is the maximum number of HARQ processes in NR downlink?",
        "ground_truth": "NR supports up to 16 HARQ processes in the downlink, compared to 8 in LTE, providing greater scheduling flexibility."
    }
]

# ==========================================
# RAG Evaluator Class
# ==========================================
class RAGEvaluator:
    """
    Evaluation harness for the RAG pipeline using LLM-as-a-Judge.
    Future Extensions: To add new metrics (e.g., 'context_precision'), 
    simply update the system prompt JSON schema and the dictionary parsing in _score_with_llm.
    """
    def __init__(self, rag_engine: RAGQueryEngine):
        self.engine = rag_engine
        
        # Connect to local Ollama for evaluation
        self.eval_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "sk-dummy-key")
        )
        
        # Use a strong local model as the judge (7B is recommended for accurate scoring)
        self.judge_model = os.getenv("LLM_MODEL_ID", "qwen2.5-coder:7b")

    def _score_with_llm(self, question: str, answer: str, context: str, ground_truth: str) -> Dict[str, Any]:
        """Internal method: Calls the local LLM to grade the answer across multiple dimensions."""
        
        system_prompt = """You are an objective evaluator of a RAG system for 3GPP telecom standards and C++ systems.
Score the system's answer against the ground truth on three dimensions, each from 0.0 to 1.0:
- faithfulness: does the answer contain only claims supported by the provided context? (0.0 if hallucinated)
- relevancy: how directly does the answer address the question? (0.0 if off-topic)
- completeness: how fully does the answer cover the ground truth? (0.0 if missing key facts)

Return ONLY valid JSON in this exact format: 
{"faithfulness": 0.0, "relevancy": 0.0, "completeness": 0.0, "rationale": "brief explanation"}"""

        user_prompt = f"""Question: {question}
Ground truth: {ground_truth}

Retrieved context:
{context}

System answer:
{answer}

Score the answer:"""

        try:
            response = self.eval_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Strict, deterministic scoring
            )
            
            raw_output = response.choices[0].message.content.strip()
            raw_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output, flags=re.IGNORECASE).strip()
            
            return json.loads(raw_output)
            
        except Exception as e:
            logger.error(f"Evaluation LLM failed: {e}")
            return {"faithfulness": 0.0, "relevancy": 0.0, "completeness": 0.0, "rationale": f"Eval Error: {e}"}

    def evaluate_single(self, qa_pair: Dict[str, str]) -> Dict[str, Any]:
        """Executes a full RAG query and evaluates the result."""
        question = qa_pair["question"]
        ground_truth = qa_pair["ground_truth"]
        
        # 1. Generate the answer using the RAG Engine
        start_time = datetime.now()
        result = self.engine.ask(question)
        latency = (datetime.now() - start_time).total_seconds()
        
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        
        # 2. Format context chunks to pass to the judge
        # (Pass only section names to save token overhead on the evaluator)
        context_str = "\n".join([f"- TS {c.get('spec')} §{c.get('section')}" for c in citations])
        
        # 3. Grade the output
        scores = self._score_with_llm(question, answer, context_str, ground_truth)
        
        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "latency_seconds": round(latency, 2),
            "hops": result.get("retrieval_hops", 1),
            "faithfulness": scores.get("faithfulness", 0.0),
            "relevancy": scores.get("relevancy", 0.0),
            "completeness": scores.get("completeness", 0.0),
            "rationale": scores.get("rationale", "")
        }

    def evaluate_batch(self, qa_set: List[Dict[str, str]]) -> Dict[str, Any]:
        """Runs evaluation over a dataset and calculates aggregate metrics."""
        logger.info(f"Starting evaluation of {len(qa_set)} queries...")
        
        results = []
        for i, qa in enumerate(qa_set, 1):
            logger.info(f"[{i}/{len(qa_set)}] Evaluating: {qa['question'][:60]}...")
            
            eval_result = self.evaluate_single(qa)
            results.append(eval_result)
            
            logger.info(
                f"  ↳ Faithfulness: {eval_result['faithfulness']:.2f} | "
                f"Relevancy: {eval_result['relevancy']:.2f} | "
                f"Completeness: {eval_result['completeness']:.2f}"
            )

        # Aggregate the scores
        aggregate = {
            "faithfulness": statistics.mean([r["faithfulness"] for r in results]) if results else 0.0,
            "relevancy": statistics.mean([r["relevancy"] for r in results]) if results else 0.0,
            "completeness": statistics.mean([r["completeness"] for r in results]) if results else 0.0,
            "avg_latency": statistics.mean([r["latency_seconds"] for r in results]) if results else 0.0
        }
        
        return {
            "aggregate": aggregate,
            "results": results,
            "evaluated_at": datetime.utcnow().isoformat()
        }

    def save_report(self, report: Dict[str, Any], filepath: str = None) -> str:
        """Saves the JSON evaluation report to disk."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"eval_report_{timestamp}.json"
            
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return filepath

# ==========================================
# CLI Orchestration
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Evaluate the local 3GPP RAG Pipeline")
    parser.add_argument("--golden", type=str, help="Path to custom JSON file containing Q&A pairs")
    parser.add_argument("--single", type=str, help="Evaluate a single question interactively")
    args = parser.parse_args()

    # Initialize Engine and Evaluator
    logger.info("Spinning up RAG Query Engine...")
    engine = RAGQueryEngine()
    evaluator = RAGEvaluator(engine)

    if args.single:
        # Interactive single run (Requires you to provide a ground truth for scoring)
        ground_truth = input("\nEnter expected Ground Truth to score against: ")
        result = evaluator.evaluate_single({"question": args.single, "ground_truth": ground_truth})
        
        print("\n=== EVALUATION RESULT ===")
        print(f"Answer Preview: {result['answer_preview']}")
        print(f"Faithfulness: {result['faithfulness']}")
        print(f"Relevancy:    {result['relevancy']}")
        print(f"Completeness: {result['completeness']}")
        print(f"Rationale:    {result['rationale']}")
        
    else:
        # Batch evaluation
        qa_set = DEFAULT_GOLDEN_QA
        if args.golden and os.path.exists(args.golden):
            with open(args.golden, "r") as f:
                qa_set = json.load(f)
                
        report = evaluator.evaluate_batch(qa_set)
        agg = report["aggregate"]
        
        print(f"\n{'-'*50}")
        print("EVALUATION REPORT")
        print(f"{'-'*50}")
        print(f" Faithfulness : {agg['faithfulness']:.2%}")
        print(f" Relevancy    : {agg['relevancy']:.2%}")
        print(f" Completeness : {agg['completeness']:.2%}")
        print(f" Avg Latency  : {agg['avg_latency']:.2f} seconds")
        print(f"{'-'*50}")
        
        saved_path = evaluator.save_report(report)
        print(f"Full JSON report saved to: {saved_path}")
        
        # CI/CD Threshold Warnings
        if agg["faithfulness"] < 0.7:
            print("⚠ ALERT: Faithfulness below 0.70 threshold. Review pipeline constraints.")