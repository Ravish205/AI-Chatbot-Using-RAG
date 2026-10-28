"""
Automated evaluation harness for RAG retrieval quality, citation accuracy,
out-of-domain rejection, and latency profiling.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from src.document_loader import DocumentLoader
from src.embeddings import DeterministicMockEmbeddings, EmbeddingFactory
from src.rag_chain import LLMFactory, MockLLMClient, OllamaLLMClient, RAGChain
from src.retriever import Retriever
from src.text_splitter import TextSplitter
from src.utils import get_logger
from src.vector_store import VectorStoreManager

logger = get_logger("evaluation")


def run_evaluation(use_local_models: bool = False):
    """Runs end-to-end benchmark evaluation against the sample dataset."""
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "sample_documents"
    dataset_file = base_dir / "eval_dataset.json"

    if not dataset_file.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    print("=" * 80)
    print(" RAG BENCHMARK & EVALUATION HARNESS")
    print("=" * 80)

    # Setup embeddings & LLM
    if use_local_models:
        print("[*] Initializing local embeddings and LLM...")
        embeddings = EmbeddingFactory.get_embeddings(settings.embedding_provider)
        llm = LLMFactory.create(settings.llm_provider)
    else:
        print("[*] Initializing deterministic mock embeddings and LLM for fast test...")
        embeddings = DeterministicMockEmbeddings(dim=64)
        llm = MockLLMClient()

    # Create temporary vector store collection
    vector_store = VectorStoreManager(
        collection_name="eval_benchmark_collection",
        embedding_provider=embeddings,
    )
    vector_store.reset_collection()

    # Index sample documents
    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=600, chunk_overlap=100)

    print(f"[*] Ingesting sample documents from {docs_dir}...")
    indexed_files = 0
    total_chunks = 0

    for doc_path in sorted(docs_dir.glob("*.txt")):
        docs = loader.load_from_path(doc_path)
        chunks = splitter.split_documents(docs)
        vector_store.add_chunks(chunks)
        indexed_files += 1
        total_chunks += len(chunks)

    print(f"    Indexed {indexed_files} files ({total_chunks} total chunks).")
    print("-" * 80)

    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
        similarity_threshold=0.25,
        enable_hybrid_search=True,
    )
    chain = RAGChain(retriever=retriever, llm_client=llm)

    # Run benchmark metrics
    retrieval_hits = 0
    retrieval_total = 0
    rejection_hits = 0
    rejection_total = 0
    citations_present = 0
    total_queries = len(eval_cases)
    total_latency = 0.0

    results: List[Dict[str, Any]] = []

    for case in eval_cases:
        cid = case["id"]
        q = case["question"]
        target = case["target_document"]
        cat = case["category"]

        t0 = time.time()
        resp = chain.answer_question(q)
        lat = time.time() - t0
        total_latency += lat

        # Check retrieval relevance
        retrieved_sources = [s.get("metadata", {}).get("source_filename", "") for s in resp.sources]
        is_retrieval_hit = target in retrieved_sources if target != "none" else (len(resp.sources) == 0)

        # Check citation presence
        has_citation = len(resp.sources) > 0 and bool(resp.sources[0].get("citation"))

        # Check out-of-domain rejection
        is_rejection_success = False
        if cat == "unanswerable_out_of_domain":
            rejection_total += 1
            if "could not find information" in resp.answer.lower() or "not have enough information" in resp.answer.lower():
                is_rejection_success = True
                rejection_hits += 1
        elif cat == "factual_answerable":
            retrieval_total += 1
            if target in retrieved_sources:
                retrieval_hits += 1
            if has_citation:
                citations_present += 1

        results.append({
            "id": cid,
            "category": cat,
            "latency": round(lat, 3),
            "retrieval_hit": is_retrieval_hit,
            "citations_present": has_citation,
            "sources_count": len(resp.sources),
        })

        status_icon = "✅" if (is_retrieval_hit or is_rejection_success) else "⚠️"
        print(f" {status_icon} [{cid}] {q[:55]}... ({lat:.2f}s | Sources: {len(resp.sources)})")

    # Summary Report
    print("=" * 80)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 80)
    hit_rate = (retrieval_hits / retrieval_total * 100) if retrieval_total else 100.0
    rej_rate = (rejection_hits / rejection_total * 100) if rejection_total else 100.0
    cite_rate = (citations_present / retrieval_total * 100) if retrieval_total else 100.0
    avg_latency = total_latency / total_queries if total_queries else 0.0

    print(f"Total Evaluated Questions:      {total_queries}")
    print(f"Factual Retrieval Hit Rate:     {hit_rate:.1f}% ({retrieval_hits}/{retrieval_total})")
    print(f"Hallucination Rejection Rate:   {rej_rate:.1f}% ({rejection_hits}/{rejection_total})")
    print(f"Citation Attribution Rate:      {cite_rate:.1f}% ({citations_present}/{retrieval_total})")
    print(f"Average Query Latency:          {avg_latency:.3f} seconds")
    print("=" * 80)

    # Cleanup evaluation collection
    vector_store.reset_collection()


if __name__ == "__main__":
    import sys
    use_local = "--local" in sys.argv
    run_evaluation(use_local_models=use_local)
