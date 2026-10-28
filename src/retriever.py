"""
Intelligent retriever module supporting dense vector search, sparse BM25 search,
hybrid fusion ranking, and score threshold filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from src.utils import format_source_citation, get_logger
from src.vector_store import VectorStoreManager

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with associated relevance score and citation."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    chunk_id: str = ""
    citation: str = ""

    def __post_init__(self):
        if not self.citation:
            self.citation = format_source_citation(self.metadata)


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


class BM25SearchEngine:
    """Sparse keyword search using Okapi BM25 algorithm with pure Python implementation."""

    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.\@]{2,}\b", text.lower())
        return [w for w in words if w not in STOP_WORDS]

    def _build_index(self):
        if not self.corpus:
            return

        import math
        from collections import Counter

        num_docs = len(self.corpus)
        df: Dict[str, int] = {}
        total_len = 0

        for item in self.corpus:
            tokens = self._tokenize(item["document"])
            token_counts = Counter(tokens)
            self.doc_freqs.append(token_counts)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)

            for term in token_counts.keys():
                df[term] = df.get(term, 0) + 1

        self.avgdl = total_len / num_docs if num_docs > 0 else 1.0

        # Calculate Okapi BM25 IDF
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """Returns list of (index, normalized_score) sorted by BM25 relevance."""
        if not self.corpus:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        raw_scores: List[float] = [0.0] * len(self.corpus)

        for term in tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]

            for idx, doc_counts in enumerate(self.doc_freqs):
                freq = doc_counts.get(term, 0)
                if freq == 0:
                    continue
                d_len = self.doc_len[idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (d_len / (self.avgdl or 1.0)))
                raw_scores[idx] += idf_val * (numerator / denominator)

        max_s = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0
        scored = [(idx, float(s / max_s)) for idx, s in enumerate(raw_scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class Retriever:
    """
    RAG Retriever orchestrating dense vector similarity search,
    sparse BM25 search, hybrid rank fusion, and confidence thresholding.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        enable_hybrid_search: Optional[bool] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k if top_k is not None else settings.top_k
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        self.enable_hybrid_search = (
            enable_hybrid_search
            if enable_hybrid_search is not None
            else settings.enable_hybrid_search
        )
        self.vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        enable_hybrid: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes query retrieval against the document vector database.
        
        Args:
            query: The user query string.
            top_k: Optional override for number of chunks to return.
            similarity_threshold: Optional override for minimum similarity filter.
            enable_hybrid: Optional override to toggle hybrid search.
            
        Returns:
            List of RetrievedChunk instances meeting the criteria.
        """
        k = top_k if top_k is not None else self.top_k
        thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        use_hybrid = (
            enable_hybrid if enable_hybrid is not None else self.enable_hybrid_search
        )

        total_indexed = self.vector_store.count()
        if total_indexed == 0:
            logger.warning("Retriever called but vector store is empty.")
            return []

        # Step 1: Dense Vector Retrieval
        dense_hits = self.vector_store.similarity_search_with_score(query=query, top_k=k * 2)

        if not use_hybrid:
            # Dense-only mode
            results: List[RetrievedChunk] = []
            for doc, meta, score, cid in dense_hits:
                if score >= thresh:
                    results.append(
                        RetrievedChunk(
                            content=doc,
                            metadata=meta,
                            score=round(score, 4),
                            chunk_id=cid,
                        )
                    )
            return results[:k]

        # Step 2: Hybrid Search with BM25 Rank Fusion
        corpus = self.vector_store.get_all_chunks()
        bm25_engine = BM25SearchEngine(corpus)
        bm25_hits = bm25_engine.search(query=query, top_k=k * 2)

        # Merge and fuse scores
        chunk_score_map: Dict[str, Dict[str, Any]] = {}

        # Incorporate dense scores
        for doc, meta, score, cid in dense_hits:
            chunk_score_map[cid] = {
                "content": doc,
                "metadata": meta,
                "vector_score": score,
                "bm25_score": 0.0,
                "chunk_id": cid,
            }

        # Incorporate sparse BM25 scores
        for corpus_idx, bm_score in bm25_hits:
            item = corpus[corpus_idx]
            cid = item["id"]
            if cid in chunk_score_map:
                chunk_score_map[cid]["bm25_score"] = bm_score
            else:
                chunk_score_map[cid] = {
                    "content": item["document"],
                    "metadata": item["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": bm_score,
                    "chunk_id": cid,
                }

        # Calculate weighted hybrid score
        hybrid_results: List[RetrievedChunk] = []
        for cid, data in chunk_score_map.items():
            v_score = data["vector_score"]
            b_score = data["bm25_score"]
            combined = (v_score * self.vector_weight) + (b_score * self.bm25_weight)
            # Ensure strong signal in either modality is boosted
            final_score = max(combined, v_score * 0.85, b_score * 0.85)

            # Apply threshold filtering
            if final_score >= thresh:
                hybrid_results.append(
                    RetrievedChunk(
                        content=data["content"],
                        metadata=data["metadata"],
                        score=round(final_score, 4),
                        chunk_id=cid,
                    )
                )

        # Sort descending by hybrid relevance score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        logger.info("Hybrid retrieval returned %d chunks (out of %d candidates)", len(hybrid_results[:k]), len(chunk_score_map))
        return hybrid_results[:k]
