"""Load vector index and search for relevant chunks. Supports hybrid (BM25+vector) and RRF."""
import re
import json
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

try:
    import faiss
except ImportError:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.config import (
    HYBRID_FETCH_K,
    HYBRID_SEARCH_ENABLED,
    INDEX_PATH,
    MIN_RELEVANCE_SCORE,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_VARIANTS,
    RERANKER_ENABLED,
    RERANKER_TOP_N,
    TOP_K,
)
from app.rag.text_cleaning import normalize_for_embedding
from app.rag.rrf import rrf_merge, RRF_K


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization for BM25 (alphanumeric + underscores)."""
    return re.findall(r"\w+", text.lower())


def _get_embedding(client: OpenAI, text: str, model: str) -> list[float]:
    normalized = normalize_for_embedding(text)
    resp = client.embeddings.create(input=[normalized], model=model)
    return resp.data[0].embedding


class RAGRetriever:
    """Loads FAISS index + metadata and provides search(query, top_k). Optional: BM25 hybrid, RRF, reranker, query expansion."""

    def __init__(
        self,
        index_path: Path | None = None,
        openai_api_key: str | None = None,
        openai_api_base: str | None = None,
        embedding_model: str | None = None,
    ):
        self.index_path = (index_path or INDEX_PATH).resolve()
        self.api_key = openai_api_key or OPENAI_API_KEY
        self.api_base = openai_api_base or OPENAI_API_BASE
        self.embedding_model = embedding_model or OPENAI_EMBEDDING_MODEL
        self._index: Any = None
        self._metadata: list[dict[str, Any]] = []
        self._client: OpenAI | None = None
        self._bm25: Any = None

    def load(self) -> None:
        if faiss is None:
            raise RuntimeError("faiss-cpu is required. Install: pip install faiss-cpu")
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"
        if not index_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(
                f"Index not found at {self.index_path}. Run index builder first."
            )
        self._index = faiss.read_index(str(index_file))
        self._metadata = json.loads(meta_file.read_text(encoding="utf-8"))["chunks"]
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        if HYBRID_SEARCH_ENABLED and BM25Okapi is not None:
            corpus = [c["text"] for c in self._metadata]
            tokenized = [_tokenize(t) for t in corpus]
            self._bm25 = BM25Okapi(tokenized)

    def _vector_candidates(
        self, query: str, fetch_k: int, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """Return list of {chunk_id, text, source_path, score} from vector search."""
        q = _get_embedding(self._client, query, self.embedding_model)
        qv = np.array([q], dtype=np.float32)
        faiss.normalize_L2(qv)
        scores, indices = self._index.search(qv, fetch_k)
        threshold = min_score if min_score is not None else MIN_RELEVANCE_SCORE
        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or (threshold > 0 and float(score) < threshold):
                continue
            meta = self._metadata[idx]
            out.append({
                "chunk_id": int(idx),
                "text": meta["text"],
                "source_path": meta["source_path"],
                "score": float(score),
            })
        return out

    def _bm25_candidates(self, query: str, fetch_k: int) -> list[dict[str, Any]]:
        """Return list of {chunk_id, text, source_path, score} from BM25 (score = BM25 score)."""
        if self._bm25 is None or not self._metadata:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:fetch_k]
        out = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            meta = self._metadata[idx]
            out.append({
                "chunk_id": int(idx),
                "text": meta["text"],
                "source_path": meta["source_path"],
                "score": float(scores[idx]),
            })
        return out

    def _retrieve_one_query(
        self, query: str, fetch_k: int, min_score: float | None = None
    ) -> list[dict[str, Any]]:
        """Single-query retrieval: vector only or hybrid (vector + BM25 + RRF). Returns list with chunk_id."""
        if HYBRID_SEARCH_ENABLED and self._bm25 is not None:
            vec_list = self._vector_candidates(query, fetch_k, min_score=None)
            bm25_list = self._bm25_candidates(query, fetch_k)
            if not vec_list and not bm25_list:
                return []
            if not vec_list:
                return bm25_list[:fetch_k]
            if not bm25_list:
                return vec_list[:fetch_k]
            merged = rrf_merge([vec_list, bm25_list], self._metadata, k=RRF_K)
            return merged
        # Vector only (original behaviour)
        return self._vector_candidates(query, fetch_k, min_score)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return list of {text, source_path, score} for top_k nearest chunks.
        Uses query expansion, hybrid search, and reranker when enabled in config.
        """
        k = top_k if top_k is not None else TOP_K
        threshold = min_score if min_score is not None else MIN_RELEVANCE_SCORE
        if self._index is None or self._client is None:
            self.load()
        fetch_k = HYBRID_FETCH_K if HYBRID_SEARCH_ENABLED else min(k * 3, len(self._metadata))
        if threshold > 0 and not HYBRID_SEARCH_ENABLED:
            fetch_k = min(fetch_k, len(self._metadata))

        if QUERY_EXPANSION_ENABLED:
            try:
                from app.rag.query_expansion import expand_query_multi
                queries = expand_query_multi(query, num_variants=QUERY_EXPANSION_VARIANTS)
            except Exception:
                queries = [query]
            ranked_lists = []
            for q in queries:
                one = self._retrieve_one_query(q, fetch_k, min_score=None)
                ranked_lists.append(one)
            if not ranked_lists or all(not r for r in ranked_lists):
                return []
            # RRF over expanded queries
            merged = rrf_merge(ranked_lists, self._metadata, k=RRF_K)
            candidates = [{"text": c["text"], "source_path": c["source_path"], "score": c["score"]} for c in merged]
        else:
            raw = self._retrieve_one_query(query, fetch_k, threshold)
            if not raw:
                return []
            # _retrieve_one_query may return items with chunk_id; for single-query path we want {text, source_path, score}
            candidates = [
                {"text": r["text"], "source_path": r["source_path"], "score": r["score"]}
                for r in raw
            ]

        if RERANKER_ENABLED:
            from app.rag.reranker import rerank
            n = min(RERANKER_TOP_N, len(candidates))
            to_rerank = candidates[:n]
            candidates = rerank(query, to_rerank, top_k=k)
            return candidates[:k]

        return candidates[:k]
