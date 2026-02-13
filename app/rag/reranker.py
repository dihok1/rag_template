"""Cross-encoder reranker: re-rank candidates by (query, chunk) relevance."""
from typing import Any

from app.config import RERANK_API_KEY, RERANK_API_URL, RERANKER_MODEL


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Re-rank candidates by relevance to query. Returns top_k items with optional rerank_score.
    Uses RERANK_API_URL if set, else local sentence-transformers cross-encoder.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k and not RERANK_API_URL:
        return candidates[:top_k]
    if RERANK_API_URL:
        return _rerank_api(query, candidates, top_k)
    return _rerank_local(query, candidates, top_k)


def _rerank_api(
    query: str, candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Rerank via external API (e.g. Cohere, Jina). Expects POST with query + documents, returns scores or order."""
    import urllib.request
    import json as json_mod

    texts = [c.get("text", "") for c in candidates]
    body = {"query": query, "documents": texts}
    data = json_mod.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        RERANK_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RERANK_API_KEY}" if RERANK_API_KEY else "",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json_mod.loads(resp.read().decode())
    except Exception:
        return candidates[:top_k]
    # Common API shapes: { "results": [ {"index": 0, "relevance_score": 0.9}, ... ] } or { "results": [ {"document": {...}, "score": ...} ] }
    results = out.get("results", out.get("data", []))
    if not results:
        return candidates[:top_k]
    indexed = []
    for r in results:
        idx = r.get("index", r.get("document", {}).get("index", -1))
        if isinstance(idx, dict):
            idx = idx.get("index", -1)
        score = r.get("relevance_score", r.get("score", 0.0))
        indexed.append((idx, score))
    indexed.sort(key=lambda x: -x[1])
    seen = set()
    ordered = []
    for idx, score in indexed:
        if idx < 0 or idx >= len(candidates) or idx in seen:
            continue
        seen.add(idx)
        item = dict(candidates[idx])
        item["score"] = score
        item["rerank_score"] = score
        ordered.append(item)
        if len(ordered) >= top_k:
            break
    return ordered or candidates[:top_k]


def _rerank_local(
    query: str, candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Rerank using sentence-transformers cross-encoder."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return candidates[:top_k]
    model = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs)
    indexed = [(i, float(s)) for i, s in enumerate(scores)]
    indexed.sort(key=lambda x: -x[1])
    out = []
    for i, score in indexed[:top_k]:
        item = dict(candidates[i])
        item["score"] = score
        item["rerank_score"] = score
        out.append(item)
    return out
