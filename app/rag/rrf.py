"""Reciprocal Rank Fusion (RRF) for merging multiple ranked lists."""
from typing import Any

# Default RRF constant (k=60 is standard)
RRF_K = 60


def rrf_merge(
    ranked_lists: list[list[dict[str, Any]]],
    metadata: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Each item in ranked_lists should be a list of dicts with "chunk_id" (index into metadata).
    Returns list of {text, source_path, score} sorted by RRF score descending.
    """
    scores: dict[int, float] = {}
    for rank_list in ranked_lists:
        for rank_1based, item in enumerate(rank_list, start=1):
            cid = item.get("chunk_id", -1)
            if cid < 0 or cid >= len(metadata):
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_1based)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for cid, rrf_score in ordered:
        meta = metadata[cid]
        out.append({
            "chunk_id": cid,
            "text": meta["text"],
            "source_path": meta["source_path"],
            "score": rrf_score,
        })
    return out
