"""Weight-score hybrid retrieval: dense cosine similarity + BM25 lexical scoring.

Dense embeddings (agent.retrieval.retrieve) rank purely on semantic
similarity, which struggles to separate near-duplicate chunks that repeat
the same subject name (e.g. many chunks about the same ATT&CK technique)
and can miss exact terminology a query didn't happen to paraphrase. BM25
(agent.retrieval.bm25) catches those exact-term matches but misses
paraphrases. Combining both catches more of what either misses alone.

Each signal is fetched over its own top-FETCH_POOL_SIZE candidates,
independently min-max normalized to [0, 1] so their differently-scaled
scores contribute comparably, then combined as:

    hybrid_score = alpha * dense_norm + (1 - alpha) * bm25_norm

A chunk found by only one signal contributes 0 for the other half, rather
than being penalized further for being "missing". The two fetches overlap
only modestly on this corpus (~20%), so their union usually well exceeds
FETCH_POOL_SIZE.

A third signal, agent.retrieval.bm25.exact_id_hits, scans the whole corpus
(not just the fetched pool) for chunks containing every ATT&CK ID the
query names. It exists because pooling itself is the failure mode for this
corpus's many near-duplicate "how does X use Y" documents: when hundreds of
chunks share the same boilerplate around a technique, the one correct
chunk can rank outside both signals' fetch and never reach reranking at
all. Any chunk it finds is force-scored to 1.0 (the max a normalized score
can take), so it's always among the top by hybrid_score and survives every
cut below; rerank still decides the final order.

By default the result is then reranked (agent.retrieval.rerank): hybrid
scoring alone still struggles when many chunks share the same subject
boilerplate and differ only in a few words (e.g. several chunks about the
same ATT&CK technique), which is exactly what a cross-encoder is good at
separating. Reranking the full fetched union directly (with no
intermediate cut) was measured to cost no more latency than reranking a
much smaller pool -- the cross-encoder scores the whole pool in one
batched forward pass, so fixed overhead dominates at this corpus's scale
-- but an unbounded union (up to 2 * FETCH_POOL_SIZE) makes reranker cost
unpredictable as the corpus grows. RERANK_POOL_SIZE instead bounds it to a
fixed count, cut from the union by hybrid_score before reranking. This
costs a small amount of recall at the alpha extremes (BM25-only or
dense-only: hybrid_score's coarse ranking occasionally discards a
candidate the reranker would have surfaced) but is exactly as accurate as
reranking the full union at a balanced alpha (0.25-0.5), which is also
where HYBRID_ALPHA defaults.
"""

import os

from agent.retrieval import bm25 as bm25_index
from agent.retrieval import rerank as rerank_index
from agent.retrieval.retrieve import search as dense_search


DEFAULT_ALPHA = float(os.getenv("HYBRID_ALPHA") or 0.5)
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 20
# Per-signal (dense, BM25) fetch count when reranking.
FETCH_POOL_SIZE = int(os.getenv("FETCH_POOL_SIZE") or 100)
# Cap on how many of the fetched union (by hybrid_score) reach the cross-encoder.
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE") or 100)


def _normalize(scores: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Min-max scale a score map to [0, 1]; a flat set of scores all maps to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def search(
    query: str, collection: str, k: int = 5, alpha: float = DEFAULT_ALPHA, rerank: bool = True
) -> list[dict]:
    """Return the top-k chunks by combined dense + BM25 score, then reranked.

    Args:
        query: Search text.
        collection: Qdrant collection to search.
        k: Number of chunks to return.
        alpha: Weight on the dense signal; (1 - alpha) weights BM25.
            1.0 is dense-only, 0.0 is BM25-only. Defaults to the
            HYBRID_ALPHA env var, or 0.5.
        rerank: Cross-encoder rerank the candidate pool before truncating to
            k (see agent.retrieval.rerank). Fetches FETCH_POOL_SIZE
            candidates from each signal, then caps their union to
            RERANK_POOL_SIZE by hybrid_score before reranking, rather than
            just re-sorting what hybrid scoring already picked.

    Returns:
        Chunk payloads with "score" (the score that decided the order —
        rerank_score if rerank=True, else hybrid_score) plus "hybrid_score",
        "dense_score", "bm25_score" (each normalized to [0, 1]; 0.0 if that
        signal didn't surface the chunk in its own candidate pool at all),
        and "rerank_score" if rerank=True.
    """
    candidate_k = FETCH_POOL_SIZE if rerank else max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    dense_hits = dense_search(query, collection, k=candidate_k)
    bm25_hits = bm25_index.search(query, collection, k=candidate_k)
    exact_hits = bm25_index.exact_id_hits(query, collection)

    chunks: dict[tuple[str, int], dict] = {}
    dense_scores: dict[tuple[str, int], float] = {}
    bm25_scores: dict[tuple[str, int], float] = {}
    exact_keys: set[tuple[str, int]] = set()

    for hit in dense_hits:
        key = (hit["source"], hit["chunk_index"])
        chunks[key] = hit
        dense_scores[key] = hit["score"]
    for hit in bm25_hits:
        key = (hit["source"], hit["chunk_index"])
        chunks.setdefault(key, hit)
        bm25_scores[key] = hit["score"]
    for hit in exact_hits:
        key = (hit["source"], hit["chunk_index"])
        chunks.setdefault(key, hit)
        exact_keys.add(key)

    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(bm25_scores)

    def hybrid_score(key: tuple[str, int]) -> float:
        if key in exact_keys:
            return 1.0
        return alpha * dense_norm.get(key, 0.0) + (1 - alpha) * bm25_norm.get(key, 0.0)

    scored_chunks = [
        {**chunks[key], "hybrid_score": hybrid_score(key), "dense_score": dense_norm.get(key, 0.0),
         "bm25_score": bm25_norm.get(key, 0.0)}
        for key in chunks
    ]

    if not rerank:
        for chunk in scored_chunks:
            chunk["score"] = chunk["hybrid_score"]
        return sorted(scored_chunks, key=lambda chunk: chunk["score"], reverse=True)[:k]

    capped = sorted(scored_chunks, key=lambda chunk: chunk["hybrid_score"], reverse=True)[:RERANK_POOL_SIZE]
    return rerank_index.rerank(query, capped, k)
