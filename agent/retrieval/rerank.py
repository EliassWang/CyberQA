"""Cross-encoder reranking over a hybrid retrieval candidate pool.

Dense cosine similarity and BM25 both compress a chunk into a signal
compared against the query at a distance (a single vector, or independent
term weights) — fine for finding the right topic, but weak at separating
near-duplicate chunks about the same subject that differ in only a few
words. A cross-encoder instead jointly attends over the full query and
chunk text together, so it can catch that finer-grained distinction. It's
too slow to run over a whole collection, so it only reranks the (much
smaller) candidate pool hybrid search already narrowed things down to.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.device import get_device


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME, device=get_device())


def warm_up() -> None:
    """Load (and on first-ever use, download) the cross-encoder now.

    Reranking is on by default, so every query pays this cost the first
    time it's needed; call this during startup instead so it happens
    under a "starting up" spinner rather than silently during the user's
    first question.
    """
    _model()


def rerank(query: str, chunks: list[dict], k: int) -> list[dict]:
    """Re-score `chunks` against `query` with a cross-encoder and return the top-k.

    Args:
        query: Search text.
        chunks: Candidate chunk payloads (must include "text"); any existing
            "score"/"dense_score"/"bm25_score" fields are preserved.
        k: Number of chunks to return.

    Returns:
        The top-k chunks by cross-encoder score, with "rerank_score" added
        and "score" overwritten to match it (rerank_score is now what
        decided the order).
    """
    if not chunks:
        return []
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = _model().predict(pairs)
    scored = [{**chunk, "rerank_score": float(score), "score": float(score)} for chunk, score in zip(chunks, scores)]
    return sorted(scored, key=lambda chunk: chunk["score"], reverse=True)[:k]
