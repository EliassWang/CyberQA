"""BM25 lexical index over an ingested Qdrant collection.

Chunk text only lives in Qdrant (never persisted separately), so building
this index means scrolling the whole collection once and keeping the
result in memory. Cheap enough at this corpus size (tens of thousands of
short chunks) to rebuild per process; call refresh() after re-ingesting a
collection to drop the stale cache.
"""

import os
import re
from functools import lru_cache

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ID = re.compile(r"\b[A-Za-z]\d{4}(?:\.\d{3})?\b")

_SCROLL_PAGE_SIZE = 1000


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization shared by corpus and queries."""
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )


@lru_cache(maxsize=4)
def _index(collection: str) -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index over every chunk payload in `collection`, once per process."""
    client = _client()
    payloads: list[dict] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=_SCROLL_PAGE_SIZE,
            offset=next_page,
        )
        payloads.extend(point.payload for point in points)
        if next_page is None:
            break

    corpus = [_tokenize(payload["text"]) for payload in payloads]
    return BM25Okapi(corpus), payloads


def refresh(collection: str | None = None) -> None:
    """Drop the cached index so the next search rebuilds it from Qdrant.

    Call after re-ingesting `collection` (or any collection, if unspecified).
    """
    _index.cache_clear()


def _id_tokens(query: str) -> set[str]:
    """Tokens for every ATT&CK-style ID (e.g. "S0011", "T1547.004") named in `query`.

    Each ID is re-tokenized with `_tokenize` so a dotted sub-technique ID,
    which splits into two tokens there too (e.g. "t1547"/"004"), matches the
    same way it was indexed. Empty unless `query` names 2+ distinct IDs: a
    single ID is common enough (e.g. every chunk about that one technique)
    that it wouldn't discriminate, so callers leave that to dense/BM25/rerank.
    """
    ids = set(_ID.findall(query))
    if len(ids) < 2:
        return set()
    tokens: set[str] = set()
    for match in ids:
        tokens.update(_tokenize(match))
    return tokens


def exact_id_hits(query: str, collection: str) -> list[dict]:
    """Return every chunk whose text contains all ATT&CK-style IDs named in `query`.

    Corpus documents like "how does software X use technique Y" exist once
    per (X, Y) pair, and for a technique or software with many such pairs
    (e.g. T1105 has 348) they're near-identical boilerplate that differs
    only in the two IDs named — dense/BM25 top-N ranking is dominated by the
    shared wording and can push the one correct sibling out of the candidate
    pool entirely, regardless of k. An ID is rare enough on its own that
    requiring every ID the query names to literally appear in the chunk
    finds it deterministically, independent of corpus size.
    """
    tokens = _id_tokens(query)
    if not tokens:
        return []
    _, payloads = _index(collection)
    return [
        {**payload, "score": 1.0}
        for payload in payloads
        if tokens <= set(_tokenize(payload["text"]))
    ]


def search(query: str, collection: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks by BM25 score.

    Args:
        query: Search text.
        collection: Qdrant collection to search.
        k: Number of chunks to return.

    Returns:
        Chunk payloads (text, source, chunk_index, and any sidecar metadata)
        with an added "score" field, ordered by descending BM25 score.
    """
    bm25, payloads = _index(collection)
    if not payloads:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(payloads)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**payloads[i], "score": float(scores[i])} for i in ranked]
