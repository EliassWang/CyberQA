"""Similarity search against an ingested Qdrant collection."""

import os
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from rag.device import get_device
from rag.embedding import MODEL_NAME


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=get_device())


def warm_up() -> None:
    """Load the embedding model now rather than on the first query."""
    _model()


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )


def _short_error(exc: Exception) -> str:
    """Collapse a qdrant_client exception to one line for a status display."""
    if isinstance(exc, UnexpectedResponse):
        return f"{exc.status_code} {exc.reason_phrase}".strip()
    return str(exc).splitlines()[0][:120]


def qdrant_status(collection: str) -> dict:
    """Report whether Qdrant is reachable and how many points a collection holds.

    Best-effort, for CLI/startup display: distinguishes Qdrant being
    unreachable at all from Qdrant being up but the named collection being
    missing/unreadable, so a caller can show a more useful message than
    just "down" in the latter case. Any error is caught and returned as
    `error` rather than raised, so callers can show a status line without
    a try/except of their own.

    Returns:
        {"url": str, "reachable": bool, "count": int | None, "error": str | None}
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = _client()
        client.get_collections()  # cheap reachability probe
    except Exception as exc:
        return {"url": url, "reachable": False, "count": None, "error": _short_error(exc)}

    try:
        count = client.count(collection, exact=True).count
        return {"url": url, "reachable": True, "count": count, "error": None}
    except Exception as exc:
        return {
            "url": url,
            "reachable": True,
            "count": None,
            "error": f"collection {collection!r} not found ({_short_error(exc)})",
        }


def search(query: str, collection: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks most similar to the query.

    Args:
        query: Search text.
        collection: Qdrant collection to search.
        k: Number of chunks to return.

    Returns:
        Chunk payloads (text, source, source_uri, chunk_index, embedding_model,
        and any sidecar metadata) with an added "score" field, ordered by
        descending similarity.
    """
    vector = _model().encode(query, normalize_embeddings=True).tolist()
    hits = _client().query_points(collection_name=collection, query=vector, limit=k).points
    return [{**hit.payload, "score": hit.score} for hit in hits]
