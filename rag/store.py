"""Create the Qdrant collection and store document chunks."""

import time
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException


MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2


def ensure_collection(client: QdrantClient, name: str, dimension: int) -> None:
    """Create a cosine collection or validate an existing collection.

    Args:
        client: Connected Qdrant client.
        name: Collection name.
        dimension: Embedding vector size.

    Raises:
        ValueError: An existing collection has incompatible vector settings.
    """
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        return

    vectors = client.get_collection(name).config.params.vectors
    if (
        not isinstance(vectors, models.VectorParams)
        or vectors.size != dimension
        or vectors.distance != models.Distance.COSINE
    ):
        raise ValueError(f"Collection {name!r} has incompatible vector settings.")


def store_chunks(client: QdrantClient, collection: str, points: list[models.PointStruct]) -> None:
    """Store a batch and wait for Qdrant to apply it.

    Retries on transient API errors (e.g. a proxy-level 502 under sustained
    request volume); point IDs are deterministic, so a retried upsert is safe.

    Args:
        client: Connected Qdrant client.
        collection: Destination collection.
        points: Chunk vectors and their source metadata.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
            return
        except ApiException:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)


def point_id(source: str, chunk_index: int) -> str:
    """Return a stable UUID for a source URI and chunk index."""
    return str(uuid5(NAMESPACE_URL, f"{source}#chunk={chunk_index}"))
