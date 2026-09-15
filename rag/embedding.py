"""Generate local sentence embeddings for document chunks."""

import os

from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE") or 32)


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> list[list[float]]:
    """Encode chunks as normalized vectors.

    Args:
        model: Loaded embedding model, reused across all documents.
        chunks: Document chunks to encode.

    Returns:
        One vector per chunk, in the same order.
    """
    return model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
