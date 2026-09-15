"""Run with: uv run python -m rag.ingest DOCUMENTS_DIRECTORY."""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

from rag.chunk import chunk_text
from rag.device import get_device
from rag.embedding import BATCH_SIZE, MODEL_NAME, embed_chunks
from rag.load import load_document
from rag.store import ensure_collection, point_id, store_chunks


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Ingest a directory and return a nonzero status for incomplete ingestion."""
    parser = argparse.ArgumentParser(description="Load, chunk, embed, and store documents in Qdrant.")
    parser.add_argument("documents", type=Path, help="Directory to scan recursively")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    directory = args.documents.expanduser().resolve()
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and not path.name.endswith(".meta.json")
    )
    if not files:
        logger.error("No files found in %s", directory)
        return 1

    model = SentenceTransformer(MODEL_NAME, device=get_device())
    completed = total_chunks = failed = 0
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )
    try:
        ensure_collection(client, args.collection, model.get_sentence_embedding_dimension())
        for path in files:
            try:
                text = load_document(path)
                chunks = chunk_text(text, model.tokenizer)
                if not chunks:
                    raise ValueError("No extractable text; scanned documents require OCR first.")
                sidecar = path.with_name(path.name + ".meta.json")
                extra_metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
            except Exception as exc:
                logger.error("Could not read %s: %s", path, exc)
                failed += 1
                continue

            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]
                embeddings = embed_chunks(model, batch)
                points = [
                    models.PointStruct(
                        id=point_id(path.as_uri(), start + index),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": path.relative_to(directory).as_posix(),
                            "source_uri": path.as_uri(),
                            "chunk_index": start + index,
                            "embedding_model": MODEL_NAME,
                            **extra_metadata,
                        },
                    )
                    for index, (chunk, vector) in enumerate(zip(batch, embeddings, strict=True))
                ]
                store_chunks(client, args.collection, points)
            completed += 1
            total_chunks += len(chunks)
            logger.info("Stored %s (%d chunks)", path.relative_to(directory), len(chunks))
    finally:
        client.close()

    logger.info("Finished: %d files, %d chunks stored, %d files failed", completed, total_chunks, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
