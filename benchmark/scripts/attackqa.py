"""Download AttackQA, build a document corpus from it, and ingest it into Qdrant.

Deduplicates benchmark rows by their source `document` text and writes each
unique document as a standalone text file, with a JSON metadata sidecar
(MITRE ATT&CK subject, URL, relation) that `rag.ingest` merges into the
stored chunk payloads.

Run with: uv run python -m benchmark.scripts.attackqa
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

from rag.ingest import main as ingest


RAW_PATH = Path("data/raw/attackqa/attackqa.parquet")
DOCUMENTS_DIR = Path("data/documents/attackqa")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    """Sanitize a value for use in a filename."""
    return _UNSAFE.sub("-", value).strip("-") or "unknown"


def download(force: bool = False) -> None:
    """Download the benchmark and cache it as a local parquet file, unless already cached."""
    if RAW_PATH.exists() and not force:
        print(f"Using cached benchmark at {RAW_PATH}")
        return

    dataset = load_dataset("sambanovasystems/attackqa", split="train")
    df = dataset.to_pandas()
    print(f"Downloaded {df.shape[0]} rows")

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)


def iter_documents(df: pd.DataFrame) -> Iterator[tuple[str, Path, dict]]:
    """Yield (document text, path relative to DOCUMENTS_DIR, sidecar metadata) per unique document.

    Deduplicates `df` by its `document` column and assigns each one the same
    grouped, numbered filename that build_documents() writes to disk, so
    callers that need the mapping without touching disk (e.g. eval scripts
    matching a benchmark question back to its ingested source) stay in sync
    with it.
    """
    group_counts: Counter[str] = Counter()
    for row in df.drop_duplicates(subset=["document"]).itertuples(index=False):
        document = row.document.strip()
        if not document:
            continue

        group = f"{row.source}__{_slug(row.subject_id)}"
        group_counts[group] += 1
        stem = f"{group}__{group_counts[group]:03d}"

        metadata = {
            "url": row.url,
            "subject_id": row.subject_id,
            "subject_name": row.subject_name,
            "subject_type": row.subject_type,
            "source_category": row.source,
            "relation_name": row.relation_name if isinstance(row.relation_name, str) else None,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        yield document, Path(row.source) / f"{stem}.txt", metadata


def build_documents() -> None:
    """Write one document file plus metadata sidecar per unique benchmark document."""
    df = pd.read_parquet(RAW_PATH)

    written = 0
    for document, relative_path, metadata in iter_documents(df):
        text_path = DOCUMENTS_DIR / relative_path
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(document + "\n", encoding="utf-8")

        sidecar_path = text_path.with_name(text_path.name + ".meta.json")
        sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written += 1

    print(f"Wrote {written} documents to {DOCUMENTS_DIR}")


def load(*, collection: str = "cyberqa_documents", force_download: bool = False) -> int:
    """Download, build, and ingest AttackQA; return rag.ingest's exit status."""
    download(force=force_download)
    build_documents()
    return ingest([str(DOCUMENTS_DIR), "--collection", collection])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download AttackQA, build documents, and ingest them into Qdrant.")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name")
    parser.add_argument("--force-download", action="store_true", help="Re-download the benchmark even if cached")
    args = parser.parse_args(argv)
    return load(collection=args.collection, force_download=args.force_download)


if __name__ == "__main__":
    raise SystemExit(main())
