"""Download CTIBench's ATE subset, build a document corpus, and ingest it into Qdrant.

CTIBench (NeurIPS'24, https://arxiv.org/abs/2406.07599) is a suite of English,
peer-reviewed cyber threat intelligence benchmarks. Only one of its five
subtasks is pulled here:

- CTI-ATE: real MITRE ATT&CK software/malware descriptions, labeled with the
  main ATT&CK technique IDs the software uses. Harder than AttackQA's
  single-relation lookups: multi-label technique extraction from free text,
  with no per-relation template to pattern-match.

The official CTIBench Hugging Face dataset (AI4Sec/cti-bench) omits ground
truth for some subtasks, so this is pulled from the paper's own GitHub repo
instead, which carries the labels: cti-ate.tsv has a GT column directly.

This subtask poses its questions closed-book, with the full source text
pasted into the question — before evaluating (benchmark/scripts/eval.py),
run benchmark/scripts/rewrite_ctibench_questions.py once to replace that
with a short query that doesn't just hand the retriever its own answer.

Run with: uv run python -m benchmark.scripts.ctibench
"""

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag.ingest import main as ingest


REPO_RAW = "https://raw.githubusercontent.com/maveryn/cti-bench/main"

RAW_DIR = Path("data/raw/ctibench")
ATE_OUTPUT_PATH = RAW_DIR / "cti_ate.parquet"
DOCUMENTS_DIR = Path("data/documents/ctibench")

_MITRE_ID = re.compile(r"/software/(S\d+)/?")


def _download_ate() -> pd.DataFrame:
    df = pd.read_csv(f"{REPO_RAW}/data/cti-ate.tsv", sep="\t")
    df["mitre_id"] = df["URL"].str.extract(_MITRE_ID)
    df["gt_techniques"] = df["GT"].apply(lambda value: [t.strip() for t in value.split(",") if t.strip()])
    return df.rename(columns={"URL": "url", "Platform": "platform", "Description": "description"})[
        ["url", "mitre_id", "platform", "description", "gt_techniques"]
    ]


def download(force: bool = False) -> None:
    """Download the subset and cache it as a local parquet file, unless already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ATE_OUTPUT_PATH.exists() and not force:
        print(f"Using cached CTI-ATE at {ATE_OUTPUT_PATH}")
    else:
        ate = _download_ate()
        ate.to_parquet(ATE_OUTPUT_PATH, index=False)
        print(f"Downloaded {len(ate)} CTI-ATE rows")


def iter_ate_documents(df: pd.DataFrame) -> Iterator[tuple[str, Path, dict]]:
    """Yield (document text, path relative to DOCUMENTS_DIR, sidecar metadata) per CTI-ATE row."""
    for row in df.itertuples(index=False):
        document = row.description.strip()
        if not document:
            continue
        metadata = {
            "url": row.url,
            "mitre_id": row.mitre_id,
            "platform": row.platform,
            "gt_techniques": list(row.gt_techniques),
        }
        yield document, Path("cti_ate") / f"{row.mitre_id}.txt", metadata


def build_documents() -> None:
    """Write one document file plus metadata sidecar per CTI-ATE row."""
    written = 0
    df = pd.read_parquet(ATE_OUTPUT_PATH)
    for document, relative_path, metadata in iter_ate_documents(df):
        text_path = DOCUMENTS_DIR / relative_path
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(document + "\n", encoding="utf-8")

        sidecar_path = text_path.with_name(text_path.name + ".meta.json")
        sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written += 1

    print(f"Wrote {written} documents to {DOCUMENTS_DIR}")


def load(*, collection: str = "cyberqa_documents", force_download: bool = False) -> int:
    """Download, build, and ingest CTIBench's ATE subset; return rag.ingest's exit status."""
    download(force=force_download)
    build_documents()
    return ingest([str(DOCUMENTS_DIR), "--collection", collection])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download CTIBench ATE, build documents, and ingest them into Qdrant.")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name")
    parser.add_argument("--force-download", action="store_true", help="Re-download the benchmark even if cached")
    args = parser.parse_args(argv)
    return load(collection=args.collection, force_download=args.force_download)


if __name__ == "__main__":
    raise SystemExit(main())
