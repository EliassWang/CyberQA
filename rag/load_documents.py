"""Download AttackQA, CTIBench (ATE), and MITRE ATT&CK Mobile-matrix data,
build their document corpora, and ingest all three into the same Qdrant
collection.

Each source keeps its own document root (data/documents/attackqa,
data/documents/ctibench, data/documents/mitre_mobile) so their `source`
payloads stay independently addressable — eval.py resolves gold sources for
each task relative to their own root — but all three get ingested into one
collection so the agent retrieves across the combined corpus.

AttackQA is built only from ATT&CK's Enterprise matrix, so mitre_mobile
supplies the Mobile-matrix technique and software-technique-relation
documents AttackQA has zero coverage of (see benchmark/scripts/mitre_mobile.py
and report/main.tex sec:main-result) — without it, CTI-ATE's Mobile-platform
questions have no technique-mapping document anywhere in the corpus to
retrieve.

CTIBench's questions need a separate, one-time LLM rewrite pass before
they're usable for evaluation (see benchmark/scripts/rewrite_ctibench_questions.py) —
that isn't run here, since it costs LLM calls and doesn't need to be redone
on every reload.

Requires a running Qdrant instance (see README.md for setup).

Run with: uv run python -m rag.load_documents
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from benchmark.scripts import attackqa, ctibench, mitre_mobile


def main() -> int:
    """Run the full download-to-ingest pipeline for all three sources and return a combined exit status."""
    parser = argparse.ArgumentParser(description="Download AttackQA + CTIBench + MITRE ATT&CK Mobile data, build documents, and ingest all into Qdrant.")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name")
    parser.add_argument("--force-download", action="store_true", help="Re-download the benchmarks even if cached")
    args = parser.parse_args()

    attackqa_status = attackqa.load(collection=args.collection, force_download=args.force_download)
    ctibench_status = ctibench.load(collection=args.collection, force_download=args.force_download)
    mitre_mobile_status = mitre_mobile.load(collection=args.collection, force_download=args.force_download)
    return attackqa_status or ctibench_status or mitre_mobile_status


if __name__ == "__main__":
    raise SystemExit(main())
