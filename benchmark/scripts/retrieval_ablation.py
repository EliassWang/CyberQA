"""Retrieval-only ablation: hybrid alpha x rerank x k, graded by gold-source hit rate.

Sweeps HYBRID_ALPHA in {0.0, 0.25, 0.5, 0.75, 1.0}, rerank in {off, on}, and
k in {5, 10} over a fixed AttackQA question sample, scoring each config by
whether a plain top-k hybrid search recovers the question's known gold
source. No LLM calls: this only exercises the local dense/BM25/cross-encoder
retrieval stack (agent.retrieval.hybrid.search), independent of the agent's
own adaptive retrieval loop. Requires Qdrant running with a populated
collection (see rag/load_documents.py).

Run with:
  uv run python -m benchmark.scripts.retrieval_ablation --n 50 --seed 42
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from agent.retrieval.hybrid import search
from benchmark.scripts.eval import _attackqa_rows

RESULT_DIR = Path("benchmark/result")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RERANK_OPTIONS = [False, True]
K_VALUES = [5, 10]
MAX_RETRIES = 5


def _search_with_retry(question: str, collection: str, k: int, alpha: float, rerank: bool) -> list[dict]:
    """Retry on transient Qdrant errors (e.g. a 502 under sustained query load).

    A multi-thousand-call sweep against a local Qdrant instance occasionally
    hits a transient Bad Gateway; without this, one hiccup partway through
    kills the whole run and loses every config computed so far.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return search(question, collection, k=k, alpha=alpha, rerank=rerank)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _topk_hit(question: str, gold_sources: list[str], collection: str, k: int, alpha: float, rerank: bool) -> dict | None:
    if not gold_sources:
        return None
    hits = _search_with_retry(question, collection, k=k, alpha=alpha, rerank=rerank)
    sources = list(dict.fromkeys(hit["source"] for hit in hits))
    ranks = [sources.index(gs) + 1 for gs in gold_sources if gs in sources]
    rank = min(ranks) if ranks else None
    return {"hit": rank is not None, "reciprocal_rank": (1 / rank) if rank else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collection", default="cyberqa_documents")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Existing result JSON to resume; already-computed configs are skipped.")
    args = parser.parse_args()

    console = Console()
    rows = [r for r in _attackqa_rows(0, args.n, args.n, args.seed) if r["gold_sources"]]
    console.print(f"[dim]Loaded {len(rows)} AttackQA questions with a known gold source (n={args.n}, seed={args.seed})[/dim]")

    output = args.resume or args.output or RESULT_DIR / f"retrieval_ablation_n{args.n}-seed{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = []
    done = set()
    if args.resume and output.exists():
        configs = json.loads(output.read_text())["configs"]
        done = {(c["k"], c["alpha"], c["rerank"]) for c in configs}
        console.print(f"[dim]Resuming {output}: {len(done)} configs already done[/dim]")

    for k in K_VALUES:
        for rerank in RERANK_OPTIONS:
            for alpha in ALPHAS:
                if (k, alpha, rerank) in done:
                    continue
                graded = [_topk_hit(r["question"], r["gold_sources"], args.collection, k, alpha, rerank) for r in rows]
                hit_rate = round(sum(1 for g in graded if g["hit"]) / len(graded), 3)
                mrr = round(sum(g["reciprocal_rank"] for g in graded) / len(graded), 3)
                configs.append({"k": k, "alpha": alpha, "rerank": rerank, "n": len(graded), "hit_rate": hit_rate, "mrr": mrr})
                console.print(f"k={k} alpha={alpha} rerank={rerank}: hit_rate={hit_rate} mrr={mrr}")
                output.write_text(json.dumps({"params": vars(args) | {"output": str(output)}, "configs": configs}, indent=2, default=str))

    table = Table(title="Retrieval ablation")
    for col in ("k", "alpha", "rerank", "n", "hit_rate", "mrr"):
        table.add_column(col)
    for c in configs:
        table.add_row(str(c["k"]), str(c["alpha"]), str(c["rerank"]), str(c["n"]), str(c["hit_rate"]), str(c["mrr"]))
    console.print(table)
    console.print(f"[dim]Wrote {output}[/dim]")


if __name__ == "__main__":
    main()
