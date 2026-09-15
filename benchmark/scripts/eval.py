"""Evaluate the CyberQA agent against AttackQA and/or CTIBench (ATE) questions.

For each selected question, runs the full agent graph (agent.graph) and
records a trace: the produced answer, the verdicts behind it, and
whether a known source document was actually retrieved — both by the
agent's own adaptive retrieval and by a plain top-k hybrid (dense + BM25,
cross-encoder reranked) search, so retrieval quality can be judged
independently of the LLM's tool-calling choices. Requires Qdrant running
with a populated collection (see rag/load_documents.py), and LLM_API_KEY/LLM_MODEL
set (see README).

Two question sources, selected with --task:

- attackqa (default): AttackQA's single-relation-template questions. Scored
  for free with local-only metrics (benchmark.scripts.score: token
  F1/exact-match + embedding cosine similarity).
- ate: CTIBench's CTI-ATE — multi-label ATT&CK technique extraction from a
  real malware description. Scored as set precision/recall/F1 over
  technique IDs (benchmark.scripts.score_ctibench.score_ate).
- all: attackqa + ate together.

CTIBench poses this subtask closed-book, with the full source text pasted
into the question — that would make retrieval hit-rate meaningless (a
near-exact self-match), so ate rows use the `question` column that
benchmark.scripts.rewrite_ctibench_questions produces instead: a short,
natural query that withholds the source document's details, so the
Retriever actually has to find it. Run that script once before evaluating
ate/all — this errors out with the exact command if it hasn't been.
--n/--seed random sampling requires a single --task, since sampling a
fixed count across heterogeneous question sets isn't well-defined.

Every run writes its parameters, summary, and full per-question trace as
JSON under benchmark/result/, auto-named by task, question range, and
timestamp (e.g. eval_attackqa_q0-20_<timestamp>.json, or
eval_attackqa_n30-seed7_<timestamp>.json for a --n sample) unless --output
is given.

Run with:
  uv run python -m benchmark.scripts.eval --start 0 --end 20
  uv run python -m benchmark.scripts.eval --n 30 --seed 7
  uv run python -m benchmark.scripts.eval --task ate --end 20
  uv run python -m benchmark.scripts.eval --task all --end 20
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.table import Table

load_dotenv()

from agent.graph import build_graph, extract_result
from benchmark.scripts.attackqa import RAW_PATH as ATTACKQA_RAW_PATH, iter_documents as iter_attackqa_documents
from benchmark.scripts.ctibench import ATE_OUTPUT_PATH, iter_ate_documents
from benchmark.scripts.mitre_mobile import iter_relationship_documents as iter_mobile_relationship_documents
from benchmark.scripts.mitre_mobile import RAW_PATH as MOBILE_RAW_PATH, _load_bundle as _load_mobile_bundle
from benchmark.scripts.score import score_answer
from benchmark.scripts.score_ctibench import score_ate
from agent.retrieval.hybrid import DEFAULT_ALPHA, search


TOP_K_DEFAULT = 10
RESULT_DIR = Path("benchmark/result")
TASK_CHOICES = ["attackqa", "ate", "all"]


def _attackqa_gold_source_map() -> dict[str, str]:
    """Map each unique AttackQA document's text to its ingested relative source path."""
    df = pd.read_parquet(ATTACKQA_RAW_PATH)
    return {document: relative_path.as_posix() for document, relative_path, _ in iter_attackqa_documents(df)}


def _attackqa_technique_sources_by_software() -> dict[str, list[str]]:
    """Map each MITRE software ID to its AttackQA `relationships_techniques_for_software` doc paths.

    Both benchmarks are ultimately grounded in the same MITRE ATT&CK
    catalog: 46 of CTI-ATE's 60 software entries already have a
    software-uses-technique relation document in the AttackQA corpus (e.g.
    "3PARA RAT" is both CTI-ATE's S0066 row and AttackQA's
    relationships_techniques_for_software__S0066__*.txt). Since both
    corpora are ingested into the same collection, retrieval finding the
    AttackQA doc instead of CTI-ATE's own is still a correct hit, not a
    miss — this feeds _ctibench_rows so gold_sources accepts either.
    """
    mapping: dict[str, list[str]] = {}
    for _, relative_path, metadata in iter_attackqa_documents(pd.read_parquet(ATTACKQA_RAW_PATH)):
        if metadata.get("source_category") == "relationships_techniques_for_software":
            mapping.setdefault(metadata["subject_id"], []).append(relative_path.as_posix())
    return mapping


def _mitre_mobile_technique_sources_by_software() -> dict[str, list[str]]:
    """Map each MITRE software ID to its mitre_mobile `relationships_techniques_for_software` doc path(s).

    Mirrors _attackqa_technique_sources_by_software: mitre_mobile
    (benchmark/scripts/mitre_mobile.py) supplies the Mobile-matrix
    technique-relation documents AttackQA has zero coverage of, ingested
    into the same collection under its own document root, so a hit there is
    also a correct retrieval, not a miss. Empty if the bundle hasn't been
    downloaded (rag/load_documents.py / mitre_mobile.py not yet run).
    """
    if not MOBILE_RAW_PATH.exists():
        return {}
    software, techniques, uses = _load_mobile_bundle()
    mapping: dict[str, list[str]] = {}
    for _, relative_path, metadata in iter_mobile_relationship_documents(software, techniques, uses):
        mapping.setdefault(metadata["subject_id"], []).append(relative_path.as_posix())
    return mapping


def _slice_or_sample(df: pd.DataFrame, start: int, end: int, n: int | None, seed: int) -> pd.DataFrame:
    if n is not None:
        return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df.iloc[start:end].reset_index(drop=True)


def _attackqa_rows(start: int, end: int, n: int | None, seed: int) -> list[dict]:
    """Build unified {task, question, gold_answer, gold_sources} rows for AttackQA."""
    df = pd.read_parquet(ATTACKQA_RAW_PATH)
    df = df[df["document"].str.strip() != ""]
    df = df.drop_duplicates(subset=["document"]).reset_index(drop=True)
    df = _slice_or_sample(df, start, end, n, seed)

    gold_map = _attackqa_gold_source_map()
    rows = []
    for row in df.itertuples(index=False):
        gold_source = gold_map.get(row.document.strip())
        rows.append({
            "task": "attackqa", "question": row.question,
            "gold_answer": row.answer, "gold_sources": [gold_source] if gold_source else [],
        })
    return rows


def _ate_rows(start: int, end: int, n: int | None, seed: int) -> list[dict]:
    """Build unified {task, question, gold_answer, gold_sources} rows for CTI-ATE.

    Uses the rewritten `question` column from
    benchmark.scripts.rewrite_ctibench_questions rather than the raw
    description text — the raw text embeds the answer document verbatim,
    which would make retrieval hit-rate meaningless (see that script's
    docstring). Run it first; this errors out if it hasn't been.
    """
    df = pd.read_parquet(ATE_OUTPUT_PATH)
    if "question" not in df.columns or df["question"].isna().any():
        raise SystemExit(
            f"{ATE_OUTPUT_PATH} has no rewritten questions yet. Run: "
            f"uv run python -m benchmark.scripts.rewrite_ctibench_questions"
        )
    df = _slice_or_sample(df, start, end, n, seed)

    attackqa_overlap = _attackqa_technique_sources_by_software()
    mobile_overlap = _mitre_mobile_technique_sources_by_software()
    rows = []
    for row, (_, gold_source, _) in zip(df.itertuples(index=False), iter_ate_documents(df), strict=True):
        gold_sources = [gold_source.as_posix()] + attackqa_overlap.get(row.mitre_id, []) + mobile_overlap.get(row.mitre_id, [])
        rows.append({
            "task": "ate", "question": row.question,
            "gold_answer": list(row.gt_techniques), "gold_sources": gold_sources,
        })
    return rows


def _select_rows(task: str, start: int, end: int, n: int | None, seed: int) -> list[dict]:
    if n is not None and task == "all":
        raise SystemExit("--n sampling requires a single --task (attackqa or ate), not a combined one.")

    rows = []
    if task in ("attackqa", "all"):
        rows += _attackqa_rows(start, end, n, seed)
    if task in ("ate", "all"):
        rows += _ate_rows(start, end, n, seed)
    return rows


def _topk_retrieval(question: str, gold_sources: list[str], collection: str, k: int, alpha: float, rerank: bool) -> dict | None:
    """Plain top-k hybrid (dense + BM25 [+ rerank]) search, graded against any of the gold sources.

    gold_sources has more than one entry only for CTI-ATE rows whose
    software also has an AttackQA relation doc (see
    _attackqa_technique_sources_by_software); rank is the best (lowest)
    rank among whichever of them was actually retrieved. None if no gold
    source is known for this row.
    """
    if not gold_sources:
        return None
    hits = search(question, collection, k=k, alpha=alpha, rerank=rerank)
    sources = list(dict.fromkeys(hit["source"] for hit in hits))  # de-dup, keep first-seen rank
    ranks = [sources.index(gs) + 1 for gs in gold_sources if gs in sources]
    rank = min(ranks) if ranks else None
    return {
        "hit": rank is not None,
        "rank": rank,
        "reciprocal_rank": (1 / rank) if rank else 0.0,
        "retrieved_sources": sources,
    }


def _score_answer(task: str, answer: str | None, gold_answer) -> dict:
    if task == "attackqa":
        return score_answer(answer, gold_answer)
    return score_ate(answer, gold_answer)


def run_eval(
    rows: list[dict],
    *,
    collection: str,
    k: int,
    alpha: float,
    rerank: bool,
    console: Console,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Run the agent over each row and return one trace record per question.

    If checkpoint_path is given, the records collected so far (with a summary
    over just those) are written to it after every question -- a long sweep
    against a real LLM can run for tens of minutes, and losing every record
    to an interruption partway through (a killed process, a dropped session)
    would waste all the LLM calls made before that point, not just the ones
    after it.
    """
    app = build_graph()
    records = []

    for position, row in enumerate(rows, start=1):
        task, question, gold_answer, gold_sources = row["task"], row["question"], row["gold_answer"], row["gold_sources"]
        topk = _topk_retrieval(question, gold_sources, collection, k, alpha, rerank)

        start_time = time.monotonic()
        error = None
        try:
            raw_state = app.invoke({"query": question}, config={"recursion_limit": 50})
        except Exception as exc:  # keep the sweep going; record the failure instead
            raw_state = {}
            error = str(exc)
        elapsed = time.monotonic() - start_time
        result = extract_result(raw_state)

        pipeline_sources = [chunk.get("source") for chunk in result.get("sources") or []]
        pipeline_hit = any(gs in pipeline_sources for gs in gold_sources) if gold_sources else None

        record = {
            "task": task,
            "question": question,
            "gold_answer": gold_answer,
            "gold_sources": gold_sources,
            "answer": result.get("answer"),
            "message": result.get("message"),
            "answer_score": _score_answer(task, result.get("answer"), gold_answer),
            "scope": raw_state.get("scope"),
            "evidence_verdict": raw_state.get("evidence_verdict"),
            "grounding_verdict": raw_state.get("grounding_verdict"),
            "total_steps": raw_state.get("total_steps"),
            "retrieval_attempts": raw_state.get("retrieval_attempts"),
            "generation_attempts": raw_state.get("generation_attempts"),
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "pipeline_retrieval_hit": pipeline_hit,
            "topk_retrieval": topk,
            "trace": raw_state.get("trace", []),
        }
        records.append(record)
        _print_trace(console, position, len(rows), record)
        if checkpoint_path is not None:
            save_results(checkpoint_path, {"checkpoint": True, "k": k, "alpha": alpha, "rerank": rerank}, summarize(records, k), records)

    return records


def _score_str(task: str, score: dict) -> str:
    if task == "attackqa":
        return f"f1={score['f1']:.2f} sim={score['embedding_similarity']:.2f}"
    if task == "ate":
        return f"f1={score['f1']:.2f}"
    return f"verdict={score['verdict']}"


def _print_trace(console: Console, position: int, total: int, record: dict) -> None:
    status = "error" if record["error"] else ("answered" if record["answer"] else "refused")
    style = {"answered": "green", "refused": "yellow", "error": "red"}[status]
    topk = record["topk_retrieval"]
    topk_str = f"hit@k(rank {topk['rank']})" if topk and topk["hit"] else ("miss@k" if topk else "n/a")
    pipeline_str = "hit" if record["pipeline_retrieval_hit"] else ("miss" if record["pipeline_retrieval_hit"] is not None else "n/a")
    score_str = _score_str(record["task"], record["answer_score"])

    console.print(
        f"[dim]{position}/{total}[/dim] [{style}]{status}[/{style}] [{record['task']}] "
        f"topk={topk_str} pipeline={pipeline_str} "
        f"{score_str} "
        f"({record['elapsed_seconds']}s) — {escape(record['question'][:90])}"
    )
    if record["error"]:
        console.print(f"  [red]{escape(record['error'])}[/red]")


def summarize(records: list[dict], k: int) -> dict:
    """Aggregate per-question records into headline metrics, split by task."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    graded = [r for r in records if r["topk_retrieval"] is not None]
    answered = sum(1 for r in records if r["answer"])
    errors = sum(1 for r in records if r["error"])

    summary = {
        "n": n,
        "answered_rate": round(answered / n, 3),
        "refused_rate": round((n - answered - errors) / n, 3),
        "error_rate": round(errors / n, 3),
        "avg_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in records) / n, 2),
    }
    if graded:
        summary[f"retrieval_hit_rate@{k}"] = round(sum(1 for r in graded if r["topk_retrieval"]["hit"]) / len(graded), 3)
        summary[f"retrieval_mrr@{k}"] = round(sum(r["topk_retrieval"]["reciprocal_rank"] for r in graded) / len(graded), 3)
        summary["pipeline_retrieval_hit_rate"] = round(
            sum(1 for r in graded if r["pipeline_retrieval_hit"]) / len(graded), 3
        )

    attackqa_records = [r for r in records if r["task"] == "attackqa"]
    if attackqa_records:
        m = len(attackqa_records)
        summary["attackqa_n"] = m
        summary["attackqa_avg_f1"] = round(sum(r["answer_score"]["f1"] for r in attackqa_records) / m, 3)
        summary["attackqa_avg_embedding_similarity"] = round(
            sum(r["answer_score"]["embedding_similarity"] for r in attackqa_records) / m, 3
        )
        summary["attackqa_avg_total_steps"] = round(sum(r["total_steps"] or 0 for r in attackqa_records) / m, 2)

    ate_records = [r for r in records if r["task"] == "ate"]
    if ate_records:
        m = len(ate_records)
        summary["ate_n"] = m
        summary["ate_avg_f1"] = round(sum(r["answer_score"]["f1"] for r in ate_records) / m, 3)
        summary["ate_avg_precision"] = round(sum(r["answer_score"]["precision"] for r in ate_records) / m, 3)
        summary["ate_avg_recall"] = round(sum(r["answer_score"]["recall"] for r in ate_records) / m, 3)

    return summary


def _print_summary(console: Console, summary: dict) -> None:
    table = Table(title="Eval summary", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in summary.items():
        table.add_row(key, json.dumps(value) if isinstance(value, dict) else str(value))
    console.print(table)


def _question_range_label(start: int, end: int, n: int | None, seed: int) -> str:
    """A short, filename-safe label identifying which questions a run covered."""
    return f"n{n}-seed{seed}" if n is not None else f"q{start}-{end}"


def _default_output_path(task: str, start: int, end: int, n: int | None, seed: int) -> Path:
    """Auto-named result path: task, label identifying the question range, then a UTC timestamp."""
    label = _question_range_label(start, end, n, seed)
    timestamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    return RESULT_DIR / f"eval_{task}_{label}_{timestamp}.json"


def save_results(path: Path, params: dict, summary: dict, records: list[dict]) -> None:
    """Write a run's parameters, summary, and full per-question traces to `path` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "summary": summary, "records": records}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the CyberQA agent against AttackQA and/or CTIBench questions.")
    parser.add_argument("--task", choices=TASK_CHOICES, default="attackqa", help="Which question set(s) to evaluate (default attackqa).")
    parser.add_argument("--start", type=int, default=0, help="Start index into each task's question set (default 0).")
    parser.add_argument("--end", type=int, default=20, help="End index, exclusive (default 20).")
    parser.add_argument("--n", type=int, default=None, help="Randomly sample N questions instead of --start/--end (requires a single --task).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --n sampling (default 42).")
    parser.add_argument("--k", type=int, default=TOP_K_DEFAULT, help=f"Top-k for retrieval metrics (default {TOP_K_DEFAULT}).")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Hybrid retrieval weight on the dense signal, 0-1; (1-alpha) weights BM25 "
             f"(default {DEFAULT_ALPHA}, from HYBRID_ALPHA).",
    )
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking; use raw hybrid score.")
    parser.add_argument("--collection", default="cyberqa_documents", help="Qdrant collection name.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Write the summary and full per-question traces to this JSON file "
             f"(default: an auto-named file under {RESULT_DIR}/, e.g. eval_attackqa_q0-20_<timestamp>.json).",
    )
    parser.add_argument("--no-output", action="store_true", help="Don't write a result file, just print the summary.")
    args = parser.parse_args(argv)

    if args.n is None and args.end <= args.start:
        parser.error("--end must be greater than --start (or use --n to sample instead).")

    console = Console()
    rows = _select_rows(args.task, args.start, args.end, args.n, args.seed)
    if not rows:
        parser.error("No questions selected; check --start/--end or --n against the dataset size.")

    rerank = not args.no_rerank
    output_path = None if args.no_output else (args.output or _default_output_path(args.task, args.start, args.end, args.n, args.seed))
    console.print(
        f"[dim]Evaluating {len(rows)} question(s) (task={args.task}) against collection '{args.collection}' "
        f"(alpha={args.alpha}, rerank={rerank})...[/dim]"
    )
    records = run_eval(
        rows, collection=args.collection, k=args.k, alpha=args.alpha, rerank=rerank,
        console=console, checkpoint_path=output_path,
    )
    summary = summarize(records, args.k)

    console.print()
    _print_summary(console, summary)

    if output_path is not None:
        params = {
            "task": args.task, "start": args.start, "end": args.end, "n": args.n, "seed": args.seed,
            "k": args.k, "alpha": args.alpha, "rerank": rerank, "collection": args.collection,
        }
        save_results(output_path, params, summary, records)
        console.print(f"\n[dim]Full trace written to {output_path}[/dim]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
