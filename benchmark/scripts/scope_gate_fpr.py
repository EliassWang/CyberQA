"""Scope Gate false-positive-rate check on a benchmark's questions.

Every question in these benchmarks is, by construction, a legitimate
cybersecurity question the corpus should be able to speak to -- so any
Scope Gate verdict other than "in_scope" (out_of_scope or ambiguous) on one
of them is a false positive. This isolates the Scope Gate (one LLM call, no
retrieval or generation) and reports that rate, matching the methodology
behind Table tab:scope-gate-fpr in the report (there run on AttackQA; this
script also covers CTI-ATE).

Run with:
  uv run python -m benchmark.scripts.scope_gate_fpr --task ate
  uv run python -m benchmark.scripts.scope_gate_fpr --task attackqa --n 1000 --seed 42
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from agent.graph import scope_gate_node
from benchmark.scripts.eval import _select_rows

RESULT_DIR = Path("benchmark/result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["attackqa", "ate"], default="ate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--n", type=int, default=None, help="Randomly sample N questions instead of --start/--end.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    rows = _select_rows(args.task, args.start, args.end, args.n, args.seed)
    console.print(f"[dim]Classifying {len(rows)} {args.task} question(s) with the Scope Gate...[/dim]")

    records = []
    for i, row in enumerate(rows, start=1):
        decision = scope_gate_node({"query": row["question"]})
        records.append({"question": row["question"], "scope": decision["scope"], "reason": decision["scope_reason"]})
        if decision["scope"] != "in_scope":
            console.print(f"[yellow]{i}/{len(rows)} {decision['scope']}[/yellow]: {row['question'][:90]} ({decision['scope_reason']})")

    n = len(records)
    rejected = [r for r in records if r["scope"] != "in_scope"]
    hard = [r for r in records if r["scope"] == "out_of_scope"]
    fpr = round(len(rejected) / n, 4)
    hard_fpr = round(len(hard) / n, 4)

    console.print(f"\n[bold]FPR (any rejection): {fpr} ({len(rejected)}/{n})[/bold]")
    console.print(f"Hard FPR (out_of_scope only): {hard_fpr} ({len(hard)}/{n})")

    output = args.output or RESULT_DIR / f"scope_gate_fpr_{args.task}_n{n}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "params": {"task": args.task, "start": args.start, "end": args.end, "n": args.n, "seed": args.seed},
        "summary": {"n": n, "fpr": fpr, "hard_fpr": hard_fpr, "rejected": len(rejected), "out_of_scope": len(hard)},
        "records": records,
    }, indent=2))
    console.print(f"[dim]Wrote {output}[/dim]")


if __name__ == "__main__":
    main()
