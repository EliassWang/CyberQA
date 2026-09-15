"""Rewrite CTI-ATE questions so they no longer embed their own answer document.

CTIBench poses this subtask closed-book: the question already contains the
full source text verbatim (see ctibench.py's module docstring), so a
retriever trivially "finds" the gold document by near-exact match — it isn't
being tested. This calls the configured LLM (llm.provider, same
LLM_API_KEY/LLM_MODEL as the agent) once per row to compress each question
down to the kind of short, natural query an analyst would actually type,
holding back the technical details that only live in the document. That
turns retrieval back into a real search instead of a self-match, e.g.
"Which MITRE ATT&CK techniques does the Windows malware 3PARA RAT use?" —
names the software, withholds its behavior.

This is an approximation, not a guarantee: the LLM decides what counts as
"a detail only the document has," and a too-loose rewrite could still leak
enough to make retrieval easy, or a too-tight one could make the question
unanswerable even with the right document retrieved. Spot-check a sample
before trusting the resulting hit-rate numbers.

Writes the result back into the same cached parquet file (ctibench.py's
ATE_OUTPUT_PATH) as a new `question` column, skipping rows that already
have one unless --force is given, saving after every row so an interrupted
run loses no progress.

Run with:
  uv run python -m benchmark.scripts.rewrite_ctibench_questions
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from benchmark.scripts.ctibench import ATE_OUTPUT_PATH
from llm.provider import generate


_ATE_SYSTEM_PROMPT = """You write short benchmark questions for testing a threat-intel retrieval system.
Given a MITRE ATT&CK software's platform and description, output ONLY a single short
natural-language question asking which ATT&CK techniques that software uses. Name the
software by the actual name found in the description. Do NOT restate any technical or
behavioral details from the description (no encryption/protocol/command specifics) — the
question must be answerable only by separately looking up the description, not from the
question text alone. Output only the question, nothing else."""


def rewrite_ate_question(description: str, platform: str) -> str:
    return generate(f"Platform: {platform}\nDescription: {description}", system=_ATE_SYSTEM_PROMPT).strip()


def _rewrite(path: str, rewrite_row, force: bool, label: str) -> None:
    df = pd.read_parquet(path)
    if "question" not in df.columns:
        df["question"] = None

    pending = df.index[df["question"].isna()] if not force else df.index
    for position, index in enumerate(pending, start=1):
        df.loc[index, "question"] = rewrite_row(df.loc[index])
        df.to_parquet(path, index=False)
        print(f"[{label}] {position}/{len(pending)} rewritten")

    if not len(pending):
        print(f"[{label}] All {len(df)} questions already rewritten (use --force to redo).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rewrite CTI-ATE questions to remove embedded source text.")
    parser.add_argument("--force", action="store_true", help="Re-rewrite rows that already have a `question`.")
    args = parser.parse_args(argv)

    _rewrite(
        ATE_OUTPUT_PATH,
        lambda row: rewrite_ate_question(row["description"], row["platform"]),
        args.force, "ate",
    )


if __name__ == "__main__":
    main()
