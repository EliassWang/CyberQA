"""Run a fixed set of representative queries through the live agent and
write question/answer/sources to a JSON file for the "test queries and
system outputs" deliverable.

Drops `source_uri` from each cited chunk before writing: it's an absolute
`file://` path into this machine's local checkout (data/ is gitignored and
rebuilt per-machine by rag/load_documents.py), not something that belongs in a
committed deliverable. `source` (the corpus-relative path) is kept.

Run with: uv run python -m scripts.run_sample_queries
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, extract_result

QUERIES = [
    # --- Easy: single-subject direct lookup ---
    "What is a potential indicator of the 'T1539: Steal Web Session Cookie' attack technique?",
    "Which mitigations are recommended for 'T1110: Brute Force'?",
    "What data sources can be used to detect 'T1055: Process Injection'?",
    # --- Medium: single-subject but requires synthesis or attribution ---
    "What software has the threat group APT29 been observed using?",
    "How can Process Creation be used to detect Remote System Discovery?",
    "What is the CVE number for the Log4Shell vulnerability?",
    # --- Hard: multi-subject comparison / cross-referencing / broad synthesis ---
    "Compare the detection methods for T1110 Brute Force and T1556 Modify Authentication Process.",
    "Which threat groups have used both Mimikatz and Cobalt Strike?",
    "How do I detect lateral movement?",
    # --- Out-of-scope (not a cybersecurity question) ---
    "What's a good recipe for chocolate chip cookies?",
]


def _sanitize_sources(sources: list[dict]) -> list[dict]:
    """Drop fields that leak local machine details (e.g. absolute paths)."""
    return [{k: v for k, v in chunk.items() if k != "source_uri"} for chunk in sources]


def main() -> None:
    app = build_graph()
    results = []
    for q in QUERIES:
        print(f"--- {q}")
        r = extract_result(app.invoke({"query": q}, config={"recursion_limit": 50}))
        r["sources"] = _sanitize_sources(r.get("sources") or [])
        results.append({"query": q, **r})
        print(json.dumps(r, indent=2, default=str)[:500])

    out_path = Path("examples/sample_queries.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
