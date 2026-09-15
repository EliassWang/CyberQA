"""The `retrieve_evidence` tool: Retriever + Evidence Validator.

Retriever does not rewrite a single-subject objective — search quality on
this corpus is sensitive to exact wording (near-duplicate chunks about the
same subject compete on small differences), so a search on the model's own
paraphrase can rank worse than the literal objective would have. Retriever
only involves the model in shaping the search at all when the objective
actually covers multiple independent subjects (e.g. a comparison): it
decomposes that into one targeted sub-query per subject, searched alongside
— never instead of — the verbatim objective, since a relationship question
naming two entities (e.g. "how does software X use technique Y") is a
common misclassification, and the specific X-Y document often only ranks
well against the combined query. Evidence across Orchestrator retry
attempts accumulates rather than being replaced, so a retry can only
add coverage, never lose a chunk an earlier attempt already found. Evidence
Validator then grades the combined result. The whole thing is bundled
behind one Orchestrator-facing tool so the Orchestrator never sees raw chunks — only a
validated verdict — and the two budgets below (sub-queries per Retriever
call, and the Orchestrator's calls to this tool) are each enforced in code
by the component that owns that decision, not left to either model.
"""

import json
import os
import re

from agent.retrieval.hybrid import search
from llm.provider import generate


TOP_K_DEFAULT = int(os.getenv("RETRIEVAL_TOP_K") or 10)
MAX_RETRIEVAL_ATTEMPTS = 2  # Orchestrator calls to retrieve_evidence
MAX_SUB_QUERIES = 3  # sub-queries actually searched per Retriever call, if decomposed
COLLECTION = os.getenv("CYBERQA_COLLECTION", "cyberqa_documents")


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object in a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group(0))


def _format_evidence(chunks: list[dict]) -> str:
    """Render chunks as a numbered list, matching the citation markers agents use."""
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1))


# --- Retriever: decompose only if multi-subject, else search verbatim ----

_DECOMPOSE_SYSTEM_PROMPT = """You plan searches for a cybersecurity knowledge base Retriever.
Decide whether the research objective is a single subject or relationship, or asks about/
compares multiple genuinely independent subjects.

- Single subject or relationship: respond with an empty list. This includes an objective
  that names two specific entities but asks about ONE relationship or fact connecting
  them — e.g. "how does software X use technique Y", "which mitigations apply to Y",
  "which group used tool X". These need one search on the objective's own exact wording,
  because the document that actually answers them is usually the specific X-Y relationship
  record itself, which a query for "X" or "Y" alone will often miss — not generic pages
  about X and Y independently. Do not paraphrase or split these.
- Multiple independent subjects (e.g. "compare X and Y", "what do A, B, and C have in
  common", "how does X's overall purpose differ from Y's"): respond with one short,
  targeted search query per subject, so each subject gets searched on its own terms.

When unsure, prefer an empty list: splitting a single X-Y relationship query can cause a
real miss (neither half's search surfaces the specific record connecting them), while
leaving a genuine comparison unsplit still works, just less precisely.

Respond with only a JSON object:
{"sub_queries": ["query for subject 1", "query for subject 2", ...]}
Use an empty list for a single-subject/single-relationship objective.
"""


def _decompose(objective: str) -> list[str]:
    """Return one search query per subject if `objective` covers several, else []."""
    try:
        decision = _parse_json(generate(objective, system=_DECOMPOSE_SYSTEM_PROMPT))
        sub_queries = decision.get("sub_queries") or []
    except ValueError:
        return []
    if not isinstance(sub_queries, list):
        return []
    return [q for q in sub_queries if isinstance(q, str) and q.strip()][:MAX_SUB_QUERIES]


def _run_retrieval(objective: str, prior_attempts: int, prior_chunks: list[dict]) -> dict:
    """Gather evidence for `objective`, decomposing it first only if it's multi-subject.

    A single-subject objective is searched with its own exact text — search
    quality on this corpus is sensitive to exact wording (many chunks about
    the same subject differ only slightly), so an unnecessary paraphrase can
    rank worse than the literal objective would have. When the objective
    *is* decomposed, the verbatim objective is still searched alongside the
    sub-queries rather than replaced by them: a relationship question naming
    two entities (e.g. "how does software X use technique Y") is a common
    misclassification for _decompose, and the specific X-Y relationship
    document often only ranks well against the combined query — a search for
    "X" or "Y" alone can miss it even though both entities are individually
    well covered elsewhere in the corpus. Keeping the verbatim search means
    a bad decomposition call can only add unnecessary sub-queries, never
    cost the verbatim search's own coverage. Accumulates evidence across
    Orchestrator attempts (seeded from `prior_chunks`) rather than replacing
    it, for the same reason: a retry's rewritten query searches fresh, and a
    rewrite that happens to rank worse than the original must not be able to
    lose a chunk an earlier attempt already found — it can only add
    coverage. Chunks from every query, across all attempts, are merged and
    de-duplicated by (source, chunk_index).
    """
    sub_queries = _decompose(objective)
    queries = [objective, *sub_queries] if sub_queries else [objective]

    collected: dict[tuple[str, int], dict] = {
        (chunk["source"], chunk["chunk_index"]): chunk for chunk in prior_chunks
    }
    sub_calls: list[dict] = []
    for step, query in enumerate(queries, start=1):
        chunks = search(query, COLLECTION, k=TOP_K_DEFAULT)
        for chunk in chunks:
            collected[(chunk["source"], chunk["chunk_index"])] = chunk
        sub_calls.append({
            "step": step,
            "tool": "search_documents",
            "query": query,
            "k": TOP_K_DEFAULT,
            "chunks_found": len(chunks),
            "retrieved_sources": [chunk["source"] for chunk in chunks],
            "total_unique_chunks_so_far": len(collected),
        })

    return {
        "retrieved": list(collected.values()),
        "search_query": "; ".join(queries),
        "retrieval_attempts": prior_attempts + 1,
        "sub_calls": sub_calls,
    }


# --- Evidence Validator --------------------------------------------------

_EVIDENCE_VALIDATOR_PROMPT = """You grade whether retrieved document excerpts contain enough
evidence to answer a cybersecurity question. Grade as:
- "sufficient": the excerpts directly and fully answer the question.
- "partial": the excerpts are relevant but only partially answer it, or leave gaps.
- "insufficient": the excerpts do not meaningfully address the question.

Respond with only a JSON object:
{"verdict": "sufficient"|"partial"|"insufficient", "notes": "what's missing, if anything, phrased so it could guide a better search query"}
"""


def _run_evidence_validation(query: str, chunks: list[dict]) -> dict:
    """Grade a set of retrieved chunks against the original question."""
    if not chunks:
        return {"evidence_verdict": "insufficient", "evidence_notes": "No chunks were retrieved."}
    prompt = f"Question: {query}\n\nRetrieved excerpts:\n{_format_evidence(chunks)}"
    response = generate(prompt, system=_EVIDENCE_VALIDATOR_PROMPT)
    try:
        decision = _parse_json(response)
        verdict = decision["verdict"]
        if verdict not in ("sufficient", "partial", "insufficient"):
            raise ValueError(f"Unexpected verdict: {verdict!r}")
        notes = decision.get("notes", "")
    except (ValueError, KeyError):
        verdict, notes = "insufficient", "Could not parse evidence validator response."
    return {"evidence_verdict": verdict, "evidence_notes": notes}


# --- Tool schema + dispatch, for the Orchestrator -------------------------

RETRIEVE_EVIDENCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_evidence",
        "description": (
            "Search the cybersecurity knowledge base and grade whether the results answer "
            "the question. The Retriever agent may run several sub-searches internally to "
            "cover a multi-part question, so a phrase or a full multi-part objective both "
            "work as input. Use the original question the first time; on a retry, rewrite "
            "the objective to target what the previous result's notes said was missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search objective."}},
            "required": ["query"],
        },
    },
}


def call_retrieve_evidence(state: dict, query: str) -> tuple[dict, str, str | None, dict]:
    """Run Retriever then Evidence Validator.

    Returns:
        (state updates, tool result text for the model, terminal outcome —
        "refuse" to end the loop, or None to let the Orchestrator continue —
        and a trace entry describing this call for eval/debugging).
    """
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        skipped_trace = {"tool": "retrieve_evidence", "objective": query, "skipped": "retrieval budget exhausted"}
        if not state.get("retrieved"):
            return {}, "Retrieval budget exhausted with no usable evidence.", "refuse", skipped_trace
        return {}, json.dumps({
            "error": "retrieval budget exhausted, do not call retrieve_evidence again",
            "last_verdict": state.get("evidence_verdict"),
        }), None, skipped_trace

    retrieval_update = _run_retrieval(query, state.get("retrieval_attempts", 0), state.get("retrieved") or [])
    sub_calls = retrieval_update.pop("sub_calls")
    validation_update = _run_evidence_validation(state["query"], retrieval_update["retrieved"])
    updates = {**retrieval_update, **validation_update}

    terminal = None
    if validation_update["evidence_verdict"] == "insufficient" and updates["retrieval_attempts"] >= MAX_RETRIEVAL_ATTEMPTS:
        terminal = "refuse"

    result_text = json.dumps({
        "verdict": validation_update["evidence_verdict"],
        "notes": validation_update["evidence_notes"],
        "chunks_retrieved": len(retrieval_update["retrieved"]),
        "retrieval_attempts_used": f"{updates['retrieval_attempts']}/{MAX_RETRIEVAL_ATTEMPTS}",
    })
    trace_entry = {
        "tool": "retrieve_evidence",
        "objective": query,
        "retrieval_attempt": updates["retrieval_attempts"],
        "retriever_sub_calls": sub_calls,
        "chunks_retrieved": len(retrieval_update["retrieved"]),
        "retrieved_sources": [chunk["source"] for chunk in retrieval_update["retrieved"]],
        "evidence_verdict": validation_update["evidence_verdict"],
        "evidence_notes": validation_update["evidence_notes"],
    }
    return updates, result_text, terminal, trace_entry
