"""The `generate_answer` tool: Answer Generator + Grounding Verifier.

Bundled into one tool so a hallucinated draft can never reach the
Orchestrator without first being checked — only a grounded result is
usable as a final answer — and so the retry budget below is enforced here
in code, not left to the model's judgment.
"""

import json
import re

from llm.provider import generate


MAX_GENERATION_ATTEMPTS = 2


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object in a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group(0))


def _format_evidence(chunks: list[dict]) -> str:
    """Render chunks as a numbered list, matching the citation markers agents use."""
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1))


def _renumber_citations(draft_answer: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """Rewrite [n] markers to a dense 1..N sequence in first-citation order,
    and return the chunks that sequence points to.

    Citation numbers in `draft_answer` are 1-indexed positions into
    `chunks`, the same numbering `_format_evidence` gave the Answer
    Generator — so [3] means chunks[2]. That numbering is sparse (it spans
    the full retrieved pool, not just what got cited), so left as-is it
    doesn't match a Sources list of only the cited chunks. Renumbering here
    makes the text's [1][2][3]... line up exactly with that list. A number
    outside range (a rare model slip) is left untouched in the text and
    contributes no chunk, since it doesn't refer to real evidence.
    """
    mapping: dict[int, int] = {}
    cited: list[dict] = []

    def replace(match: re.Match) -> str:
        i = int(match.group(1))
        if not (1 <= i <= len(chunks)):
            return match.group(0)
        if i not in mapping:
            mapping[i] = len(cited) + 1
            cited.append(chunks[i - 1])
        return f"[{mapping[i]}]"

    renumbered = re.sub(r"\[(\d+)\]", replace, draft_answer)
    return renumbered, cited


# --- Answer Generator ----------------------------------------------------

_ANSWER_GENERATOR_PROMPT = """You answer cybersecurity questions using ONLY the numbered
evidence excerpts provided. Cite the excerpt number in brackets, like [1] or
[2][3], immediately after each claim it supports. Do not use any knowledge
beyond the excerpts. If the excerpts only partially answer the question,
answer what they support and say what's missing — do not fill gaps with
unsupported claims.
"""


def _run_answer_generation(
    query: str, chunks: list[dict], prior_attempts: int, unsupported_claims: list[str] | None
) -> dict:
    """Draft an answer, incorporating feedback from a prior failed grounding check."""
    prompt = f"Question: {query}\n\nEvidence:\n{_format_evidence(chunks)}"
    if unsupported_claims:
        prompt += (
            "\n\nYour previous answer made claims the evidence didn't support:\n"
            + "\n".join(f"- {c}" for c in unsupported_claims)
            + "\nFix or remove these; cite only what the excerpts actually say."
        )
    answer = generate(prompt, system=_ANSWER_GENERATOR_PROMPT)
    return {"draft_answer": answer, "generation_attempts": prior_attempts + 1}


# --- Grounding Verifier ----------------------------------------------------

_GROUNDING_VERIFIER_PROMPT = """You fact-check a draft answer against the numbered evidence
excerpts it was written from. Break the draft into its individual factual
claims and check each one against the excerpts. A claim is unsupported if the
excerpts don't state it, even if it sounds plausible.

Respond with only a JSON object:
{"verdict": "grounded"|"unsupported", "unsupported_claims": ["claim text", ...]}
Use "grounded" with an empty list only if every claim is directly supported.
"""


def _run_grounding_verification(query: str, chunks: list[dict], draft_answer: str) -> dict:
    """Verify the draft answer's claims are each supported by cited evidence."""
    prompt = f"Question: {query}\n\nEvidence:\n{_format_evidence(chunks)}\n\nDraft answer:\n{draft_answer}"
    response = generate(prompt, system=_GROUNDING_VERIFIER_PROMPT)
    try:
        decision = _parse_json(response)
        verdict = decision["verdict"]
        if verdict not in ("grounded", "unsupported"):
            raise ValueError(f"Unexpected verdict: {verdict!r}")
        unsupported_claims = decision.get("unsupported_claims", [])
    except (ValueError, KeyError):
        verdict, unsupported_claims = "unsupported", ["Could not parse grounding verifier response."]
    return {"grounding_verdict": verdict, "unsupported_claims": unsupported_claims}


# --- Tool schema + dispatch, for the Orchestrator -------------------------

GENERATE_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_answer",
        "description": (
            "Draft an answer from the evidence already retrieved and check it for "
            "grounding. Only call this once retrieve_evidence has returned a "
            "sufficient or partial verdict."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def call_generate_answer(state: dict) -> tuple[dict, str, str | None, dict]:
    """Run Answer Generator then Grounding Verifier.

    Returns:
        (state updates, tool result text for the model, terminal outcome —
        "grounded" to end the loop with an answer, "refuse" to end it
        without one, or None to let the Orchestrator continue — and a trace
        entry describing this call for eval/debugging).
    """
    if not state.get("retrieved"):
        skipped_trace = {"tool": "generate_answer", "skipped": "no evidence retrieved yet"}
        return {}, "No evidence retrieved yet; call retrieve_evidence first.", None, skipped_trace
    if state.get("generation_attempts", 0) >= MAX_GENERATION_ATTEMPTS:
        skipped_trace = {"tool": "generate_answer", "skipped": "generation budget exhausted"}
        return {}, "Generation budget exhausted.", "refuse", skipped_trace

    generation_update = _run_answer_generation(
        state["query"], state["retrieved"], state.get("generation_attempts", 0), state.get("unsupported_claims")
    )
    verification_update = _run_grounding_verification(
        state["query"], state["retrieved"], generation_update["draft_answer"]
    )
    renumbered_answer, cited_sources = _renumber_citations(generation_update["draft_answer"], state["retrieved"])
    updates = {
        **generation_update,
        **verification_update,
        "draft_answer": renumbered_answer,
        "cited_sources": cited_sources,
    }

    if verification_update["grounding_verdict"] == "grounded":
        terminal = "grounded"
    elif updates["generation_attempts"] >= MAX_GENERATION_ATTEMPTS:
        terminal = "refuse"
    else:
        terminal = None

    result_text = json.dumps({
        "verdict": verification_update["grounding_verdict"],
        "unsupported_claims": verification_update["unsupported_claims"],
        "generation_attempts_used": f"{updates['generation_attempts']}/{MAX_GENERATION_ATTEMPTS}",
    })
    trace_entry = {
        "tool": "generate_answer",
        "generation_attempt": updates["generation_attempts"],
        "draft_answer": renumbered_answer,
        "grounding_verdict": verification_update["grounding_verdict"],
        "unsupported_claims": verification_update["unsupported_claims"],
        "cited_sources": [c["source"] for c in cited_sources],
    }
    return updates, result_text, terminal, trace_entry
