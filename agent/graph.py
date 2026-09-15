"""CyberQA multi-agent workflow.

Scope Gate -> Orchestrator, which drives a native LLM tool-calling loop
over Retriever and Answer Generator (retrieve_tool.py, generate_tool.py)
until it produces a grounded answer or gives up. See agent/README.md for
the diagram and the loop's internal structure.
"""

import json
import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.orchestrator import orchestrator_node, route_orchestrator
from llm.provider import generate


class CyberQAState(TypedDict, total=False):
    """State threaded through every node in the graph."""

    query: str

    # Ordered log of every node/tool-call decision, for tracing and eval.
    trace: list[dict]

    scope: Literal["in_scope", "ambiguous", "out_of_scope"]
    scope_reason: str

    orchestrator_action: Literal["answered", "refuse"]
    orchestrator_reason: str
    total_steps: int

    retrieved: list[dict]
    search_query: str
    retrieval_attempts: int
    evidence_verdict: Literal["sufficient", "partial", "insufficient"]
    evidence_notes: str

    draft_answer: str
    generation_attempts: int
    grounding_verdict: Literal["grounded", "unsupported"]
    unsupported_claims: list[str]
    cited_sources: list[dict]

    # Final result, set by refusal_node or compose_answer_node.
    answer: str | None
    message: str | None
    sources: list[dict]


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object in a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(match.group(0))


# --- Scope Gate: cheap pre-filter before the Orchestrator runs -----------

_SCOPE_GATE_PROMPT = """You are a fast pre-filter for a cybersecurity question-answering system whose
knowledge base is built entirely from MITRE ATT&CK technique data: techniques, detection
data components, mitigations, software, threat groups, and campaigns. The user's message may
be phrased as a direct question ("What mitigates T1110?") or as a short incident description
with no question mark at all ("A workstation began beaconing to a newly registered domain
after a user opened a macro-enabled attachment.") — treat both forms the same way: judge
scope by whether the knowledge base could plausibly speak to the behavior described or asked
about, not by sentence form. Your job is to catch messages that are obviously unrelated to
that domain — not to require security-sounding vocabulary or an explicit question.

Classify the user's message into exactly one scope:
- "in_scope": could plausibly be answered, or explained, from ATT&CK technique/detection/
  mitigation data. This includes questions or incident descriptions that don't say "attack"
  or "malicious" but reference a specific OS artifact, program, file, process, registry
  behavior, network behavior, or monitoring/detection method — these are normal ways a
  detection engineer or analyst asks about or reports technique behavior.
  Examples, all in_scope: "What accessibility programs can be run from the Ease of Access
  center?", "How can Process Creation be used to detect X?", "Our EDR flagged rundll32.exe
  loading a DLL right after a user clicked a phishing link.", "Multiple failed logins from
  an unusual country followed by a successful one for admin@corp.local."
- "ambiguous": plausibly relevant but too vague or underspecified to search for — a question
  or incident description that doesn't name or describe any specific technique, tool,
  system, artifact, or behavior at all (e.g. "How do I stop an attacker?", "Something seems
  wrong with our server.").
- "out_of_scope": clearly unrelated to attack techniques, detection, or mitigation —
  general chit-chat, unrelated general knowledge, or requests with no plausible connection
  to this domain (e.g. weather, recipes, creative writing, unrelated general tech support).

When in doubt between "in_scope" and "out_of_scope", prefer "in_scope" — this system's
Evidence Validator will catch it downstream if nothing relevant is actually retrievable.

Respond with only a JSON object:
{"scope": "in_scope"|"ambiguous"|"out_of_scope", "reason": "one short sentence"}
"""


def scope_gate_node(state: CyberQAState) -> dict:
    """Classify the query's scope before any retrieval or generation."""
    response = generate(state["query"], system=_SCOPE_GATE_PROMPT)
    try:
        decision = _parse_json(response)
        scope = decision["scope"]
        if scope not in ("in_scope", "ambiguous", "out_of_scope"):
            raise ValueError(f"Unexpected scope: {scope!r}")
        reason = decision.get("reason", "")
    except (ValueError, KeyError):
        scope, reason = "ambiguous", "Could not classify the query."
    trace = [{"node": "scope_gate", "scope": scope, "reason": reason}]
    return {"scope": scope, "scope_reason": reason, "trace": trace}


def route_scope(state: CyberQAState) -> str:
    """Send in-scope queries to the Orchestrator, everything else to refusal."""
    return "orchestrator" if state["scope"] == "in_scope" else "refusal"


# --- Terminal nodes --------------------------------------------------------

def refusal_node(state: CyberQAState) -> dict:
    """Assemble a message explaining why no answer is being returned.

    Reached directly from Scope Gate (out-of-scope/ambiguous), or from the
    Orchestrator once a retry budget is spent or it calls give_up.
    """
    if state.get("scope") == "out_of_scope":
        message = "This question isn't about cybersecurity, so it's out of scope for this system."
    elif state.get("scope") == "ambiguous":
        message = f"This question needs more detail before it can be answered: {state.get('scope_reason', '')}"
    elif state.get("evidence_verdict") == "insufficient":
        message = f"The knowledge base doesn't contain enough evidence to answer this. {state.get('evidence_notes', '')}"
    elif state.get("grounding_verdict") == "unsupported":
        message = (
            "A draft answer couldn't be fully grounded in the retrieved evidence "
            "after multiple attempts, so it's being withheld."
        )
    else:
        message = state.get("orchestrator_reason") or "This question could not be answered."

    return {
        "answer": None,
        "message": message.strip(),
        "sources": state.get("retrieved", []),
    }


def compose_answer_node(state: CyberQAState) -> dict:
    """Package a grounded draft answer with the sources it actually cites.

    Falls back to the full retrieved pool only if citation parsing found
    nothing (e.g. the model answered without bracketed citations) — should
    be rare, since the Grounding Verifier requires every claim be
    supported, but this keeps a grounded answer from surfacing with no
    sources at all.
    """
    return {
        "answer": state["draft_answer"],
        "message": None,
        "sources": state.get("cited_sources") or state.get("retrieved", []),
    }


# --- Graph wiring + entry points -------------------------------------------

def build_graph():
    """Build and compile the CyberQA multi-agent graph."""
    graph = StateGraph(CyberQAState)

    graph.add_node("scope_gate", scope_gate_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("compose_answer", compose_answer_node)

    graph.add_edge(START, "scope_gate")
    graph.add_conditional_edges(
        "scope_gate", route_scope, {"orchestrator": "orchestrator", "refusal": "refusal"}
    )
    graph.add_conditional_edges(
        "orchestrator", route_orchestrator, {"compose_answer": "compose_answer", "refusal": "refusal"}
    )
    graph.add_edge("refusal", END)
    graph.add_edge("compose_answer", END)

    return graph.compile()


def extract_result(raw: dict) -> dict:
    """Pull the user-facing fields out of a graph invocation's final state."""
    return {
        "answer": raw.get("answer"),
        "message": raw.get("message"),
        "sources": raw.get("sources", []),
    }


def ask(query: str) -> dict:
    """Run the graph on a single question and return the structured result.

    Builds a fresh graph per call; for a multi-turn session, build once with
    build_graph() and call app.invoke directly instead (see main.py).
    """
    app = build_graph()
    return extract_result(app.invoke({"query": query}, config={"recursion_limit": 50}))


