"""Orchestrator: the central control node, driven by native LLM tool-calling.

Runs a tool-calling loop against the LLM: the model picks retrieve_evidence,
generate_answer, or give_up each turn, and the result is fed back into the
conversation until it produces a grounded answer, gives up, or the step
budget runs out. Attempt budgets for the two tools are enforced in
retrieve_tool.py/generate_tool.py, not left to the model's judgment, so a
bad or repeated tool choice can't turn into an unbounded loop.
"""

import json

from agent.generate_tool import GENERATE_ANSWER_SCHEMA, call_generate_answer
from agent.retrieve_tool import RETRIEVE_EVIDENCE_SCHEMA, call_retrieve_evidence
from llm.provider import generate_with_tools, message_to_dict


MAX_TOTAL_STEPS = 6

GIVE_UP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "give_up",
        "description": "Report that the question cannot be answered from the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why it can't be answered."}},
            "required": ["reason"],
        },
    },
}

TOOL_SCHEMAS = [RETRIEVE_EVIDENCE_SCHEMA, GENERATE_ANSWER_SCHEMA, GIVE_UP_SCHEMA]

SYSTEM_PROMPT = """You are the Orchestrator of a cybersecurity question-answering system.
You must act by calling exactly one tool each turn:

- retrieve_evidence(query): search the knowledge base. Use the original question the
  first time. If a result's notes say evidence was partial or insufficient, rewrite
  the query to target what's missing — never repeat the same query verbatim.
- generate_answer(): draft an answer from evidence already retrieved and check it for
  grounding. Only call this once retrieve_evidence has returned a sufficient or
  partial verdict.
- give_up(reason): call this if the knowledge base clearly cannot answer the
  question, or a tool result tells you a budget is exhausted.

Never write a final answer directly — an answer only counts once generate_answer
reports it as grounded.
"""


def orchestrator_node(state: dict) -> dict:
    """Run the tool-calling loop until a grounded answer or a refusal is reached.

    Builds `trace`, an ordered log of every tool call this node makes —
    including the Retriever's own sub-searches and the Evidence
    Validator/Grounding Verifier verdicts nested under each call — on top of
    whatever scope_gate_node already logged, so the full pipeline's decision
    history survives into the final state for tracing/eval.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["query"]},
    ]
    updates: dict = {}
    trace: list[dict] = list(state.get("trace") or [])

    for step in range(1, MAX_TOTAL_STEPS + 1):
        message = generate_with_tools(messages, TOOL_SCHEMAS, tool_choice="required")
        messages.append(message_to_dict(message))

        if not message.tool_calls:
            trace.append({"node": "orchestrator", "step": step, "tool_call": None,
                          "stopped": "model returned no tool call"})
            return {**updates, "trace": trace, "total_steps": step, "orchestrator_action": "refuse",
                    "orchestrator_reason": message.content or "Model returned no tool call."}

        tool_call = message.tool_calls[0]
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        current_state = {**state, **updates}

        if name == "retrieve_evidence":
            call_updates, result_text, terminal, detail = call_retrieve_evidence(
                current_state, args.get("query", state["query"])
            )
        elif name == "generate_answer":
            call_updates, result_text, terminal, detail = call_generate_answer(current_state)
        elif name == "give_up":
            trace.append({"node": "orchestrator", "step": step, "tool_call": "give_up", "args": args})
            return {**updates, "trace": trace, "total_steps": step, "orchestrator_action": "refuse",
                    "orchestrator_reason": args.get("reason", "")}
        else:
            call_updates, result_text, terminal, detail = {}, f"Unknown tool {name!r}.", None, None

        updates.update(call_updates)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
        trace.append({"node": "orchestrator", "step": step, "tool_call": name, "args": args, "detail": detail})

        if terminal == "grounded":
            return {**updates, "trace": trace, "total_steps": step, "orchestrator_action": "answered",
                    "orchestrator_reason": "Grounded answer produced."}
        if terminal == "refuse":
            return {**updates, "trace": trace, "total_steps": step, "orchestrator_action": "refuse",
                    "orchestrator_reason": result_text}

    trace.append({"node": "orchestrator", "stopped": "step budget exhausted"})
    return {**updates, "trace": trace, "total_steps": MAX_TOTAL_STEPS, "orchestrator_action": "refuse",
            "orchestrator_reason": "Step budget exhausted."}


def route_orchestrator(state: dict) -> str:
    """Dispatch to the terminal node matching the loop's outcome."""
    return "compose_answer" if state["orchestrator_action"] == "answered" else "refusal"
