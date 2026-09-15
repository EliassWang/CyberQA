"""Graph/orchestrator routing decisions — pure functions, no LLM or DB calls."""

from agent.graph import route_scope
from agent.orchestrator import route_orchestrator


def test_route_scope_in_scope_goes_to_orchestrator():
    assert route_scope({"scope": "in_scope"}) == "orchestrator"


def test_route_scope_ambiguous_goes_to_refusal():
    assert route_scope({"scope": "ambiguous"}) == "refusal"


def test_route_scope_out_of_scope_goes_to_refusal():
    assert route_scope({"scope": "out_of_scope"}) == "refusal"


def test_route_orchestrator_answered_goes_to_compose():
    assert route_orchestrator({"orchestrator_action": "answered"}) == "compose_answer"


def test_route_orchestrator_refuse_goes_to_refusal():
    assert route_orchestrator({"orchestrator_action": "refuse"}) == "refusal"
