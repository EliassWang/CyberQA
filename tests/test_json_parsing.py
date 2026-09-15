"""_parse_json: the regex-based "extract the first {...} block" helper that
agent.graph, agent.retrieve_tool, and agent.generate_tool each keep their
own private copy of, to tolerate a model wrapping its JSON reply in prose.
Parametrized across all three copies as a regression net against them
silently diverging.
"""

import pytest

import agent.generate_tool as generate_tool
import agent.graph as graph
import agent.retrieve_tool as retrieve_tool

PARSE_JSON_IMPLS = [graph._parse_json, retrieve_tool._parse_json, generate_tool._parse_json]


@pytest.mark.parametrize("parse_json", PARSE_JSON_IMPLS)
def test_parses_bare_json_object(parse_json):
    assert parse_json('{"scope": "in_scope"}') == {"scope": "in_scope"}


@pytest.mark.parametrize("parse_json", PARSE_JSON_IMPLS)
def test_parses_json_wrapped_in_prose(parse_json):
    text = 'Sure, here you go:\n{"verdict": "sufficient", "notes": "ok"}\nHope that helps!'
    assert parse_json(text) == {"verdict": "sufficient", "notes": "ok"}


@pytest.mark.parametrize("parse_json", PARSE_JSON_IMPLS)
def test_parses_json_in_markdown_fence(parse_json):
    text = '```json\n{"verdict": "grounded", "unsupported_claims": []}\n```'
    assert parse_json(text) == {"verdict": "grounded", "unsupported_claims": []}


@pytest.mark.parametrize("parse_json", PARSE_JSON_IMPLS)
def test_no_json_object_raises_value_error(parse_json):
    with pytest.raises(ValueError):
        parse_json("I cannot help with that.")


@pytest.mark.parametrize("parse_json", PARSE_JSON_IMPLS)
def test_malformed_json_raises_value_error(parse_json):
    # json.JSONDecodeError is a ValueError subclass; _parse_json doesn't
    # catch it, so a matched-but-invalid block propagates as-is.
    with pytest.raises(ValueError):
        parse_json('{"scope": "in_scope",}')
