"""agent.retrieval.bm25._tokenize: the lowercase, alphanumeric-only tokenizer
shared by corpus indexing and query search, so both sides must split text
identically.

agent.retrieval.bm25._id_tokens: the ATT&CK-ID extraction that
exact_id_hits uses to recover a specific relationship document among
hundreds of near-duplicate siblings that dense/BM25 top-N ranking can push
out of the candidate pool.
"""

from agent.retrieval.bm25 import _id_tokens, _tokenize


def test_lowercases_and_splits_on_non_alphanumeric():
    assert _tokenize("Steal Web Session Cookie") == ["steal", "web", "session", "cookie"]


def test_keeps_alphanumeric_technique_ids_together():
    assert _tokenize("T1539.001 sub-technique") == ["t1539", "001", "sub", "technique"]


def test_drops_punctuation_and_empty_tokens():
    assert _tokenize("brute-force, credential-stuffing!!") == ["brute", "force", "credential", "stuffing"]


def test_empty_string_yields_no_tokens():
    assert _tokenize("") == []


def test_id_tokens_finds_two_plain_ids():
    query = "How does attack software 'S0011: Taidoor' use attack technique 'T1057: Process Discovery'?"
    assert _id_tokens(query) == {"s0011", "t1057"}


def test_id_tokens_splits_dotted_sub_technique_id():
    query = "How does attack software 'S0534: Bazar' use attack technique 'T1547.004: Winlogon Helper DLL'?"
    assert _id_tokens(query) == {"s0534", "t1547", "004"}


def test_id_tokens_empty_for_a_single_id():
    assert _id_tokens("What is 'T1057: Process Discovery'?") == set()


def test_id_tokens_empty_with_no_ids():
    assert _id_tokens("What is process discovery?") == set()
