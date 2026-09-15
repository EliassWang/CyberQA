"""_format_evidence and _renumber_citations (agent/generate_tool.py): the numbered
evidence list an Answer Generator drafts from, and the citation-rewrite that maps a
draft's [n] markers to a dense 1..N sequence matching only the chunks it actually
used — the fix that keeps a response's Sources list and inline citations in sync,
instead of the citation numbers spanning the full retrieved pool.
"""

from agent.generate_tool import _format_evidence, _renumber_citations


def _chunk(source: str, text: str = "text") -> dict:
    return {"source": source, "text": text}


def test_format_evidence_numbers_from_one():
    chunks = [_chunk("a.txt", "first"), _chunk("b.txt", "second")]
    formatted = _format_evidence(chunks)
    assert "[1] (source: a.txt)\nfirst" in formatted
    assert "[2] (source: b.txt)\nsecond" in formatted


def test_renumber_citations_selects_only_cited_indices():
    chunks = [_chunk("a.txt"), _chunk("b.txt"), _chunk("c.txt")]
    draft = "Some claim [1]. Another claim [3]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "Some claim [1]. Another claim [2]."
    assert cited == [chunks[0], chunks[2]]


def test_renumber_citations_uses_first_citation_order_not_numeric_order():
    chunks = [_chunk("a.txt"), _chunk("b.txt"), _chunk("c.txt")]
    draft = "Claim citing [3] first, then [1]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "Claim citing [1] first, then [2]."
    assert cited == [chunks[2], chunks[0]]


def test_renumber_citations_dedupes_repeated_citations():
    chunks = [_chunk("a.txt"), _chunk("b.txt")]
    draft = "Claim [1]. Same source again [1]. Different [2]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "Claim [1]. Same source again [1]. Different [2]."
    assert cited == [chunks[0], chunks[1]]


def test_renumber_citations_ignores_out_of_range_numbers():
    chunks = [_chunk("a.txt")]
    draft = "Claim [1]. A hallucinated citation [7]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "Claim [1]. A hallucinated citation [7]."
    assert cited == [chunks[0]]


def test_renumber_citations_handles_adjacent_brackets():
    chunks = [_chunk("a.txt"), _chunk("b.txt"), _chunk("c.txt")]
    draft = "Multiple sources support this [1][2][3]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "Multiple sources support this [1][2][3]."
    assert cited == chunks


def test_renumber_citations_empty_when_no_citations():
    chunks = [_chunk("a.txt")]
    renumbered, cited = _renumber_citations("An answer with no citations at all.", chunks)
    assert renumbered == "An answer with no citations at all."
    assert cited == []


def test_renumber_citations_collapses_sparse_original_numbers():
    chunks = [_chunk(f"{i}.txt") for i in range(1, 21)]
    draft = "CosmicDuke [2][16], SeaDuke [4]."
    renumbered, cited = _renumber_citations(draft, chunks)
    assert renumbered == "CosmicDuke [1][2], SeaDuke [3]."
    assert cited == [chunks[1], chunks[15], chunks[3]]
