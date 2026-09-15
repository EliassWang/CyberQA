"""agent.retrieval.hybrid._normalize: min-max scaling of one signal's scores
to [0, 1] before combining dense + BM25 — the step that keeps a chunk only
one signal surfaced from being penalized further just for missing the
other's pool.
"""

from agent.retrieval.hybrid import _normalize


def test_empty_input_returns_empty():
    assert _normalize({}) == {}


def test_min_maps_to_zero_and_max_to_one():
    scores = {"a": 1.0, "b": 3.0, "c": 5.0}
    normalized = _normalize(scores)
    assert normalized["a"] == 0.0
    assert normalized["c"] == 1.0
    assert normalized["b"] == 0.5


def test_flat_scores_all_map_to_one():
    scores = {"a": 2.0, "b": 2.0, "c": 2.0}
    assert _normalize(scores) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_single_score_maps_to_one():
    assert _normalize({"only": 0.42}) == {"only": 1.0}
