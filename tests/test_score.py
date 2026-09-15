"""benchmark.scripts.score: token_f1 is pure (no model call); score_answer's
missing-prediction short-circuit is also pure. embedding_similarity needs
the local Sentence-BERT model, so it's out of scope for this fast suite.
"""

from benchmark.scripts.score import score_answer, token_f1


def test_identical_strings_score_f1_one():
    assert token_f1("Multi-factor Authentication", "Multi-factor Authentication") == 1.0


def test_disjoint_strings_score_f1_zero():
    assert token_f1("Multi-factor Authentication", "Password Policies") == 0.0


def test_partial_overlap_scores_between_zero_and_one():
    f1 = token_f1("Account Use Policies and Password Policies", "Password Policies")
    assert 0.0 < f1 < 1.0


def test_ignores_case_articles_and_punctuation():
    assert token_f1("The Brute Force Attack", "brute force, attack") == 1.0


def test_both_empty_after_normalization_scores_one():
    # "the" is stripped as an article by _normalize; both sides go empty.
    assert token_f1("the", "the") == 1.0


def test_one_sided_empty_scores_zero():
    assert token_f1("the", "Password Policies") == 0.0


def test_missing_prediction_scores_full_miss():
    assert score_answer(None, "Multi-factor Authentication") == {"f1": 0.0, "embedding_similarity": 0.0}


def test_empty_string_prediction_scores_full_miss():
    assert score_answer("", "Multi-factor Authentication") == {"f1": 0.0, "embedding_similarity": 0.0}
