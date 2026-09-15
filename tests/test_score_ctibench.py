"""benchmark.scripts.score_ctibench: set-based technique scoring (CTI-ATE) — pure, no model calls."""

from benchmark.scripts.score_ctibench import score_ate


def test_score_ate_exact_match():
    result = score_ate("Uses T1055 and T1110 per the report.", ["T1055", "T1110"])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_score_ate_folds_subtechniques_into_parent():
    result = score_ate("Uses T1110.001 (Password Guessing).", ["T1110"])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_score_ate_partial_overlap():
    result = score_ate("Uses T1055 and T1027.", ["T1055", "T1110"])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_score_ate_no_predicted_techniques_scores_zero():
    assert score_ate("No technique IDs mentioned here.", ["T1055"]) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_score_ate_none_prediction_scores_zero():
    assert score_ate(None, ["T1055"]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
