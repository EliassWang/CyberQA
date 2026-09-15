"""Answer-quality metrics for CTI-ATE -- different shape than score.py.

The subtask has a short, structured gold answer (a set of technique IDs)
rather than a reference sentence, so token F1/embedding-similarity
(benchmark.scripts.score) doesn't apply: a prose answer that correctly
names the techniques would score low on those metrics just for being
longer than the gold string. This scorer instead pulls the structured
answer back out of the agent's prose.
"""

import re


_TECHNIQUE_ID = re.compile(r"\bT(\d{4})(?:\.\d{3})?\b", re.IGNORECASE)


def score_ate(prediction: str | None, gt_techniques: list[str]) -> dict:
    """Set precision/recall/F1 over main ATT&CK technique IDs (sub-techniques folded into their parent).

    Extracts every `T####[.###]` occurrence anywhere in the prediction (the
    agent writes prose with inline citations, not CTIBench's own "last line
    only" format) and compares the set of main technique IDs against gt_techniques.
    """
    gold = {t.strip().upper() for t in gt_techniques}
    predicted = {f"T{match.group(1)}" for match in _TECHNIQUE_ID.finditer(prediction or "")}

    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = predicted & gold
    precision = len(overlap) / len(predicted)
    recall = len(overlap) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}
