"""Free, local answer-quality metrics — no LLM calls.

Two complementary signals, combined because each covers the other's blind
spot: token F1 catches exact factual terms (technique IDs, file paths) an
embedding can blur past, while embedding similarity catches valid
paraphrases token overlap would unfairly penalize.
"""

import re
import string
from collections import Counter
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag.device import get_device
from rag.embedding import MODEL_NAME


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=get_device())


def _normalize(text: str) -> str:
    """Lowercase, drop articles/punctuation, collapse whitespace (SQuAD-style)."""
    text = _ARTICLES.sub(" ", text.lower())
    text = text.translate(_PUNCTUATION)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap precision/recall F1."""
    pred_tokens, ref_tokens = _normalize(prediction).split(), _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def embedding_similarity(prediction: str, reference: str) -> float:
    """Cosine similarity between sentence embeddings (same local model as retrieval)."""
    vectors = _model().encode([prediction, reference], normalize_embeddings=True)
    return float(vectors[0] @ vectors[1])


def score_answer(prediction: str | None, reference: str) -> dict:
    """Score a candidate answer against the reference answer.

    A missing prediction (the system refused) scores as a full miss on
    every metric — withholding an answer conveys none of the reference's
    content either.

    Returns:
        {"f1", "embedding_similarity"}.
    """
    if not prediction:
        return {"f1": 0.0, "embedding_similarity": 0.0}
    return {"f1": token_f1(prediction, reference), "embedding_similarity": embedding_similarity(prediction, reference)}
