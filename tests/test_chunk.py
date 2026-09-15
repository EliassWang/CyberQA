"""rag.chunk.chunk_text: 200-token windows with 40-token overlap, split using
the actual embedding-model tokenizer (offsets must line up with real text,
not an approximation) — so this needs the real tokenizer, unlike the rest
of this suite. Skipped if it isn't already cached locally and no network is
available to fetch it.
"""

import pytest

from rag.chunk import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        from rag.embedding import MODEL_NAME

        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:  # offline with no local cache of the model
        pytest.skip(f"embedding tokenizer unavailable: {exc}")


def test_short_text_yields_a_single_chunk(tokenizer):
    text = "Adversaries may steal web session cookies to bypass authentication."
    chunks = chunk_text(text, tokenizer)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_is_split_into_overlapping_chunks(tokenizer):
    # Comfortably longer than CHUNK_SIZE tokens so it must split at least twice.
    text = " ".join(f"word{i}" for i in range(3 * CHUNK_SIZE))
    chunks = chunk_text(text, tokenizer)
    assert len(chunks) > 1
    # Consecutive chunks should share overlapping content (the last words of
    # one chunk reappear at the start of the next).
    first_tail_word = chunks[0].split()[-1]
    assert first_tail_word in chunks[1]


def test_empty_text_yields_no_chunks(tokenizer):
    assert chunk_text("", tokenizer) == []


def test_chunk_overlap_is_smaller_than_chunk_size():
    # Sanity check on the module constants themselves: a non-positive stride
    # (CHUNK_SIZE - CHUNK_OVERLAP) would make chunk_text loop forever.
    assert 0 < CHUNK_OVERLAP < CHUNK_SIZE
