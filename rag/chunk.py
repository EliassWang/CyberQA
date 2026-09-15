"""Split document text into overlapping token windows."""

import os

from transformers import PreTrainedTokenizerFast


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 200)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 40)


def chunk_text(text: str, tokenizer: PreTrainedTokenizerFast) -> list[str]:
    """Split text using the embedding model's tokenizer.

    Args:
        text: Extracted document text.
        tokenizer: Fast tokenizer used by the embedding model.

    Returns:
        Overlapping chunks preserving the original text and punctuation.
    """
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )["offset_mapping"]

    chunks = []
    for start in range(0, len(offsets), CHUNK_SIZE - CHUNK_OVERLAP):
        end = min(start + CHUNK_SIZE, len(offsets))
        chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(offsets):
            break
    return chunks
