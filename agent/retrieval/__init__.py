"""Query-time retrieval: dense + BM25 hybrid search with cross-encoder reranking.

Depends on rag.embedding for the shared embedding model config (the
ingestion side owns that). Importing rag here, before any submodule of
this package pulls in sentence_transformers, keeps rag/__init__.py's
HF/transformers env vars in effect regardless of which module in this
package is imported first — see rag/__init__.py for why that ordering
matters.
"""

import rag  # noqa: F401
